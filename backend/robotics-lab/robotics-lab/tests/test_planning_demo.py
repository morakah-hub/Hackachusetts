"""v0.5 integration: the obstacle-avoidance demo, planned references, closed-loop execution."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker  # noqa: E402
from simulation.engine import PROJECT_ROOT  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402
from tasks.joint_target import JointTargetTask  # noqa: E402
from tasks.planned_joint import PlannedJointTask, PlanningFailed  # noqa: E402
from tasks.reach import CartesianReachTask  # noqa: E402
from tasks.trajectory import check_reference  # noqa: E402

DEMO_CONFIG = PROJECT_ROOT / "configs" / "planning_demo.yaml"
LEFT = [0.246, 0.621, 0.211, -1.791, 0.089, 1.756, -0.785]     # demo start (EE at y = +0.25)
RIGHT = [-0.246, 0.621, -0.211, -1.791, -0.089, 1.756, -0.785]  # its mirror image (EE at y = -0.25)


@pytest.fixture(scope="module")
def demo_config():
    return load_config(DEMO_CONFIG)


def with_controller(cfg, kind, **task_changes):
    cfg = copy.deepcopy(cfg)
    cfg["controller"]["type"] = kind
    cfg["task"].update(task_changes)
    return cfg


@pytest.fixture(scope="module")
def ctc_runner(demo_config):
    return SimulationRunner(with_controller(demo_config, "torque_ctc"))


def run_with_contacts(runner, duration=None):
    """Headless run counting physics steps in which the simulated robot touches anything."""
    steps = []

    def count(r):
        if r.engine.data.ncon and r.engine.robot_contacts():
            steps.append(r.engine.time)

    log = runner.run_headless(duration or runner.task.duration, log_every=1, before_step=count).arrays()
    return log, len(steps)


# --- The saved demonstration scenario -------------------------------------------------


def test_demo_scenario_is_valid_and_needs_planning(ctc_runner):
    """Start and goals collision-free with clearance; IK reaches them; the straight joint
    move from start to each goal hits the wall, the planned path does not."""
    task, planner = ctc_runner.task, ctc_runner.planner
    assert isinstance(task, CartesianReachTask) and planner is not None
    assert planner.checker.check(np.array(LEFT)).valid
    assert [p.name for p in task.reachable] == ["right_of_wall", "back_left"] and not task.rejected
    q_from = np.array(LEFT)
    for p in task.plans:
        assert p.status == "success" and p.stats["direct_edge_valid"] is False
        assert not planner.checker.edge_valid(q_from, p.q)  # direct route blocked
        assert planner.checker.path_valid(p.path)[0]        # planned route valid
        assert planner.kin.fk(p.q) == pytest.approx(p.position, abs=1e-4)
        assert len(p.path) > 2
        q_from = p.q


def test_planned_reference_is_continuous_feasible_and_collision_free(ctc_runner):
    runner = ctc_runner
    task, robot = runner.task, runner.engine.robot
    t = np.arange(0.0, task.duration + 0.5, 0.002)
    refs = [task.reference(x) for x in t]
    q = np.array([r.q for r in refs])
    qd = np.array([r.qd for r in refs])
    qdd = np.array([r.qdd for r in refs])
    limits = runner.trajectory_options(runner.config["task"]["trajectory"])["limits"]
    rep = check_reference(t, q, qd, qdd, limits)
    assert rep.feasible, rep.violations
    assert np.abs(np.diff(q, axis=0)).max() <= np.max(robot.velocity_limits) * 0.002  # no jumps
    np.testing.assert_allclose(q[0], LEFT)
    np.testing.assert_allclose(qd[0], 0.0)
    # The reference passes through every path vertex and stops there.
    vertices = [v for p in task.plans for v in p.path]
    for v in vertices:
        assert np.min(np.abs(q - v).max(axis=1)) < 1e-9
    # Every reference configuration keeps >= clearance - cartesian_step / 2 from the environment.
    s = runner.planner.settings
    monitor = CollisionChecker.from_engine(runner.engine, clearance=0.05)
    dmin = min(monitor.min_distance(x) for x in q[::5])
    assert dmin >= s.clearance - s.cartesian_step / 2 - 1e-6
    assert not any(monitor.in_collision(x) and monitor.min_distance(x) <= 0 for x in q[::25])


def test_planned_demo_executes_without_contacts_ctc(ctc_runner):
    runner = ctc_runner
    runner.reset()
    log, contact_steps = run_with_contacts(runner)
    assert contact_steps == 0
    lo, hi = runner.engine.joint_limits[:, 0], runner.engine.joint_limits[:, 1]
    assert np.all((log["q"] >= lo) & (log["q"] <= hi))
    assert np.all(np.abs(log["qd"]) <= np.asarray(runner.engine.robot.velocity_limits))
    for p in runner.task.reachable:
        i = np.searchsorted(log["time"], runner.task.hold_end_time(p)) - 1
        assert np.linalg.norm(runner.engine.forward_kinematics(log["q"][i]) - p.position) < 1e-3


def test_planned_demo_executes_with_torque_mpc(demo_config):
    runner = SimulationRunner(with_controller(demo_config, "torque_mpc"))
    log, contact_steps = run_with_contacts(runner)
    assert contact_steps == 0
    stats = runner.controller.stats.summary(runner.controller.control_period)
    assert stats["fallbacks"] == 0
    assert np.abs(log["q"] - log["q_ref"]).max() < 0.01
    for p in runner.task.reachable:
        i = np.searchsorted(log["time"], runner.task.hold_end_time(p)) - 1
        assert np.linalg.norm(runner.engine.forward_kinematics(log["q"][i]) - p.position) < 1e-3


def test_planning_disabled_rejects_blocked_target(demo_config):
    cfg = with_controller(demo_config, "torque_ctc")
    cfg["task"]["planning"]["enabled"] = False
    runner = SimulationRunner(cfg)
    assert runner.planner is None
    task = runner.task
    assert [p.name for p in task.rejected] == ["right_of_wall"]
    assert "collision on path" in task.rejected[0].reason
    assert task.planned_paths == []


def test_unchecked_direct_mode_really_collides(demo_config):
    """The comparison mode without any path check drives into the wall; the contact
    counter used for the planned runs does detect it."""
    cfg = with_controller(demo_config, "torque_ctc")
    cfg["task"]["planning"]["enabled"] = False
    cfg["task"]["ik"]["check_path"] = False
    cfg["task"]["trajectory"]["torque_scale"] = None
    runner = SimulationRunner(cfg)
    assert not runner.task.rejected
    _, contact_steps = run_with_contacts(runner, duration=runner.task.segment_bounds(0)[2])
    assert contact_steps > 0


# --- Planned joint-waypoint tasks -----------------------------------------------------


def joint_cfg(demo_config, waypoints, planning=True, loop=True, env="planning_demo"):
    cfg = with_controller(demo_config, "torque_ctc")
    cfg["environment"] = env
    cfg["task"] = {"type": "joint_target", "move_time": 2.0, "hold_time": 0.5, "loop": loop,
                   "waypoints": waypoints, "trajectory": demo_config["task"]["trajectory"],
                   "planning": {**demo_config["task"]["planning"], "enabled": planning}}
    return cfg


def test_planned_joint_task_goes_around_wall_and_loops(demo_config):
    runner = SimulationRunner(joint_cfg(demo_config, {"left": LEFT, "right": RIGHT}))
    task = runner.task
    assert isinstance(task, PlannedJointTask) and len(task.plans) == 2  # left->right and the loop back
    assert len(task.waypoints) > 2
    assert task.key_index[0] == 0 and len(task.key_index) == 2
    np.testing.assert_allclose(task.waypoints[task.key_index[1]], RIGHT)
    # Via points show the key waypoint they lead to; key waypoints show themselves.
    assert task.goal_index[task.key_index[1]] == task.key_index[1]
    assert all(g in task.key_index for g in task.goal_index)
    # Loop: the reference returns exactly to the start after one pass.
    np.testing.assert_allclose(task.reference(task.duration).q, LEFT, atol=1e-12)
    np.testing.assert_allclose(task.reference(0.0).q, LEFT)
    for plan in task.plans:
        assert runner.planner.checker.path_valid(plan.path)[0]
    log, contact_steps = run_with_contacts(runner)
    assert contact_steps == 0


def test_planning_on_free_joint_moves_matches_unplanned_task(demo_config):
    """Default table waypoints have free straight moves: the planner returns the direct
    edges and the reference is identical to the v0.4 joint task."""
    wps = load_config()["task"]["waypoints"]
    planned = SimulationRunner(joint_cfg(demo_config, wps, planning=True, env="table")).task
    plain = SimulationRunner(joint_cfg(demo_config, wps, planning=False, env="table")).task
    assert isinstance(plain, JointTargetTask)
    assert all(len(p.path) == 2 for p in planned.plans)
    for t in np.linspace(0, plain.duration, 97):
        a, b = planned.reference(t), plain.reference(t)
        np.testing.assert_allclose(a.q, b.q, atol=1e-12)
        np.testing.assert_allclose(a.qd, b.qd, atol=1e-12)


def test_unplannable_joint_move_raises(demo_config):
    cfg = joint_cfg(demo_config, {"left": LEFT, "right": RIGHT}, loop=False)
    cfg["task"]["planning"]["rrt"] = {"max_nodes": 4}
    with pytest.raises(PlanningFailed, match="left -> right: node_budget"):
        SimulationRunner(cfg)


def test_runner_draws_planned_paths_headless(ctc_runner):
    import mujoco

    scene = mujoco.MjvScene(ctc_runner.engine.model, maxgeom=3000)
    assert ctc_runner.draw_planned_paths(scene) > 0


# --- CLI ---------------------------------------------------------------------------------


@pytest.mark.parametrize("flags, expect", [
    ([], "direct edge BLOCKED -> RRT-Connect"),
    (["--no-planning"], "REJECTED (collision on path"),
])
def test_cli_headless_demo(flags, expect):
    out = subprocess.run(
        [sys.executable, "main.py", "--config", str(DEMO_CONFIG), "--controller", "ctc", "--headless",
         "--duration", "0.5", *flags],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert expect in out.stdout
    assert "Physics steps with robot contacts during the run: 0" in out.stdout
