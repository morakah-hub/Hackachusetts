"""Tests for v0.3 Cartesian reaching: kinematics, IK, collision checks, reach task, closed loop."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robots.panda import HOME_QPOS  # noqa: E402
from simulation.engine import PROJECT_ROOT, SimulationEngine  # noqa: E402
from simulation.kinematics import ArmKinematics  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402
from tasks.reach import CartesianReachTask  # noqa: E402

REACH_CONFIG = PROJECT_ROOT / "configs" / "reach.yaml"


@pytest.fixture(scope="module")
def reach_config():
    return load_config(REACH_CONFIG)


@pytest.fixture
def kin():
    return ArmKinematics(SimulationEngine("table"))


def with_controller(cfg, kind):
    return {**cfg, "controller": {**cfg["controller"], "type": kind}}


# --- Kinematics ---------------------------------------------------------------


def test_fk_matches_engine_and_jacobian_matches_finite_differences(kin):
    q = np.array([0.3, -0.2, 0.1, -1.8, 0.2, 1.9, 0.4])
    np.testing.assert_allclose(kin.fk(q), kin.engine.forward_kinematics(q), atol=1e-12)
    J = kin.jacobian(q)
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = 1e-6
        np.testing.assert_allclose((kin.fk(q + dq) - kin.fk(q - dq)) / 2e-6, J[:, j], atol=1e-6)


def test_ik_converges_within_joint_limits(kin):
    target = np.array([0.4, 0.35, 0.6])
    res = kin.solve_ik(target, HOME_QPOS)
    assert res.success and res.error < 1e-4
    np.testing.assert_allclose(kin.fk(res.q), target, atol=1e-4)
    assert np.all(res.q >= kin.q_min - 1e-12) and np.all(res.q <= kin.q_max + 1e-12)


def test_ik_respects_limits_when_target_pulls_past_them(kin):
    """A target behind the robot needs joint 1 beyond +-2.9 rad from the home side; the
    clamped solution must stay inside the margin even when it fails."""
    res = kin.solve_ik(np.array([-0.5, 0.05, 0.6]), HOME_QPOS)
    assert np.all(res.q >= kin.q_min - 1e-12) and np.all(res.q <= kin.q_max + 1e-12)


def test_unreachable_target_rejected(kin):
    res = kin.plan_target(np.array([1.3, 0.0, 0.6]), HOME_QPOS)
    assert not res.success and "did not converge" in res.reason
    assert res.error > 0.3


def test_target_inside_table_rejected_by_collision_check(kin):
    res = kin.plan_target(np.array([0.5, 0.1, 0.38]), HOME_QPOS)
    assert not res.success and "collision at goal" in res.reason and "table" in res.reason


def test_collision_on_path_rejected_without_planning_around():
    """Both goals are collision-free, but the straight joint path between them
    sweeps the hand through an obstacle. Collision checking rejects it; no
    obstacle-avoiding planner exists (documented limitation)."""
    kin = ArmKinematics(SimulationEngine("obstacles"))
    a = kin.plan_target(np.array([0.68, 0.42, 0.52]), HOME_QPOS)
    assert a.success
    b_from_home = kin.plan_target(np.array([0.70, 0.18, 0.52]), HOME_QPOS)
    assert b_from_home.success
    b_from_a = kin.plan_target(np.array([0.70, 0.18, 0.52]), a.q)
    assert not b_from_a.success and "collision on path" in b_from_a.reason


def test_kinematics_never_touch_live_state(kin):
    engine = kin.engine
    before = engine.data.qpos.copy(), engine.data.qvel.copy(), engine.data.ctrl.copy()
    kin.plan_target(np.array([0.45, -0.3, 0.85]), HOME_QPOS)
    kin.contacts(np.zeros(7))
    for a, b in zip(before, (engine.data.qpos, engine.data.qvel, engine.data.ctrl)):
        np.testing.assert_array_equal(a, b)


# --- Reach task ------------------------------------------------------------------


def test_reach_task_skips_rejected_targets(kin):
    targets = [("a", [0.5, 0.0, 0.7]), ("far", [1.3, 0.0, 0.6]), ("b", [0.45, -0.3, 0.85])]
    task = CartesianReachTask(targets, HOME_QPOS, plan=kin.plan_target, move_time=2.0, hold_time=1.0)
    assert [p.name for p in task.reachable] == ["a", "b"]
    assert [p.name for p in task.rejected] == ["far"]
    assert len(task.waypoints) == 3 and task.num_segments == 2
    np.testing.assert_allclose(task.reference(0.0).q, HOME_QPOS)
    b = task.reachable[1]
    np.testing.assert_allclose(kin.fk(task.reference(task.hold_end_time(b) - 0.1).q), b.position, atol=1e-4)
    assert task.reference(1.0).qdd is not None  # acceleration feedforward available


def test_reach_task_with_no_reachable_target_holds_start(kin):
    task = CartesianReachTask([("far", [1.3, 0.0, 0.6])], HOME_QPOS, plan=kin.plan_target)
    assert not task.reachable
    for t in (0.0, 2.0, 10.0):
        np.testing.assert_allclose(task.reference(t).q, HOME_QPOS)


def test_reach_task_rejects_bad_target_shape(kin):
    with pytest.raises(ValueError, match="x, y, z"):
        CartesianReachTask([("bad", [0.5, 0.0])], HOME_QPOS, plan=kin.plan_target)


# --- Configuration and closed loop -------------------------------------------------


def test_reach_config_extends_default(reach_config):
    default = load_config()
    assert reach_config["controller"] == default["controller"]
    assert reach_config["task"]["type"] == "cartesian_reach"
    assert "waypoints" not in reach_config["task"]  # task replaced, not merged
    names = [t["name"] for t in reach_config["task"]["targets"]]
    assert len(names) >= 4 and "out_of_reach" in names


@pytest.mark.parametrize("kind", ["joint_pid", "torque_mpc"])
def test_closed_loop_reaches_configured_targets(reach_config, kind):
    r = SimulationRunner(with_controller(reach_config, kind))
    task = r.task
    assert len(task.reachable) == 3
    assert {p.name for p in task.rejected} == {"out_of_reach", "inside_table"}
    log = r.run_headless(task.duration, log_every=1).arrays()
    assert np.all(np.isfinite(log["q"]))
    for p in task.reachable:
        i = np.searchsorted(log["time"], task.hold_end_time(p)) - 1
        assert np.linalg.norm(log["ee"][i] - p.position) < 1e-3  # 1 mm
    assert r.engine.robot_contacts() == []
    # The marker shows the requested Cartesian goal while moving to it.
    first = task.reachable[0]
    np.testing.assert_allclose(r.waypoint_ee[first.waypoint_index], first.position)
