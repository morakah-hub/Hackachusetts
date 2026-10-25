"""Tests for Robotics Lab v0.1: model assets, scene composition, task, controller, closed loop."""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.pid import JointPIDController  # noqa: E402
from robots.panda import HOME_QPOS, MODEL_DIR, MODEL_XML, Panda  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402
from tasks.joint_target import JointTargetTask, min_jerk  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def runner(config):
    return SimulationRunner(config)


# --- Model and assets -------------------------------------------------------


def test_menagerie_model_files_present():
    assert MODEL_XML.is_file()
    assert (MODEL_DIR / "LICENSE").is_file()
    assert "google-deepmind/mujoco_menagerie" in (MODEL_DIR / "SOURCE.txt").read_text()
    mesh_files = re.findall(r'file="([^"]+)"', MODEL_XML.read_text())
    assert mesh_files, "no mesh references found"
    missing = [f for f in mesh_files if not (MODEL_DIR / "assets" / f).is_file()]
    assert not missing, f"missing mesh files: {missing}"


def test_panda_actuators_are_position_servos():
    model = Panda().load_spec().compile()
    assert model.nu == 8
    for i, name in enumerate(Panda().arm_actuators):
        act = model.actuator(name)
        assert model.jnt_type[act.trnid[0]] == mujoco.mjtJoint.mjJNT_HINGE
        assert model.joint(act.trnid[0]).name == f"joint{i + 1}"
        assert model.actuator_biastype[act.id] == mujoco.mjtBias.mjBIAS_AFFINE
        kp = model.actuator_gainprm[act.id, 0]
        assert kp > 0 and model.actuator_biasprm[act.id, 1] == pytest.approx(-kp)


@pytest.mark.parametrize("environment", ["table", "empty", "obstacles"])
def test_environments_compile_with_robot(environment):
    engine = SimulationEngine(environment)
    assert engine.model.nq == 9  # 7 arm joints + 2 finger joints
    assert engine.model.nu == 8
    assert engine.target_mocap_id >= 0
    np.testing.assert_allclose(engine.arm_qpos(), HOME_QPOS)


def test_scene_contains_floor_table_and_target():
    model = SimulationEngine("table").model
    for geom in ("floor", "table_top", "target_marker"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0
    marker = model.geom("target_marker")
    assert model.geom_contype[marker.id] == 0 and model.geom_conaffinity[marker.id] == 0


def test_offscreen_render_produces_image():
    engine = SimulationEngine("table")
    try:
        renderer = mujoco.Renderer(engine.model, 240, 320)
    except Exception as exc:  # no OpenGL context available on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    with renderer:
        renderer.update_scene(engine.data)
        image = renderer.render()
    assert image.shape == (240, 320, 3)
    assert image.std() > 5  # not a blank frame


# --- Task and controller units ----------------------------------------------


def test_min_jerk_boundary_conditions():
    assert min_jerk(0.0) == (0.0, 0.0)
    assert min_jerk(1.0) == pytest.approx((1.0, 0.0))
    assert min_jerk(0.5)[0] == pytest.approx(0.5)


def test_joint_target_task_reference(config):
    wps = list(config["task"]["waypoints"].values())
    task = JointTargetTask(wps, move_time=2.0, hold_time=1.0)
    np.testing.assert_allclose(task.reference(0.0).q, wps[0])
    np.testing.assert_allclose(task.reference(2.5).q, wps[1])  # holding at waypoint 1
    np.testing.assert_allclose(task.reference(2.5).qd, 0.0)
    np.testing.assert_allclose(task.reference(5.5).q, wps[2])
    np.testing.assert_allclose(task.reference(8.5).q, wps[0])  # looped back home
    mid = task.reference(1.0)  # halfway through the first move
    np.testing.assert_allclose(mid.q, (np.asarray(wps[0]) + wps[1]) / 2)
    moving = np.asarray(wps[1]) != wps[0]
    assert np.all(np.abs(mid.qd[moving]) > 0)


def test_task_rejects_waypoints_outside_joint_limits():
    engine = SimulationEngine("table")
    bad = [HOME_QPOS, HOME_QPOS.copy()]
    bad[1][3] = 0.5  # joint4 must be negative
    with pytest.raises(ValueError, match="joint 4"):
        JointTargetTask(bad, joint_limits=engine.joint_limits)


def test_pid_integral_is_clamped():
    pid = JointPIDController(dof=2, kp=0.0, ki=10.0, velocity_feedforward=0.0, integral_limit=0.05)
    for _ in range(10_000):
        ctrl = pid.compute(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), 0.002)
    np.testing.assert_allclose(ctrl, 1.05)


# --- Closed-loop physics ----------------------------------------------------


def test_waypoints_within_limits_and_collision_free(config):
    engine = SimulationEngine(config["environment"])
    table_top = 0.40
    for name, q in config["task"]["waypoints"].items():
        q = np.asarray(q)
        lo, hi = engine.joint_limits[:, 0], engine.joint_limits[:, 1]
        assert np.all((q >= lo) & (q <= hi)), name
        engine.reset(q)
        assert engine.robot_contacts() == [], f"{name} starts in collision"
        assert engine.ee_position()[2] > table_top + 0.15, f"{name} too close to table"


def test_arm_moves_only_through_actuators(runner):
    """qpos is only written by reset(); afterwards motion comes from ctrl + mj_step."""
    engine = runner.engine
    q0 = engine.arm_qpos()
    for _ in range(int(1.5 / engine.dt)):
        runner.step()
    assert np.linalg.norm(engine.arm_qpos() - q0) > 0.1
    # Actuators are producing force, and it is within each actuator's force range.
    forces = engine.data.actuator_force[engine.arm_actuator_ids]
    assert np.any(np.abs(forces) > 1.0)
    fr = engine.model.actuator_forcerange[engine.arm_actuator_ids]
    assert np.all(forces >= fr[:, 0] - 1e-6) and np.all(forces <= fr[:, 1] + 1e-6)


def test_without_actuation_arm_falls_under_gravity(runner):
    """Sanity check that the arm is physically simulated, not kinematically posed."""
    engine = runner.engine
    engine.model.actuator_gainprm[:] = 0
    engine.model.actuator_biasprm[:] = 0
    q0 = engine.arm_qpos()
    engine.step(int(1.0 / engine.dt))
    assert np.linalg.norm(engine.arm_qpos() - q0) > 0.05


def test_closed_loop_reaches_all_waypoints(runner):
    engine, task = runner.engine, runner.task
    for k in range(1, len(task.waypoints) + 1):
        target = task.waypoints[k % len(task.waypoints)]
        # Run to the end of segment k-1's hold period.
        while engine.time < k * task.segment_time - 1e-9:
            runner.step()
            assert np.all(np.isfinite(engine.data.qpos))
            assert engine.robot_contacts() == []
        np.testing.assert_allclose(engine.arm_qpos(), target, atol=2e-3)
        assert np.max(np.abs(engine.arm_qvel())) < 0.02


def test_closed_loop_tracking_error_small(runner):
    log = runner.run_headless(runner.task.duration, log_every=5).arrays()
    error = np.abs(log["q"] - log["q_ref"]).max()
    assert error < 0.01, f"max tracking error {error:.4f} rad"


def test_integral_action_removes_gravity_sag(config):
    """The outer loop must beat the bare Menagerie servos at steady state."""

    def hold_error(ki: float, kp: float) -> float:
        cfg = {**config, "controller": {**config["controller"], "ki": ki, "kp": kp}}
        r = SimulationRunner(cfg)
        target = r.task.waypoints[1]
        while r.engine.time < r.task.segment_time - 1e-9:
            r.step()
        return float(np.abs(r.engine.arm_qpos() - target).max())

    servo_only = hold_error(ki=0.0, kp=0.0)
    closed_loop = hold_error(ki=config["controller"]["ki"], kp=config["controller"]["kp"])
    assert servo_only > 3e-3
    assert closed_loop < servo_only / 5


def test_target_marker_tracks_current_goal(runner):
    engine = runner.engine
    target_body = engine.model.body("target").id
    np.testing.assert_allclose(engine.data.xpos[target_body], runner.waypoint_ee[0])
    runner.step()
    np.testing.assert_allclose(engine.data.mocap_pos[engine.target_mocap_id], runner.waypoint_ee[1])
