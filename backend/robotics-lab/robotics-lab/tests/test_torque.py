"""Tests for v0.3 torque actuation: actuator variant, inverse dynamics, computed-torque control, torque MPC."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.computed_torque import ComputedTorqueController  # noqa: E402
from controllers.dynamics import ArmInverseDynamics  # noqa: E402
from controllers.prediction import MuJoCoPredictionModel  # noqa: E402
from controllers.torque_mpc import TorqueMPCController  # noqa: E402
from robots.panda import HOME_QPOS, VELOCITY_LIMITS, Panda  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.runner import SimulationRunner, build_controller, load_config  # noqa: E402
from tasks.joint_target import JointTargetTask  # noqa: E402

LEFT = np.array([0.7, 0.2, 0.0, -2.0, 0.0, 2.2, 1.2])
QD = np.array([0.3, -0.2, 0.2, 0.3, -0.2, 0.2, 0.25])
TORQUE_LIMITS = np.array([87.0] * 4 + [12.0] * 3)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def engine():
    return SimulationEngine("table", Panda(actuation="torque"))


def torque_config(config, kind="torque_mpc", **section):
    cfg = {**config, "controller": {**config["controller"], "type": kind}}
    if section:
        cfg["controller"]["torque_mpc"] = {**config["controller"]["torque_mpc"], **section}
    return cfg


def make_mpc(engine, preview=None, **kw):
    model = MuJoCoPredictionModel(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr,
                                  engine.arm_actuator_ids, substeps=10)
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    kw.setdefault("velocity_limits", VELOCITY_LIMITS)
    kw.setdefault("joint_limits", engine.joint_limits)
    return TorqueMPCController(model, ComputedTorqueController(dyn, engine.ctrl_limits), reference_preview=preview, **kw)


# --- Actuator variant ---------------------------------------------------------


def test_torque_variant_keeps_physical_properties(engine):
    original = SimulationEngine("table")
    m, o = engine.model, original.model
    for field in ("body_mass", "body_inertia", "dof_armature", "dof_damping", "jnt_range", "geom_size"):
        np.testing.assert_array_equal(getattr(m, field), getattr(o, field), err_msg=field)
    ids = engine.arm_actuator_ids
    np.testing.assert_array_equal(m.actuator_gainprm[ids, 0], 1.0)
    assert np.all(m.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_NONE)
    np.testing.assert_array_equal(engine.ctrl_limits[:, 1], TORQUE_LIMITS)
    np.testing.assert_array_equal(m.actuator_forcerange[ids, 1], TORQUE_LIMITS)
    # Gripper actuator untouched; arm torques start at zero after reset.
    np.testing.assert_array_equal(m.actuator_gainprm[7], o.actuator_gainprm[7])
    np.testing.assert_array_equal(engine.data.ctrl[ids], 0.0)


def test_unknown_actuation_rejected():
    with pytest.raises(ValueError, match="actuation"):
        Panda(actuation="velocity")


def test_torque_command_is_applied_force(engine):
    """The commanded torque is exactly the force MuJoCo applies to each joint."""
    tau = np.array([5.0, -10.0, 3.0, 8.0, -2.0, 4.0, -1.5])
    engine.set_arm_command(tau)
    engine.step()
    np.testing.assert_allclose(engine.arm_actuator_force(), tau)
    np.testing.assert_allclose(engine.data.qfrc_actuator[engine.arm_dof_adr], tau)
    engine.set_arm_command(np.array([200.0, -200.0, 0, 0, 50.0, -50.0, 0]))
    engine.step()
    np.testing.assert_allclose(engine.arm_actuator_force()[[0, 1, 4, 5]], [87, -87, 12, -12])


# --- Inverse dynamics and reference acceleration ---------------------------------


def test_inverse_dynamics_reproduces_acceleration(engine):
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    qdd = np.array([1.0, -1.0, 0.5, 0.8, -2.0, 1.0, 3.0])
    tau = dyn.torque(LEFT, QD, qdd)
    before = engine.data.qpos.copy()
    engine.reset(LEFT)
    engine.data.qvel[engine.arm_dof_adr] = QD
    engine.set_arm_command(tau)
    mujoco.mj_forward(engine.model, engine.data)
    np.testing.assert_allclose(engine.data.qacc[engine.arm_dof_adr], qdd, atol=5e-4)
    assert not np.array_equal(before, engine.data.qpos)  # sanity: reset changed it, dyn did not


def test_inverse_dynamics_does_not_touch_live_state(engine):
    engine.reset(LEFT)
    snapshot = {k: getattr(engine.data, k).copy() for k in ("qpos", "qvel", "qacc", "ctrl", "qfrc_inverse")}
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    dyn.torque(HOME_QPOS, QD, np.ones(7))
    for k, v in snapshot.items():
        np.testing.assert_array_equal(getattr(engine.data, k), v, err_msg=k)


def test_reference_acceleration_matches_finite_difference():
    task = JointTargetTask([np.zeros(7), np.ones(7)], move_time=2.0, hold_time=1.0)
    for t in (0.2, 1.0, 1.9, 2.5):
        fd = (task.reference(t + 1e-6).qd - task.reference(t - 1e-6).qd) / 2e-6
        np.testing.assert_allclose(task.reference(t).qdd, fd, atol=1e-4)


# --- Prediction model with torque inputs -------------------------------------------


def test_prediction_model_detects_torque_actuators(engine):
    model = MuJoCoPredictionModel(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr,
                                  engine.arm_actuator_ids, substeps=5)
    assert model.actuation == "torque" and model.servo is None
    np.testing.assert_array_equal(model.force_limits[:, 1], TORQUE_LIMITS)
    # Short-horizon prediction with torque inputs (gravity hold + moderate perturbation).
    g = model.gravity_torque(LEFT)
    rng = np.random.default_rng(0)
    commands = np.array([g + rng.uniform(-1, 1, 7) * 0.1 * TORQUE_LIMITS for _ in range(5)])
    lin = model.linearize(LEFT, QD, commands[0])
    err = np.abs(lin.predict(lin.x0, commands) - model.rollout(LEFT, QD, commands))
    assert err[:, :7].max() < 5e-4  # 5 x 10 ms; LTI error grows with horizon (see report)


# --- Computed-torque baseline -----------------------------------------------------


def test_computed_torque_tracks_demo_trajectory(config):
    r = SimulationRunner(torque_config(config, "torque_ctc"))
    log = r.run_headless(r.task.duration, log_every=1).arrays()
    assert np.abs(log["q"] - log["q_ref"]).max() < 1e-3
    assert np.all(np.abs(log["force"]) <= TORQUE_LIMITS + 1e-9)


# --- Torque MPC ------------------------------------------------------------------


def test_torque_mpc_rejects_servo_actuators():
    servo = SimulationEngine("table")
    model = MuJoCoPredictionModel(servo.model, servo.data, servo.arm_qpos_adr, servo.arm_dof_adr,
                                  servo.arm_actuator_ids, substeps=10)
    with pytest.raises(ValueError, match="torque"):
        TorqueMPCController(model, fallback=None)


def test_torque_mpc_holds_with_gravity_torque_at_rest(engine):
    mpc = make_mpc(engine)
    q = engine.arm_qpos()
    tau = mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt, t=0.0)
    assert mpc.stats.updates[-1].status == "solved"
    # Exact hold torque is gravity (verified: zero drift). The QP's tolerance (1e-5) allows ~0.2 Nm
    # on an 87 Nm joint because that changes the cost by ~5e-6; the plan must still keep the arm still.
    np.testing.assert_array_less(np.abs(tau - mpc.model.gravity_torque(q)), 0.01 * TORQUE_LIMITS)
    p = mpc.last_prediction
    assert np.abs(p["states"][:, :7] - q).max() < 1e-4
    np.testing.assert_allclose(p["model"].predict(p["x0"], p["commands"]), p["states"], atol=1e-8)


def test_torque_mpc_plan_respects_hard_torque_and_soft_state_limits(engine):
    far = HOME_QPOS.copy()
    far[0] = 2.5
    far[6] = 2.5
    mpc = make_mpc(engine, preview=lambda t: (far, np.zeros(7)), solver_settings={"max_iter": 4000})
    mpc.compute(HOME_QPOS, np.zeros(7), far, np.zeros(7), engine.dt, t=0.0)
    p = mpc.last_prediction
    assert mpc.stats.updates[-1].status in ("solved", "solved inaccurate")  # both accepted by the controller
    assert np.all(np.abs(p["commands"]) <= TORQUE_LIMITS + 1e-4)          # hard (to OSQP feasibility tolerance)
    assert np.abs(p["commands"]).max(axis=0)[6] > 0.99 * 12.0               # wrist torque limit is active
    assert np.all(np.abs(p["states"][:, 7:]) <= VELOCITY_LIMITS + 1e-3)    # soft, slack ~0
    assert mpc.stats.updates[-1].max_slack < 1e-3


def test_torque_mpc_state_feedback(engine):
    q_ref = HOME_QPOS
    nominal = make_mpc(engine).compute(q_ref, np.zeros(7), q_ref, np.zeros(7), engine.dt, t=0.0)
    dq = np.zeros(7)
    dq[1] = 0.05
    pushed = make_mpc(engine).compute(q_ref + dq, np.zeros(7), q_ref, np.zeros(7), engine.dt, t=0.0)
    gravity_shift = make_mpc(engine).model.gravity_torque(q_ref + dq)[1] - make_mpc(engine).model.gravity_torque(q_ref)[1]
    # Restoring torque on joint 2 beyond the gravity change at the displaced pose.
    assert pushed[1] - nominal[1] - gravity_shift < -1.0


def test_torque_mpc_tracks_and_respects_speed_limits(config):
    cfg = torque_config(config)
    cfg["task"] = {**config["task"], "move_time": 1.0, "hold_time": 1.0,
                   "trajectory": None}  # unchecked v0.3 timing: the reference exceeds the speed limits
    r = SimulationRunner(cfg)
    log = r.run_headless(r.task.duration, log_every=1).arrays()
    assert np.all(np.isfinite(log["q"]))
    assert np.max(np.abs(log["qd"]) / VELOCITY_LIMITS) < 1.02
    assert np.all(np.abs(log["force"]) <= TORQUE_LIMITS + 1e-9)
    s = r.controller.stats.summary(r.controller.control_period)
    assert s["fallbacks"] <= 2
    # Settles on the final waypoint once the reference holds still.
    np.testing.assert_allclose(r.engine.arm_qpos(), r.task.reference(r.engine.time).q, atol=2e-3)


def test_torque_mpc_demo_tracking(config):
    r = SimulationRunner(torque_config(config))
    log = r.run_headless(r.task.duration, log_every=1).arrays()
    assert np.abs(log["q"] - log["q_ref"]).max() < 2e-3
    assert r.controller.stats.summary(0.02)["fallbacks"] == 0


def test_torque_mpc_solver_failure_uses_computed_torque_fallback(engine, monkeypatch):
    mpc = make_mpc(engine)
    q = HOME_QPOS
    mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt, t=0.0)

    def broken(*a, **k):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(mpc._solver, "solve", broken)
    for _ in range(9):
        mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
    tau = mpc.compute(q + 0.01, np.zeros(7), q, np.zeros(7), engine.dt)
    assert mpc.using_fallback
    np.testing.assert_allclose(tau, mpc.fallback.compute(q + 0.01, np.zeros(7), q, np.zeros(7), engine.dt))


# --- Wiring ----------------------------------------------------------------------


@pytest.mark.parametrize("kind, cls, actuation", [
    ("torque_ctc", ComputedTorqueController, "torque"),
    ("ctc", ComputedTorqueController, "torque"),
    ("torque_mpc", TorqueMPCController, "torque"),
])
def test_torque_controllers_build_torque_engine(config, kind, cls, actuation):
    r = SimulationRunner(torque_config(config, kind))
    assert isinstance(r.controller, cls) and r.engine.robot.actuation == actuation


def test_actuator_mismatch_rejected(config, engine):
    with pytest.raises(ValueError, match="position actuators"):
        build_controller({**config["controller"], "type": "joint_pid"}, engine)


@pytest.mark.parametrize("rate, substeps", [(50, 10), (100, 5), (250, 2)])
def test_control_rate_independent_of_physics_timestep(config, rate, substeps):
    r = SimulationRunner(torque_config(config, control_rate_hz=rate, horizon=10))
    assert r.engine.dt == 0.002  # physics timestep fixed by the environment
    assert r.controller.substeps == substeps
    assert np.isclose(r.controller.control_period, substeps * r.engine.dt)
    r.run_headless(0.2)
    assert len(r.controller.stats.updates) == int(round(0.2 * rate))
