"""Tests for v0.4: constraint reporting and tightening, disturbance observer, integral
action and anti-windup, model mismatch isolation, orientation metrics, sensor noise."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.computed_torque import ComputedTorqueController  # noqa: E402
from controllers.disturbance import JointDisturbanceObserver  # noqa: E402
from controllers.dynamics import ArmInverseDynamics  # noqa: E402
from robots.panda import HOME_QPOS, VELOCITY_LIMITS, Panda  # noqa: E402
from simulation.benchmark import BENCHMARK_CONFIG, longest_run, run_case, scenario_config  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.kinematics import ArmKinematics, rotation_angle  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

BENCH = yaml.safe_load(open(BENCHMARK_CONFIG, encoding="utf-8"))
HOLD = {"home": HOME_QPOS.tolist(), "home2": HOME_QPOS.tolist()}


@pytest.fixture(scope="module")
def config():
    return load_config()


def torque_cfg(config, kind, task=None, **section):
    """Config for a torque controller; section overrides go to controller.<ctc|torque_mpc>."""
    cfg = {**config, "controller": {**config["controller"], "type": kind}}
    key = "ctc" if kind == "torque_ctc" else "torque_mpc"
    cfg["controller"][key] = {**config["controller"][key], **section}
    if task is not None:
        cfg["task"] = task
    return cfg


def hold_task(seconds=1.0):
    return {"type": "joint_target", "move_time": seconds, "hold_time": 10.0, "loop": False, "waypoints": HOLD}


def push(runner, body, force, start, end):
    def before(r):
        if start <= r.engine.time < end:
            r.engine.apply_external_force(body, np.asarray(force, float))
        else:
            r.engine.clear_external_forces()
    return before


def body_jacobian_torque(engine, body, force):
    """J^T F for a world-frame force at a body's center of mass (arm joints)."""
    jacp = np.zeros((3, engine.model.nv))
    mujoco.mj_jacBodyCom(engine.model, engine.data, jacp, None, engine.model.body(body).id)
    return jacp[:, engine.arm_dof_adr].T @ force


# --- Constraint handling and reporting ----------------------------------------------------


def bench_case(controller, scenario, duration=None, **override):
    bench = {**BENCH, "controllers": {**BENCH["controllers"]}, "scenarios": {**BENCH["scenarios"]}}
    sc = {**bench["scenarios"][scenario], **override}
    if duration is not None:
        sc["duration"] = duration
    bench["scenarios"][scenario] = sc
    return run_case(scenario_config(load_config(), bench, controller, scenario), sc, 0.002)


def test_violations_reported_separately_and_removed_by_tightening():
    """On the infeasible `fast` reference the v0.3 torque MPC plans exactly at the speed
    limit and the real arm exceeds it slightly; 0.5% tightening keeps it below."""
    old, _ = bench_case("torque_mpc_v03", "fast", duration=1.2)
    new, _ = bench_case("torque_mpc", "fast", duration=1.2)
    assert old["reference_feasible"] is False and old["reference_velocity_ratio"] > 1.4
    assert old["mpc_max_predicted_velocity_ratio"] <= 1.0 + 1e-4       # the plan believes it is feasible
    assert old["velocity_limit_violation_steps"] > 0 and 0 < old["velocity_violation_max_excess"] < 0.01
    assert old["velocity_violation_longest_s"] > 0 and old["velocity_violation_integral_rad"] > 0
    assert old["mpc_max_slack"] < 1e-3
    assert new["velocity_limit_violation_steps"] == 0 and new["max_velocity_ratio"] < 1.0
    assert new["mpc_max_predicted_velocity_ratio"] <= 0.995 + 1e-4
    assert new["mpc_fallbacks"] == 0 and new["finite_state"]


def test_feasible_scenario_reports_feasible_reference_and_no_violations():
    m, _ = bench_case("torque_ctc", "fast_feasible", duration=3.0)
    assert m["reference_feasible"] is True and m["reference_velocity_ratio"] <= 0.9 + 1e-9
    assert m["velocity_limit_violation_steps"] == 0 and m["max_error_rad"] < 1e-3


def test_longest_run():
    assert longest_run(np.array([], dtype=bool)) == 0
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == 3
    assert longest_run(np.ones(5, dtype=bool)) == 5


def test_tightening_validated():
    cfg = torque_cfg(load_config(), "torque_mpc", velocity_margin=1.0)
    with pytest.raises(ValueError, match="velocity_margin"):
        SimulationRunner(cfg)


# --- Disturbance observer ---------------------------------------------------------------


def test_observer_is_zero_with_exact_model_and_matches_external_torque(config):
    cfg = torque_cfg(config, "torque_ctc", hold_task(), disturbance_observer={"enabled": True, "time_constant": 0.01})
    r = SimulationRunner(cfg)
    r.run_headless(0.5, log_every=50)
    assert np.abs(r.controller.disturbance_estimate).max() < 1e-6  # nothing unknown: no estimate
    force = np.array([0.0, 30.0, 0.0])
    r.run_headless(0.5, log_every=50, before_step=push(r, "hand", force, 0.0, 10.0))
    expected = body_jacobian_torque(r.engine, "hand", force)
    np.testing.assert_allclose(r.controller.disturbance_estimate, expected, atol=0.02 * np.abs(expected).max())


def test_observer_rejects_push_where_plain_ctc_deflects(config):
    errs = {}
    for name, obs in (("ctc", {"enabled": False}), ("dob", {"enabled": True, "time_constant": 0.01})):
        r = SimulationRunner(torque_cfg(config, "torque_ctc", hold_task(), disturbance_observer=obs))
        log = r.run_headless(2.0, log_every=1, before_step=push(r, "hand", [0, 30, 0], 1.0, 2.0)).arrays()
        late = (log["time"] > 1.8) & (log["time"] < 2.0)
        errs[name] = np.abs(log["q"] - log["q_ref"])[late].max()
    assert errs["ctc"] > 0.02 and errs["dob"] < 1e-4  # steady deflection removed


def test_offset_free_torque_mpc_rejects_push_and_stays_stable(config):
    errs = {}
    for name, obs in (("v03", {"enabled": False}), ("v04", None)):
        section = {} if obs is None else {"disturbance_observer": obs}
        r = SimulationRunner(torque_cfg(config, "torque_mpc", hold_task(), **section))
        log = r.run_headless(3.0, log_every=1, before_step=push(r, "hand", [0, 30, 0], 1.0, 2.0)).arrays()
        assert np.all(np.isfinite(log["q"])) and r.controller.stats.summary(0.02)["fallbacks"] == 0
        late = (log["time"] > 1.8) & (log["time"] < 2.0)
        errs[name] = (np.abs(log["q"] - log["q_ref"])[late].max(), np.abs(log["q"] - log["q_ref"])[log["time"] > 2.8].max())
    assert errs["v03"][0] > 5e-3 and errs["v04"][0] < 1e-4
    assert errs["v04"][1] < 1e-4  # back on the reference after release, no lingering estimate


def test_observer_saturation_does_not_wind_up():
    """A push the actuators cannot resist: the estimate stays bounded at its clip and
    recovers within a few time constants after release (it is driven by applied torque)."""
    engine = SimulationEngine("table", Panda(actuation="torque"))
    obs = JointDisturbanceObserver(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr,
                                   time_constant=0.01, limit=np.array([43.5] * 4 + [6.0] * 3))
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    ctrl = ComputedTorqueController(dyn, engine.ctrl_limits, observer=obs)
    engine.reset(HOME_QPOS)
    peak, saturated = np.zeros(7), 0
    for i in range(1500):  # 3 s; 0.5-0.7 s a 200 N push on the hand (needs > 87 Nm on joint 2)
        if 0.5 <= engine.time < 0.7:
            engine.apply_external_force("hand", np.array([200.0, 0.0, 0.0]))
        else:
            engine.clear_external_forces()
        tau = ctrl.compute(engine.arm_qpos(), engine.arm_qvel(), HOME_QPOS, np.zeros(7), engine.dt)
        engine.set_arm_command(tau)
        engine.step()
        saturated += bool(np.any(np.abs(tau) >= engine.ctrl_limits[:, 1] - 1e-9))
        peak = np.maximum(peak, np.abs(obs.estimate))
        if i == 349:  # last step of the push
            at_release = obs.estimate.copy()
    assert saturated > 50  # the actuators really were saturated
    assert np.all(peak <= obs.limit + 1e-9) and np.any(peak >= obs.limit - 1e-6)  # clipped, not unbounded
    assert np.abs(at_release).max() > 5.0
    assert np.all(np.isfinite(engine.data.qpos))
    assert np.abs(obs.estimate).max() < 0.05  # 1.5 s after release: no windup memory
    np.testing.assert_allclose(engine.arm_qpos(), HOME_QPOS, atol=1e-3)


def test_ctc_integral_action_removes_offset_and_anti_windup_freezes_when_clipped(config):
    r = SimulationRunner(torque_cfg(config, "torque_ctc", hold_task(), integral_gain=3000.0,
                                    disturbance_observer={"enabled": False}))
    log = r.run_headless(2.0, log_every=1, before_step=push(r, "hand", [0, 30, 0], 0.5, 10.0)).arrays()
    assert np.abs(log["q"][-1] - log["q_ref"][-1]).max() < 1e-3  # plain CTC deflects ~35 mrad
    # Anti-windup: a constant error the (weak) actuators cannot correct. Every joint saturates
    # on the first call, so the integral freezes after one step instead of growing.
    engine = SimulationEngine("table", Panda(actuation="torque"))
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    weak = np.column_stack([-np.full(7, 0.5), np.full(7, 0.5)])
    c = ComputedTorqueController(dyn, weak, integral_gain=3000.0, integral_limit=0.1)
    for _ in range(200):
        c.compute(HOME_QPOS, np.zeros(7), HOME_QPOS + 0.3, np.zeros(7), engine.dt)
    assert c._saturated.all()
    np.testing.assert_allclose(c.integral, 0.3 * engine.dt)  # frozen after the first step
    c.reset()
    c.torque_limits = np.column_stack([-np.full(7, 1e6), np.full(7, 1e6)])  # never saturated: integrates up to the clamp
    for _ in range(200):
        c.compute(HOME_QPOS, np.zeros(7), HOME_QPOS + 0.3, np.zeros(7), engine.dt)
    np.testing.assert_allclose(c.integral, 0.1)


def test_integral_gain_must_satisfy_routh_bound():
    engine = SimulationEngine("table", Panda(actuation="torque"))
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    with pytest.raises(ValueError, match="Kd \\* Kp"):
        ComputedTorqueController(dyn, engine.ctrl_limits, integral_gain=40000.0)  # Kd Kp = 31250


# --- Model mismatch ---------------------------------------------------------------------


def test_observer_uses_controller_model_never_the_plant(config):
    cfg = torque_cfg(config, "torque_mpc")
    cfg["controller_model"] = {"mass_scale": 0.75}
    r = SimulationRunner(cfg)
    obs_model = r.controller.observer.dynamics.model
    hand = r.engine.model.body("hand").id
    assert obs_model is not r.engine.model and obs_model is not r.engine.controller_model  # private copy
    assert np.isclose(obs_model.body_mass[hand], 0.75 * r.engine.model.body_mass[hand])
    assert r.engine.controller_model.opt.enableflags == r.engine.model.opt.enableflags  # flag stays private


@pytest.mark.parametrize("scale", [0.75, 1.25])
def test_offset_free_mpc_improves_tracking_under_mass_error(config, scale):
    task = {**config["task"], "loop": False}
    errs = {}
    for name, obs in (("v03", {"enabled": False}), ("v04", None)):
        section = {} if obs is None else {"disturbance_observer": obs}
        cfg = torque_cfg(config, "torque_mpc", task, **section)
        cfg["controller_model"] = {"mass_scale": scale}
        r = SimulationRunner(cfg)
        log = r.run_headless(4.0, log_every=1).arrays()  # first move + hold
        assert np.all(np.isfinite(log["q"])) and r.controller.stats.summary(0.02)["fallbacks"] == 0
        errs[name] = np.abs(log["q"] - log["q_ref"]).max()
    assert errs["v04"] < 0.5 * errs["v03"]


# --- Fallback with the v0.4 controller ----------------------------------------------------


def test_v04_torque_mpc_falls_back_on_bad_measurement_and_recovers(config):
    r = SimulationRunner(torque_cfg(config, "torque_mpc", hold_task()))
    c, e = r.controller, r.engine
    q, qd = e.arm_qpos(), e.arm_qvel()
    bad = q.copy()
    bad[2] = np.nan
    c.compute(bad, qd, HOME_QPOS, np.zeros(7), e.dt, t=0.0)
    assert c.using_fallback and c.stats.updates[-1].fallback
    log = r.run_headless(1.0, log_every=1).arrays()
    assert not c.using_fallback and np.all(np.isfinite(log["q"]))
    assert np.abs(log["q"][-1] - HOME_QPOS).max() < 1e-3


# --- Orientation ------------------------------------------------------------------------


def axis_angle(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    quat = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


@pytest.mark.parametrize("angle", [0.0, 1e-7, 0.3, 1.0, np.pi / 2, np.pi - 1e-6, np.pi])
def test_rotation_angle_matches_axis_angle(angle):
    base = axis_angle([0.3, -1.0, 0.2], 0.7)
    rotated = base @ axis_angle([1.0, 2.0, -0.5], angle)
    assert np.isclose(rotation_angle(base, rotated), angle, atol=1e-9)
    assert np.isclose(rotation_angle(rotated, base), angle, atol=1e-9)  # symmetric


def test_fk_pose_and_benchmark_orientation_metrics():
    engine = SimulationEngine("table")
    kin = ArmKinematics(engine)
    pos, R = kin.fk_pose(HOME_QPOS)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pos, engine.forward_kinematics(HOME_QPOS), atol=1e-12)
    # Rotating only joint 7 (the flange axis) changes orientation by exactly that angle.
    q = HOME_QPOS.copy()
    q[6] += 0.25
    assert np.isclose(rotation_angle(R, kin.fk_pose(q)[1]), 0.25, atol=1e-9)
    # Joint tasks report tracking error; reach tasks have no orientation reference.
    joint, _ = bench_case("torque_ctc", "demo", duration=2.0)
    assert 0 < joint["ee_orientation_max_error_deg"] < 0.1 and joint["ee_orientation_change_max_deg"] > 1.0
    reach, _ = bench_case("torque_ctc", "reach", duration=5.0)
    assert np.isnan(reach["ee_orientation_max_error_deg"]) and reach["ee_orientation_change_max_deg"] > 1.0
    assert reach["ee_orientation_hold_drift_max_deg"] < 0.01


# --- Sensor noise -----------------------------------------------------------------------


def test_sensor_noise_is_seeded_and_only_affects_measurements(config):
    cfg = torque_cfg(config, "torque_ctc", hold_task())
    cfg["sensor_noise"] = {"position_std": 1e-4, "velocity_std": 1e-3, "seed": 3}
    a = SimulationRunner(cfg).run_headless(0.3, log_every=1).arrays()
    b = SimulationRunner(cfg).run_headless(0.3, log_every=1).arrays()
    clean = SimulationRunner({**cfg, "sensor_noise": None}).run_headless(0.3, log_every=1).arrays()
    np.testing.assert_array_equal(a["q"], b["q"])  # reproducible
    assert not np.array_equal(a["force"], clean["force"])  # the controller saw noise
    assert np.abs(a["q"] - clean["q"]).max() < 1e-3  # logged q is the true state, still on target
