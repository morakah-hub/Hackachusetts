"""Tests for v0.2: MuJoCo prediction model, joint-space MPC, controller switching, benchmark."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.mpc import JointMPCController  # noqa: E402
from controllers.pid import JointPIDController  # noqa: E402
from controllers.prediction import MuJoCoPredictionModel  # noqa: E402
from robots.panda import HOME_QPOS, VELOCITY_LIMITS  # noqa: E402
from simulation.benchmark import BENCHMARK_CONFIG, run_benchmark, settling_times  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.runner import SimulationRunner, build_controller, load_config  # noqa: E402

LEFT = np.array([0.7, 0.2, 0.0, -2.0, 0.0, 2.2, 1.2])
# Step references and far-off targets need ~700-1800 OSQP iterations, more than
# the real-time default (500). Tests about plan validity use this budget;
# test_step_reference_exceeds_realtime_budget_and_falls_back covers the default.
OFFLINE_SOLVER = {"max_iter": 4000}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def engine():
    return SimulationEngine("table")


def make_model(engine, substeps=10, method="power"):
    return MuJoCoPredictionModel(
        engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr, engine.arm_actuator_ids, substeps, method
    )


def make_mpc(engine, preview=None, **kw):
    kw.setdefault("velocity_limits", VELOCITY_LIMITS)
    kw.setdefault("joint_limits", engine.joint_limits)
    return JointMPCController(
        make_model(engine), JointPIDController(7, ctrl_limits=engine.ctrl_limits), reference_preview=preview, **kw
    )


def mpc_config(config, **mpc):
    cfg = {**config, "controller": {**config["controller"], "type": "joint_mpc"}}
    cfg["controller"]["mpc"] = {**config["controller"].get("mpc", {}), **mpc}
    return cfg


def hold_command(model, q):
    """Setpoint that holds q still: the servo offset that balances gravity."""
    return q + model.gravity_torque(q) / model.servo.kp


# --- Prediction model -------------------------------------------------------


def test_actuators_modeled_as_position_servos(engine):
    servo = make_model(engine).servo
    np.testing.assert_allclose(servo.kd / servo.kp, 0.1)
    np.testing.assert_allclose(servo.force_limits[:, 1], [87] * 4 + [12] * 3)
    # Command limits are joint angles (rad), distinct from the torque limits.
    np.testing.assert_allclose(servo.ctrl_limits, engine.joint_limits)


@pytest.mark.parametrize("method", ["power", "rollout"])
@pytest.mark.parametrize("start", ["home_rest", "left_moving"])
def test_prediction_matches_mujoco_rollout(engine, method, start):
    """With torques inside the force range, the linear model reproduces MuJoCo
    over a 20-interval (0.4 s) horizon."""
    model = make_model(engine, method=method)
    if start == "home_rest":
        q, qd = HOME_QPOS.copy(), np.zeros(7)
    else:
        q, qd = LEFT.copy(), np.array([0.3, -0.2, 0.2, 0.3, -0.2, 0.2, 0.25])
    u0 = hold_command(model, q) + 0.1 * qd  # gravity hold + servo damping lag
    lin = model.linearize(q, qd, u0)
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-1, 1, (20, 7)) * 0.3 * model.servo.force_limits[:, 1] / model.servo.kp
    commands = np.array([u0 + qd * k * model.dt + offsets[k] for k in range(20)])
    true = model.rollout(q, qd, commands)
    predicted = lin.predict(lin.x0, commands)
    tol = 1e-5 if start == "home_rest" else 5e-3  # moving case includes force-clip spikes
    assert np.abs(predicted[:, :7] - true[:, :7]).max() < tol
    # Exact at the linearization point (c is computed from a true rollout).
    np.testing.assert_allclose(lin.predict(lin.x0, u0[None])[0], model.rollout(q, qd, u0[None])[0], atol=1e-10)


def test_prediction_never_modifies_live_state(engine):
    engine.reset(LEFT)
    engine.data.qvel[:7] = 0.2
    before = {k: getattr(engine.data, k).copy() for k in ("qpos", "qvel", "ctrl", "qacc_warmstart", "xfrc_applied")}
    t0 = engine.data.time
    model = make_model(engine)
    model.linearize(LEFT, np.full(7, 0.2), LEFT)
    model.rollout(LEFT, np.zeros(7), np.tile(LEFT, (5, 1)))
    model.gravity_torque(HOME_QPOS)
    for k, v in before.items():
        assert np.array_equal(getattr(engine.data, k), v), k
    assert engine.data.time == t0


def test_external_forces_are_not_visible_to_the_model(engine):
    """A controller cannot measure a push; the model must not see xfrc_applied."""
    model = make_model(engine)
    u = hold_command(model, HOME_QPOS)
    free = model.rollout(HOME_QPOS, np.zeros(7), u[None])
    engine.apply_external_force("hand", [0, 50, 0])
    pushed = model.rollout(HOME_QPOS, np.zeros(7), u[None])
    np.testing.assert_array_equal(free, pushed)


# --- MPC optimization -------------------------------------------------------


def test_mpc_solution_valid_at_rest(engine):
    """Holding still: the optimum is the gravity-compensating setpoint, and the
    QP's predicted states are the model's response to its own commands."""
    mpc = make_mpc(engine)
    q = engine.arm_qpos()
    u = mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt, t=0.0)
    upd = mpc.stats.updates[-1]
    assert upd.status == "solved" and not upd.fallback
    np.testing.assert_allclose(u, hold_command(mpc.model, q), atol=1e-5)
    p = mpc.last_prediction
    np.testing.assert_allclose(p["model"].predict(p["x0"], p["commands"]), p["states"], atol=1e-8)
    lo, hi = engine.ctrl_limits[:, 0], engine.ctrl_limits[:, 1]
    assert np.all((p["commands"] >= lo - 1e-9) & (p["commands"] <= hi + 1e-9))


def test_mpc_holds_command_between_updates(engine):
    mpc = make_mpc(engine, solver_settings=OFFLINE_SOLVER)
    engine.reset(HOME_QPOS)
    target = LEFT
    commands = []
    for i in range(25):
        commands.append(mpc.compute(engine.arm_qpos(), engine.arm_qvel(), target, np.zeros(7), engine.dt, t=engine.time))
        engine.set_arm_command(commands[-1])
        engine.step()
    assert len(mpc.stats.updates) == 3  # steps 0, 10, 20 at 50 Hz / 500 Hz
    assert not any(u.fallback for u in mpc.stats.updates)
    for block in (commands[0:10], commands[10:20]):
        assert all(np.array_equal(c, block[0]) for c in block)
    assert not np.array_equal(commands[0], commands[10])
    # The step reference drives the setpoint at its slew limit (v_max * Ts per update).
    step = np.abs(commands[10] - commands[0])
    assert np.all(step <= mpc.slew + 1e-6) and np.isclose(step.max() / mpc.slew[np.argmax(step)], 1.0, atol=1e-3)


def test_state_feedback_changes_action_when_disturbed(engine):
    """Same reference, different measured state -> different command that pushes back."""
    q_ref = HOME_QPOS
    nominal = make_mpc(engine).compute(q_ref, np.zeros(7), q_ref, np.zeros(7), engine.dt, t=0.0)
    for j in (0, 3, 6):
        dq = np.zeros(7)
        dq[j] = 0.05
        u = make_mpc(engine).compute(q_ref + dq, np.zeros(7), q_ref, np.zeros(7), engine.dt, t=0.0)
        assert np.abs(u - nominal).max() > 1e-3
        # Displaced by +0.05 rad: the new setpoint must be below the displaced
        # position (servo pulls back towards the reference).
        assert u[j] < q_ref[j] + dq[j]
        # Velocity disturbance also changes the action.
    qd = np.zeros(7)
    qd[1] = 0.5
    u = make_mpc(engine).compute(q_ref, qd, q_ref, np.zeros(7), engine.dt, t=0.0)
    assert np.abs(u - nominal).max() > 1e-3


def test_predicted_trajectory_respects_constraints(engine):
    """Reference far beyond the speed limits and past a joint limit, arm already
    moving fast: the plan stays inside every modeled constraint."""
    jl = engine.joint_limits
    far = HOME_QPOS.copy()
    far[0] = 2.5          # 2.5 rad ahead: far above what 2.175 rad/s allows in 0.3 s
    far[3] = jl[3, 0]     # joint 4 exactly at its limit, inside the 0.02 margin
    qd0 = np.zeros(7)
    qd0[0] = 2.0          # already moving at 92% of joint 1's speed limit
    # Slew scale 2 lets the setpoint outrun v_max, so the velocity limit itself must hold the plan back.
    mpc = make_mpc(engine, preview=lambda t: (far, np.zeros(7)), command_slew_scale=2.0, solver_settings=OFFLINE_SOLVER)
    mpc.compute(HOME_QPOS, qd0, far, np.zeros(7), engine.dt, t=0.0)
    p = mpc.last_prediction
    X, U = p["states"], p["commands"]
    q, qd = X[:, :7], X[:, 7:]
    tol = 1e-4
    assert np.all(q >= mpc.q_bounds[:, 0] - tol) and np.all(q <= mpc.q_bounds[:, 1] + tol)
    assert np.all(np.abs(qd) <= VELOCITY_LIMITS + tol)
    assert np.all((U >= engine.ctrl_limits[:, 0] - 1e-9) & (U <= engine.ctrl_limits[:, 1] + 1e-9))
    tau = mpc.kp * (U - q) - mpc.kd * qd  # sustained torque, interval k at state k+1
    assert np.all(np.abs(tau) <= mpc.torque_limit_fraction * mpc.force_limits[:, 1] + 1e-2)
    assert mpc.stats.updates[-1].max_slack < 1e-4
    # Setpoint slew limit, including the step from the command before the plan.
    u_prev = HOME_QPOS + mpc.model.gravity_torque(HOME_QPOS) / mpc.kp + 0.1 * qd0
    steps = np.abs(np.diff(np.vstack([u_prev, U]), axis=0))
    assert np.all(steps <= mpc.slew + 1e-6)
    # The velocity limit is active, so the test exercises it (the plan is not trivially slow).
    assert np.abs(qd[:, 0]).max() > 0.99 * VELOCITY_LIMITS[0]


# --- Closed loop ------------------------------------------------------------


def test_mpc_tracks_reachable_target(config):
    cfg = mpc_config(config)
    cfg["task"] = {**config["task"], "move_time": 2.0, "hold_time": 1.0, "loop": False,
                   "waypoints": {"home": HOME_QPOS.tolist(), "left": LEFT.tolist()}}
    r = SimulationRunner(cfg)
    log = r.run_headless(3.0, log_every=1).arrays()
    assert np.abs(log["q"] - log["q_ref"]).max() < 0.02
    np.testing.assert_allclose(r.engine.arm_qpos(), LEFT, atol=2e-3)
    assert np.max(np.abs(r.engine.arm_qvel())) < 0.02
    s = r.controller.stats.summary(r.controller.control_period)
    assert s["fallbacks"] == 0 and s["updates"] == 150


def test_joint_limits_respected_in_simulation(config):
    """Reference ends exactly on joint 1's upper limit (collision-free there);
    the MPC keeps the real joint at its configured margin instead."""
    target = HOME_QPOS.copy()
    target[0] = 2.8973
    cfg = mpc_config(config)
    cfg["task"] = {**config["task"], "move_time": 2.5, "hold_time": 1.0, "loop": False,
                   "waypoints": {"home": HOME_QPOS.tolist(), "limit": target.tolist()}}
    r = SimulationRunner(cfg)
    log = r.run_headless(3.5, log_every=1).arrays()
    hi = r.engine.joint_limits[0, 1]
    margin = hi - r.controller.q_bounds[0, 1]
    assert log["q"][:, 0].max() < hi - margin + 2e-3
    assert abs(log["q"][-1, 0] - (hi - margin)) < 5e-3  # pressed against the margin, not short of it
    assert r.controller.stats.summary(0.02)["fallbacks"] == 0
    # PID, by contrast, follows the reference onto the limit itself.
    pid = SimulationRunner({**cfg, "controller": {**cfg["controller"], "type": "joint_pid"}})
    pid_log = pid.run_headless(3.5, log_every=1).arrays()
    assert pid_log["q"][:, 0].max() > hi - margin + 0.01


def test_no_nan_or_instability_under_aggressive_reference(config):
    """1 s moves ask for more than the joint speed limits; MPC must stay stable."""
    cfg = mpc_config(config)
    cfg["task"] = {**config["task"], "move_time": 1.0, "hold_time": 1.0,
                   "trajectory": None}  # v0.3 unchecked timing: the point is the infeasible reference
    r = SimulationRunner(cfg)
    log = r.run_headless(4.0, log_every=1).arrays()
    assert np.all(np.isfinite(log["q"])) and np.all(np.isfinite(log["qd"]))
    assert np.abs(log["qd"]).max() < 1.3 * VELOCITY_LIMITS.max()
    assert r.controller.stats.summary(0.02)["fallbacks"] == 0
    # Back at home after one pass through the waypoints (t = 4 s is mid-pass, so
    # check the final hold of a full pass instead).
    r.run_headless(2.0)
    np.testing.assert_allclose(r.engine.arm_qpos(), r.task.reference(r.engine.time).q, atol=5e-3)


# --- Fallback ---------------------------------------------------------------


def test_solver_exception_activates_pid_fallback(engine, monkeypatch):
    mpc = make_mpc(engine)
    q = engine.arm_qpos()
    mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt, t=0.0)  # builds the solver
    pid = JointPIDController(7, ctrl_limits=engine.ctrl_limits)
    pid.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)          # mirror the shadow PID

    def broken(*a, **k):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(mpc._solver, "solve", broken)
    for _ in range(9):
        mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
        pid.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
    u = mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
    expected = pid.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
    assert mpc.using_fallback and mpc.stats.updates[-1].fallback
    np.testing.assert_allclose(u, expected)
    # Recovers at the next update once the solver works again.
    monkeypatch.undo()
    for _ in range(10):
        mpc.compute(q, np.zeros(7), q, np.zeros(7), engine.dt)
    assert not mpc.using_fallback and mpc.stats.updates[-1].status == "solved"


def test_recovery_after_fallback_anchors_to_applied_command(engine, monkeypatch):
    """After a fallback the servo holds the PID's command, so the next QP must be
    anchored (linearization point, command-rate penalty, slew limit) to that
    command, not the stale MPC solution. v0.2 used the stale one, and fallbacks
    cascaded in the wrist benchmark (275 -> 43 of 450 updates after the fix)."""
    mpc = make_mpc(engine)
    engine.reset(HOME_QPOS)
    anchors = []
    qp_data = JointMPCController._qp_data

    def spy(self, lin, x0, u_prev, *rest):
        anchors.append(u_prev.copy())
        return qp_data(self, lin, x0, u_prev, *rest)

    monkeypatch.setattr(JointMPCController, "_qp_data", spy)

    def step(target):
        u = mpc.compute(engine.arm_qpos(), engine.arm_qvel(), target, np.zeros(7), engine.dt, t=engine.time)
        engine.set_arm_command(u)
        engine.step()
        return u

    for _ in range(10):
        step(HOME_QPOS)
    stale = mpc._command.copy()
    solve = mpc._solver.solve

    def broken(*a, **k):
        raise RuntimeError("solver crashed")

    mpc._solver.solve = broken
    applied = [step(HOME_QPOS + 0.1) for _ in range(10)][-1]
    mpc._solver.solve = solve
    step(HOME_QPOS + 0.1)
    assert np.abs(applied - stale).max() > 2 * mpc.slew.max()  # the two candidates differ
    np.testing.assert_array_equal(anchors[-1], applied)


def test_infeasible_problem_activates_fallback(engine):
    """Hard constraints and a measured speed no command can bring under the
    velocity limit within one interval: OSQP reports infeasibility."""
    mpc = make_mpc(engine, soft_constraints=False)
    qd = np.zeros(7)
    qd[0] = 5.0           # 2.3x joint 1's speed limit; cannot brake below it in 20 ms
    q = HOME_QPOS
    mpc.compute(q, qd, q, np.zeros(7), engine.dt, t=0.0)
    upd = mpc.stats.updates[-1]
    assert upd.fallback and "infeasible" in upd.status
    # With soft constraints the same state is solvable (slack absorbs it).
    soft = make_mpc(engine, solver_settings=OFFLINE_SOLVER)
    soft.compute(q, qd, q, np.zeros(7), engine.dt, t=0.0)
    assert not soft.stats.updates[-1].fallback and soft.stats.updates[-1].max_slack > 0.1


def test_step_reference_exceeds_realtime_budget_and_falls_back(config):
    """A step reference (no preview) is the hardest QP here: it needs more
    iterations than the real-time budget, so updates fall back to PID. The arm
    must still reach the target without NaNs or limit violations."""
    cfg = mpc_config(config, reference_preview=False)
    cfg["task"] = {**config["task"], "move_time": 0.01, "hold_time": 2.0, "loop": False, "trajectory": None,
                   "waypoints": {"home": HOME_QPOS.tolist(), "left": LEFT.tolist()}}
    r = SimulationRunner(cfg)
    log = r.run_headless(2.0, log_every=1).arrays()
    s = r.controller.stats.summary(r.controller.control_period)
    assert s["fallbacks"] > 0
    assert any(u.status == "maximum iterations reached" for u in r.controller.stats.updates)
    assert np.all(np.isfinite(log["q"]))
    lo, hi = r.engine.joint_limits[:, 0], r.engine.joint_limits[:, 1]
    assert np.all((log["q"] >= lo) & (log["q"] <= hi))
    np.testing.assert_allclose(r.engine.arm_qpos(), LEFT, atol=2e-3)


def test_non_finite_measurement_activates_fallback(engine):
    mpc = make_mpc(engine)
    q = HOME_QPOS.copy()
    q[2] = np.nan
    mpc.compute(q, np.zeros(7), HOME_QPOS, np.zeros(7), engine.dt, t=0.0)
    assert mpc.using_fallback and "non-finite" in mpc.stats.updates[-1].status


# --- Switching and configuration ---------------------------------------------


@pytest.mark.parametrize("kind, cls", [("joint_pid", JointPIDController), ("pid", JointPIDController),
                                       ("joint_mpc", JointMPCController), ("mpc", JointMPCController)])
def test_controller_selected_by_config(config, kind, cls):
    cfg = {**config, "controller": {**config["controller"], "type": kind}}
    assert isinstance(SimulationRunner(cfg).controller, cls)


def test_unknown_controller_rejected(config, engine):
    with pytest.raises(ValueError, match="Unknown controller"):
        build_controller({"type": "lqr"}, engine)


def test_control_rate_configurable(config, engine):
    assert build_controller(mpc_config(config, control_rate_hz=100)["controller"], engine).substeps == 5
    with pytest.raises(ValueError, match="divide"):
        build_controller(mpc_config(config, control_rate_hz=30)["controller"], engine)


def test_same_task_runs_with_both_controllers(config):
    """Identical task and start state; both controllers reach the same waypoint."""
    finals = {}
    for kind in ("joint_pid", "joint_mpc"):
        cfg = {**config, "controller": {**config["controller"], "type": kind}}
        r = SimulationRunner(cfg)
        np.testing.assert_allclose(r.engine.arm_qpos(), HOME_QPOS)
        r.run_headless(r.task.segment_time)
        finals[kind] = r.engine.arm_qpos()
    np.testing.assert_allclose(finals["joint_pid"], LEFT, atol=2e-3)
    np.testing.assert_allclose(finals["joint_mpc"], LEFT, atol=2e-3)


# --- Benchmark --------------------------------------------------------------


def test_benchmark_waypoints_within_limits_and_collision_free():
    bench = yaml.safe_load(BENCHMARK_CONFIG.read_text())
    engine = SimulationEngine(bench["environment"])
    lo, hi = engine.joint_limits[:, 0], engine.joint_limits[:, 1]
    for sc_name, sc in bench["scenarios"].items():
        if "targets" in sc:  # Cartesian scenario: IK + collision checks decide (tested in test_reach.py)
            from simulation.kinematics import ArmKinematics

            kin = ArmKinematics(engine)
            plans = [kin.plan_target(np.asarray(t["position"]), HOME_QPOS) for t in sc["targets"]]
            assert sum(p.success for p in plans) >= 3, sc_name
            continue
        for name, q in sc["waypoints"].items():
            q = np.asarray(q)
            assert np.all((q >= lo) & (q <= hi)), (sc_name, name)
            engine.reset(q)
            assert engine.robot_contacts() == [], (sc_name, name)
            assert engine.ee_position()[2] > 0.55, (sc_name, name)


def test_settling_time_metric():
    t = np.arange(0, 1, 0.1)
    err = np.array([1, 1, 0.5, 0.0, 0.0, 0.01, 0.0, 0, 0, 0])
    assert settling_times(t, err, [(0.0, 1.0)], tol=0.02) == [pytest.approx(0.3)]
    assert settling_times(t, err, [(0.0, 1.0)], tol=0.005) == [pytest.approx(0.6)]
    assert np.isnan(settling_times(t, np.ones(10), [(0.0, 1.0)], tol=0.1)[0])


def test_benchmark_runs_and_writes_results(tmp_path):
    bench = yaml.safe_load(BENCHMARK_CONFIG.read_text())
    bench["scenarios"] = {"push": {**bench["scenarios"]["push"], "duration": 2.5}}
    bench["controllers"] = {k: bench["controllers"][k] for k in ("pid", "mpc")}
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(bench))
    report = run_benchmark(out_dir=tmp_path / "out", plots=False, bench_path=path, verbose=False)
    assert (tmp_path / "out" / "results.json").is_file() and (tmp_path / "out" / "results.csv").is_file()
    rows = {r["controller"]: r for r in report["results"]}
    assert set(rows) == {"pid", "mpc"}
    for r in rows.values():
        assert r["finite_state"] and r["joint_limit_violation_steps"] == 0
        # The 30 N push really displaces the arm (the disturbance is applied).
        assert r["max_error_rad"] > 1e-3
    assert rows["mpc"]["mpc_updates"] == 125 and "mpc_solve_time_p95_ms" in rows["mpc"]
