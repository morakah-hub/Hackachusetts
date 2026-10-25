"""v0.7 workbench: inspectors and explanations agree with the code and the recorded data; replay; sandbox;
observation noise and force pulses; guided experiments keep failures."""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import audit_torque_mpc as audit  # noqa: E402
from controllers.dynamics import ArmInverseDynamics  # noqa: E402
from workbench import inspect as insp  # noqa: E402
from workbench.api import RunRequest  # noqa: E402
from workbench.runs import LiveRun, Recording, run_to_end, save_run  # noqa: E402
from workbench.sandbox import simulate, unconstrained_first_torque  # noqa: E402


@pytest.fixture(scope="module")
def mpc_run():
    run = LiveRun(RunRequest(scenario="pick_place", controller="torque_mpc", observer=True, payload_kg=1.15))
    while run.engine.time < 4.0 and run.step():   # into the lift / hold phases
        pass
    return run


def test_mpc_inspector_objective_equals_the_solver_objective_plus_its_constant(mpc_run):
    mpc = mpc_run.controller
    while mpc._step % mpc.substeps != 1:          # stop right after an update
        mpc_run.step()
    rec = Recording.from_live(mpc_run)
    snap = rec.snapshots[-1]
    assert snap["has_prediction"] and abs(snap["time"] - mpc.last_prediction["time"]) < 1e-12
    p = mpc.last_prediction
    hold = p["gravity"] - p["disturbance"]
    h = audit._linear_cost(mpc, p["references"], hold, mpc._tau_prev_at_update)
    P = mpc._P + sp.triu(mpc._P, 1).T
    z = mpc._z_prev
    solver_obj = 0.5 * z @ (P @ z) + 2 * h @ z
    terms = insp.objective_terms(snap, rec.meta["mpc_weights"])
    assert terms["solver_objective"] == pytest.approx(solver_obj, rel=1e-7, abs=1e-6)
    assert terms["total"] == pytest.approx(terms["tracking"] + terms["effort"] + terms["rate"] + terms["slack"])
    # The command path shown is the recorded one.
    sections = insp.mpc(rec, len(rec.t) - 1, 1)
    path = next(s for s in sections if s.get("title") == "Command path (Nm unless stated)")
    assert path["table"][0][1:] == [pytest.approx(v, abs=1e-3) for v in snap["commands"][0]]


def test_explanations_and_inspector_values_come_from_the_recording(mpc_run):
    rec = Recording.from_live(mpc_run)
    i = len(rec.t) - 20
    t = float(rec.t[i])
    ex = insp.explain(rec, t)
    snap = rec.snapshot_at(t)
    joined = "\n".join(ex["lines"])
    assert f"Latest MPC update at {snap['time']:.3f} s" in joined
    assert f"Phase '{rec.data['phase'][i]}'" in joined
    dh = rec.data["d_hat"][i]
    k = int(np.argmax(np.abs(dh)))
    assert f"{dh[k]:.2f} Nm on joint {k + 1}" in joined
    mot = insp.motion(rec, i, 0)
    table = next(s for s in mot if s.get("title", "").startswith("Measured vs reference"))["table"]
    assert [row[1] for row in table] == [pytest.approx(v, abs=1e-4) for v in rec.data["q"][i]]
    # Local approximation check: Jacobian prediction close to the scratch FK difference (second-order gap).
    jac = next(s for s in mot if s.get("title", "").startswith("Local approximation"))["table"]
    assert jac[2][4] < 0.05 * jac[1][4]


def test_saved_run_reopens_with_the_same_recorded_moment(mpc_run, tmp_path):
    run = mpc_run
    run.stop("test")
    run.finish()
    d = save_run(run, tmp_path)
    live, saved = Recording.from_live(run), Recording.load(d)
    t = 2.345
    i, j = live.index(t), saved.index(t)
    assert i == j
    for f in ("q", "q_ref", "tau", "qpos", "obj_pos", "d_hat"):
        assert np.array_equal(live.data[f][i], saved.data[f][j], equal_nan=True), f
    assert live.data["phase"][i] == saved.data["phase"][j]
    assert live.snapshot_at(t) == saved.snapshot_at(t)
    assert insp.explain(live, t)["lines"] == insp.explain(saved, t)["lines"]


def test_sandbox_matches_the_closed_form_and_the_textbook_example():
    r = simulate(inertia=0.5, torque_limit=100.0, horizon=20, w_position=10, w_velocity=1, w_effort=0.1, target=1.0)
    assert r["tau0"] == pytest.approx(r["tau0_unconstrained"], rel=1e-6)
    # N = 2 by hand: states eliminated, 2x2 normal equations.
    Ts, I = 0.1, 0.5
    A = np.array([[1, Ts], [0, 1]])
    B = np.array([Ts * Ts / (2 * I), Ts / I])
    G1 = np.array([[B[0], 0], [B[1], 0]])
    G2 = np.column_stack([A @ B, B])
    Q = np.diag([10.0, 1.0])
    r_ = np.array([1.0, 0.0])
    H = G1.T @ Q @ G1 + G2.T @ Q @ G2 + np.eye(2) * 0.1 / 3.0**2
    g = -(G1.T @ Q @ r_ + G2.T @ Q @ r_)
    assert unconstrained_first_torque(0.5, 3.0, 2, 10, 1, 0.1, 1.0) == pytest.approx(np.linalg.solve(H, -g)[0], rel=1e-10)
    lim = simulate(torque_limit=1.0)   # the v0.6 textbook example: saturated push, then braking
    assert lim["tau"][:6] == pytest.approx([1.0] * 6, abs=1e-6) and lim["tau"][7] < 0
    assert lim["tau0_unconstrained"] == pytest.approx(6.434, abs=1e-3)


def test_zero_noise_is_the_baseline_and_noise_reaches_only_the_controller():
    base = RunRequest(scenario="table_reach", controller="torque_ctc", observer=True, mpc_snapshots=False)
    r0, rec0 = run_to_end(base, save=False)
    rz, recz = run_to_end(base.replace(noise={"position_std": 0.0, "velocity_std": 0.0, "seed": 3}), save=False)
    assert np.array_equal(rec0.data["q"], recz.data["q"])
    noisy = base.replace(noise={"position_std": 0.0005, "velocity_std": 0.005, "seed": 3})
    r1, rec1 = run_to_end(noisy, save=False)
    r2, rec2 = run_to_end(noisy, save=False)
    assert np.array_equal(rec1.data["q"], rec2.data["q"])            # seeded, reproducible
    assert not np.array_equal(rec1.data["q"], rec0.data["q"])        # noise changed the commands ...
    assert np.abs(rec1.data["q"] - rec0.data["q"]).max() < 0.01      # ... but the recorded q is the true state


def test_controllers_never_see_the_external_force():
    run = LiveRun(RunRequest(scenario="table_reach", controller="torque_ctc", observer=False))
    e = run.engine
    dyn = ArmInverseDynamics(e.controller_model, e.data, e.arm_qpos_adr, e.arm_dof_adr)
    q, qd = e.arm_qpos(), e.arm_qvel()
    before = dyn.torque(q, qd, np.ones(7))
    e.data.xfrc_applied[e.model.body("hand").id, :3] = [0, 0, -50]
    assert np.array_equal(dyn.torque(q, qd, np.ones(7)), before)
    e.data.xfrc_applied[:] = 0


def test_guided_experiment_keeps_and_explains_the_failure(tmp_path):
    from workbench.experiments import start
    from workbench.session import Session
    s = Session(runs_dir=tmp_path)
    try:
        exp = start(s, "grip")
        t0 = time.time()
        while exp["state"] == "running" and time.time() - t0 < 300:
            time.sleep(0.2)
        assert exp["state"] == "done"
        outcomes = [r["outcome"] for r in exp["runs"]]
        assert outcomes == ["success", "failure"]
        assert exp["runs"][1]["failure_stage"] == "lift"
        lines = "\n".join(exp["analysis"]["lines"])
        assert "weak (Menagerie): FAILURE at lift" in lines
        assert len(list(tmp_path.iterdir())) == 2
    finally:
        s.close()
    assert not s.alive


def test_workspace_scan_separates_blocked_from_not_found():
    from workbench.inspect import reach_scan
    r = reach_scan(RunRequest(scenario="pick_place_obstacle"), step=0.06)
    by = {(round(p["x"], 2), round(p["y"], 2)): p for p in r["points"]}
    assert r["counts"]["found"] > 0 and r["counts"]["blocked"] > 0
    blocked = [p for p in r["points"] if p["class"] == "blocked"]
    assert all(abs(p["y"]) < 0.15 for p in blocked)          # only next to the obstacle (y = +-0.02)
    assert by[(0.54, 0.22)]["class"] == "found"
    free = reach_scan(RunRequest(scenario="pick_place"), step=0.06)
    assert free["counts"]["blocked"] == 0
