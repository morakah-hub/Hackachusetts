"""Audit of the torque MPC (controllers/torque_mpc.py): objective, constraints, signs, prediction.

v0.6 milestone 1. The servo MPC's QP was audited in v0.2 (tools/audit_mpc_objective.py); the
torque MPC's had not been. This script checks the assembled OSQP problem against the documented
formulation with code written independently of the controller's matrix assembly:

  1. objective: the documented cost, evaluated term by term from (x, tau, s), against
     1/2 z'Pz + q'z of the assembled QP on random pairs of z (the constant term cancels);
  2. a hand-checkable case: 1 joint, N = 2, a known affine model; the controller's first torque
     is compared with the minimizer of the documented cost found by eliminating the states and
     solving the 2x2 normal equations (unconstrained), and by a bounded scalar search with an
     active torque limit;
  3. constraint rows: dynamics rows reproduce x[k+1] = A x[k] + B tau[k] + c (+ B d_hat), torque
     rows read tau[k] / tau_scale, state rows read x[k] +- s[k], and the bounds sit on the
     matching rows (q bounds on q entries, velocity bounds on qd entries);
  4. disturbance sign: a known input disturbance d enters the prediction as B d, the hold torque
     is tau_g - d, and the feedthrough subtracts the change of d_hat since the update; on the real
     plant, a constant joint torque applied through qfrc_applied is estimated with the same sign;
  5. prediction vs nonlinear MuJoCo rollouts mid-motion: one interval (exactness of c at the
     linearization point) and over the horizon (frozen-Jacobian LTI error), and a re-linearized
     (time-varying) prediction to separate the frozen-Jacobian approximation from a bug;
  6. OSQP statuses and residuals over a closed-loop run, and how "solved inaccurate" is handled.

    .\\.venv\\Scripts\\python.exe tools\\audit_torque_mpc.py [--output results\\v0.6\\mpc_audit.json]
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.base import Controller  # noqa: E402
from controllers.torque_mpc import TorqueMPCController, TorqueMPCWeights  # noqa: E402

# ---------------------------------------------------------------- stubs for small, exact cases


@dataclass
class _Lin:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    plausible: bool = True


class StubModel:
    """Minimal prediction model with a fixed, known affine interval map (no MuJoCo)."""

    actuation = "torque"

    def __init__(self, A, B, c, gravity, force_limits, substeps=10, dt=0.02):
        self.A, self.B, self.c = (np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float)),
                                  np.asarray(c, float))
        self.dof = self.B.shape[1]
        self.substeps, self.dt = substeps, dt
        self.force_limits = np.asarray(force_limits, float)
        self._g = np.asarray(gravity, float)

    def gravity_torque(self, q):
        return self._g.copy()

    def linearize(self, q, qd, u):
        return _Lin(self.A, self.B, self.c)


class StubObserver:
    """Observer with a settable estimate (no dynamics)."""

    def __init__(self, d):
        self.estimate = np.asarray(d, float).copy()

    def reset(self):
        pass

    def update(self, q, qd, dt):
        return self.estimate

    def applied(self, tau):
        pass


class _ZeroFallback(Controller):
    def reset(self):
        pass

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        return np.zeros(len(q))


# ---------------------------------------------------------------- 1. objective


def direct_cost(mpc: TorqueMPCController, z: np.ndarray, refs: np.ndarray, tau_hold: np.ndarray,
                tau_prev: np.ndarray) -> float:
    """Documented cost, term by term (torque in Nm, normalized here, not taken from P)."""
    N, nx, nu = mpc.N, mpc.nx, mpc.nu
    X = z[:N * nx].reshape(N, nx)
    V = z[N * nx:N * (nx + nu)].reshape(N, nu)      # decision variables: tau / tau_scale
    tau = V * mpc.tau_scale
    S = z[N * (nx + nu):].reshape(N, nx) if mpc.soft else np.zeros((N, nx))
    qx = np.concatenate([np.broadcast_to(mpc.w.position, (mpc.dof,)), np.broadcast_to(mpc.w.velocity, (mpc.dof,))])
    we = np.broadcast_to(mpc.w.effort, (mpc.dof,))
    wr = np.broadcast_to(mpc.w.torque_rate, (mpc.dof,))
    J = 0.0
    for k in range(N):
        e = X[k] - refs[k]
        J += (mpc.w.terminal if k == N - 1 else 1.0) * float(np.sum(qx * e * e))
        a = (tau[k] - tau_hold) / mpc.tau_scale
        J += float(np.sum(we * a * a))
        r = (tau[k] - (tau_prev if k == 0 else tau[k - 1])) / mpc.tau_scale
        J += float(np.sum(wr * r * r))
    if mpc.soft:
        J += mpc.w.slack_quadratic * float(np.sum(S * S)) + mpc.w.slack_linear * float(np.sum(S))
    return J


def captured_qp(mpc: TorqueMPCController):
    """(P full, q, A, l, u) of the controller's last assembled QP (rebuilt from its stored data)."""
    solver = mpc._solver
    if solver is None:
        raise RuntimeError("controller has not solved a QP yet")
    P = mpc._P + sp.triu(mpc._P, 1).T
    A = sp.csc_matrix((mpc._Ax.copy(), mpc._csc_indices, mpc._csc_indptr), shape=(mpc.nc, mpc.nz))
    return P, A


def objective_mismatch(mpc: TorqueMPCController, h: np.ndarray, refs, tau_hold, tau_prev, pairs=20, seed=0) -> float:
    """Max relative mismatch of cost differences, documented vs assembled, over random z pairs."""
    P, _ = captured_qp(mpc)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(pairs):
        z1, z2 = rng.normal(size=mpc.nz), rng.normal(size=mpc.nz)
        if mpc.soft:  # slack is >= 0 in the QP; the cost formula holds for any value
            z1[-mpc.N * mpc.nx:] = np.abs(z1[-mpc.N * mpc.nx:])
            z2[-mpc.N * mpc.nx:] = np.abs(z2[-mpc.N * mpc.nx:])
        dd = direct_cost(mpc, z1, refs, tau_hold, tau_prev) - direct_cost(mpc, z2, refs, tau_hold, tau_prev)
        dq = 0.5 * z1 @ (P @ z1) + 2 * h @ z1 - (0.5 * z2 @ (P @ z2) + 2 * h @ z2)
        worst = max(worst, abs(dd - dq) / max(1.0, abs(dd)))
    return worst


def _linear_cost(mpc: TorqueMPCController, refs, tau_hold, tau_prev) -> np.ndarray:
    """h such that the QP's linear term is 2h (as the controller builds it), from its formula."""
    N, nx, nu = mpc.N, mpc.nx, mpc.nu
    nX, nU = N * nx, N * nu
    h = np.zeros(mpc.nz)
    w = np.ones(N)
    w[-1] = mpc.w.terminal
    h[:nX] = -(w[:, None] * mpc._Qx[None, :] * refs).ravel()
    h[nX:nX + nU] = -np.tile(mpc._we * tau_hold / mpc.tau_scale, N)
    h[nX:nX + nu] -= mpc._wr * tau_prev / mpc.tau_scale
    h[nX + nU:] = mpc._h_slack
    return h


# ---------------------------------------------------------------- 2. hand-checkable case


def small_case(tau_limit: float = 50.0, soft: bool = False, d_hat: float = 0.0):
    """1 joint, N = 2, known affine model. Returns (controller, inputs, expected first torque)."""
    dt = 0.02
    I, b = 0.5, 0.1  # inertia (kg m^2), viscous damping (Nm s/rad)
    A = np.array([[1.0, dt], [0.0, 1.0 - b * dt / I]])
    B = np.array([[0.5 * dt * dt / I], [dt / I]])
    g = np.array([9.81 * 0.3])                         # gravity-like load: holding needs tau_g = 2.943 Nm
    c = -B[:, 0] * g[0]
    model = StubModel(A, B, c, g, np.array([[-tau_limit, tau_limit]]), substeps=10, dt=dt)
    w = TorqueMPCWeights(position=100.0, velocity=1.0, terminal=2.0, effort=0.5, torque_rate=0.2,
                         slack_quadratic=1e3, slack_linear=1e6)
    obs = StubObserver([d_hat]) if d_hat else None
    mpc = TorqueMPCController(model, _ZeroFallback(), horizon=2, weights=w, soft_constraints=soft,
                              observer=obs, solver_settings={"eps_abs": 1e-10, "eps_rel": 1e-10,
                                                             "max_iter": 20000})
    q0, qd0, qref = 0.0, 0.0, 0.1
    tau_prev = g.copy()  # first update: tau_prev = tau_g

    def cost(u0, u1):
        """Documented cost with the states eliminated by hand (x1, x2 written out)."""
        x0 = np.array([q0, qd0])
        x1 = A @ x0 + B[:, 0] * (u0 + d_hat) + c
        x2 = A @ x1 + B[:, 0] * (u1 + d_hat) + c
        r = np.array([qref, 0.0])
        J = 100 * (x1[0] - r[0]) ** 2 + 1 * (x1[1] - r[1]) ** 2
        J += 2.0 * (100 * (x2[0] - r[0]) ** 2 + 1 * (x2[1] - r[1]) ** 2)
        hold = g[0] - d_hat
        J += 0.5 * (((u0 - hold) / tau_limit) ** 2 + ((u1 - hold) / tau_limit) ** 2)
        J += 0.2 * (((u0 - tau_prev[0]) / tau_limit) ** 2 + ((u1 - u0) / tau_limit) ** 2)
        return J

    # Quadratic in (u0, u1): exact gradient/Hessian from 3-point differences of a quadratic.
    def grad_hess():
        h = 1.0
        f = lambda u: cost(u[0], u[1])  # noqa: E731
        z = np.zeros(2)
        H = np.zeros((2, 2))
        gvec = np.zeros(2)
        for i in range(2):
            ei = np.eye(2)[i] * h
            gvec[i] = (f(z + ei) - f(z - ei)) / (2 * h)
            for j in range(2):
                ej = np.eye(2)[j] * h
                H[i, j] = (f(z + ei + ej) - f(z + ei - ej) - f(z - ei + ej) + f(z - ei - ej)) / (4 * h * h)
        return gvec, H

    gvec, H = grad_hess()
    u_unc = np.linalg.solve(H, -gvec)
    # Bounded: minimize over u0 with u1 optimal given u0, both clipped to the limit (box-constrained 2-D QP
    # solved by nested scalar searches; the cost is convex so this finds the global minimizer).
    def best_u1(u0):
        r = minimize_scalar(lambda u1: cost(u0, u1), bounds=(-tau_limit, tau_limit), method="bounded",
                            options={"xatol": 1e-10})
        return r.x
    r0 = minimize_scalar(lambda u0: cost(u0, best_u1(u0)), bounds=(-tau_limit, tau_limit), method="bounded",
                         options={"xatol": 1e-10})
    return mpc, (q0, qd0, qref), {"unconstrained": u_unc, "bounded_u0": float(r0.x), "cost": cost}


def run_small_case(tau_limit=50.0, d_hat=0.0, soft=False) -> dict:
    mpc, (q0, qd0, qref), exp = small_case(tau_limit, soft, d_hat)
    out = mpc.compute(np.array([q0]), np.array([qd0]), np.array([qref]), np.array([0.0]), 0.002, t=0.0)
    taus = mpc.last_prediction["commands"][:, 0]
    res = {"tau_limit": tau_limit, "d_hat": d_hat, "controller_plan": taus.tolist(), "applied": float(out[0]),
           "expected_unconstrained": exp["unconstrained"].tolist(), "expected_bounded_u0": exp["bounded_u0"]}
    # Predicted states must equal the hand recursion with the plan (B d enters as a known input).
    A, B, c = mpc.model.A, mpc.model.B[:, 0], mpc.model.c
    x = np.array([q0, qd0])
    pred = []
    for u in taus:
        x = A @ x + B * (u + d_hat) + c
        pred.append(x.copy())
    res["prediction_vs_recursion"] = float(np.abs(np.array(pred) - mpc.last_prediction["states"]).max())
    # Objective check on this small QP too.
    hold = mpc.model.gravity_torque(None) - d_hat
    refs = mpc._references(0.0, np.array([qref]), np.array([0.0]))
    h = _linear_cost(mpc, refs, hold, mpc.model.gravity_torque(None))
    res["objective_mismatch"] = objective_mismatch(mpc, h, refs, hold, mpc.model.gravity_torque(None))
    return res


# ---------------------------------------------------------------- 3. constraint rows


def constraint_rows(mpc: TorqueMPCController, lin, x0, d_hat, seed=1) -> dict:
    """Evaluate A z for a z built by hand and compare with the documented row meanings."""
    _, A = captured_qp(mpc)
    N, nx, nu, n = mpc.N, mpc.nx, mpc.nu, mpc.dof
    nX, nU = N * nx, N * nu
    rng = np.random.default_rng(seed)
    tau = rng.uniform(-0.5, 0.5, (N, nu)) * mpc.tau_scale
    X = []
    x = np.asarray(x0, float)
    for k in range(N):
        x = lin.A @ x + lin.B @ (tau[k] + d_hat) + lin.c
        X.append(x)
    X = np.array(X)
    S = rng.uniform(0, 0.01, (N, nx)) if mpc.soft else None
    z = np.concatenate([X.ravel(), (tau / mpc.tau_scale).ravel()] + ([S.ravel()] if mpc.soft else []))
    rows = A @ z
    lo = mpc._solver_bounds[0] if hasattr(mpc, "_solver_bounds") else mpc._lo
    out = {}
    out["dynamics_row_residual"] = float(np.abs(rows[:nX] - lo[:nX]).max())
    out["torque_rows_vs_tau_over_scale"] = float(np.abs(rows[mpc._row_u:mpc._row_u + nU] -
                                                        (tau / mpc.tau_scale).ravel()).max())
    r = mpc._row_x
    if mpc.soft:
        out["state_lower_rows_vs_x_plus_s"] = float(np.abs(rows[r:r + nX] - (X + S).ravel()).max())
        out["state_upper_rows_vs_x_minus_s"] = float(np.abs(rows[r + nX:r + 2 * nX] - (X - S).ravel()).max())
        out["slack_rows_vs_s"] = float(np.abs(rows[r + 2 * nX:r + 3 * nX] - S.ravel()).max())
        lb = mpc._lo[r:r + nX].reshape(N, nx)
        ub = mpc._hi[r + nX:r + 2 * nX].reshape(N, nx)
    else:
        out["state_rows_vs_x"] = float(np.abs(rows[r:r + nX] - X.ravel()).max())
        lb = mpc._lo[r:r + nX].reshape(N, nx)
        ub = mpc._hi[r:r + nX].reshape(N, nx)
    exp_lb = np.r_[mpc.q_bounds[:, 0], -mpc.v_max * (1 - mpc.velocity_margin)]
    exp_ub = np.r_[mpc.q_bounds[:, 1], mpc.v_max * (1 - mpc.velocity_margin)]
    out["state_bounds_on_matching_entries"] = float(max(np.abs(lb - exp_lb).max(), np.abs(ub - exp_ub).max()))
    tl = mpc._lo[mpc._row_u:mpc._row_u + nU].reshape(N, nu) * mpc.tau_scale
    th = mpc._hi[mpc._row_u:mpc._row_u + nU].reshape(N, nu) * mpc.tau_scale
    out["torque_bounds_vs_limits_nm"] = float(max(np.abs(tl - mpc.torque_limits[:, 0]).max(),
                                                  np.abs(th - mpc.torque_limits[:, 1]).max()))
    return out


# ---------------------------------------------------------------- 4. disturbance signs


def feedthrough_sign() -> dict:
    """Held torque between updates = tau_mpc - (d_hat_now - d_hat_at_update)."""
    mpc, (q0, qd0, qref), _ = small_case(50.0, False, 0.0)
    obs = StubObserver([0.0])
    mpc.observer, mpc.disturbance_feedthrough = obs, True
    mpc.reset()
    q, qd = np.array([q0]), np.array([qd0])
    first = mpc.compute(q, qd, np.array([qref]), np.array([0.0]), 0.002, t=0.0)
    obs.estimate = np.array([1.5])  # disturbance grows by 1.5 Nm before the next update
    second = mpc.compute(q, qd, np.array([qref]), np.array([0.0]), 0.002, t=0.002)
    return {"first": float(first[0]), "between_updates": float(second[0]),
            "expected_between_updates": float(first[0] - 1.5)}


def plant_observer_sign(torque=(0.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.5), seconds=0.6) -> dict:
    """Apply a constant joint torque through qfrc_applied (unmeasured) while CTC + observer holds
    the arm; the estimate must converge to +torque (plant: M qdd + h = tau + d)."""
    from simulation.runner import SimulationRunner, load_config

    cfg = load_config()
    cfg["controller"]["type"] = "torque_ctc"
    cfg["controller"]["ctc"]["disturbance_observer"]["enabled"] = True
    cfg["task"]["waypoints"] = {"a": list(np.r_[0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]),
                                "b": list(np.r_[0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])}
    r = SimulationRunner(cfg)
    e = r.engine
    d = np.asarray(torque, float)

    def push(runner):
        runner.engine.data.qfrc_applied[runner.engine.arm_dof_adr] = d

    r.run_headless(seconds, before_step=push)
    est = r.controller.disturbance_estimate
    return {"applied_nm": d.tolist(), "estimate_nm": est.round(4).tolist(), "max_error_nm": float(np.abs(est - d).max()),
            "final_deflection_rad": float(np.abs(e.arm_qpos() - r.task.waypoints[0]).max())}


# ---------------------------------------------------------------- 5. prediction agreement


def prediction_agreement(capture_times=(0.8, 1.6, 2.4), relinearize=True) -> list[dict]:
    """Mid-motion on the default waypoint task (torque MPC, observer on): compare the accepted plan's
    predicted states with nonlinear MuJoCo rollouts of the same torque sequence."""
    from simulation.runner import SimulationRunner, load_config

    cfg = load_config()
    cfg["controller"]["type"] = "torque_mpc"
    r = SimulationRunner(cfg)
    mpc = r.controller
    out = []
    for tc in capture_times:
        while r.engine.time < tc - 1e-9:
            r.step()
        # Force an update now so last_prediction belongs to the current state.
        while mpc._step % mpc.substeps != 0:
            r.step()
        r.step()
        p = mpc.last_prediction
        x0, plan, pred = p["x0"], p["commands"], p["states"]
        n = mpc.dof
        d = p["disturbance"]
        roll = mpc.model.rollout(x0[:n], x0[n:], plan + d)  # same input the prediction assumed
        err = np.abs(roll - pred)
        row = {"time": float(p["time"]), "one_interval_q_err_rad": float(err[0, :n].max()),
               "one_interval_qd_err_rad_s": float(err[0, n:].max()),
               "horizon_q_err_rad": float(err[:, :n].max()), "horizon_qd_err_rad_s": float(err[:, n:].max()),
               "q_err_by_k": err[:, :n].max(axis=1).tolist(), "disturbance_estimate_nm": d.tolist()}
        if relinearize:  # time-varying model: re-linearize at each nonlinear state along the plan
            x = x0.copy()
            tv = []
            for k in range(len(plan)):
                lin = mpc.model.linearize(x[:n], x[n:], plan[k] + d)
                x = lin.A @ x + lin.B @ (plan[k] + d) + lin.c
                tv.append(x.copy())
            row["relinearized_horizon_q_err_rad"] = float(np.abs(np.array(tv) - roll)[:, :n].max())
        out.append(row)
    return out


# ---------------------------------------------------------------- 6. statuses and residuals


def closed_loop_statuses(config="configs/planning_demo.yaml", seconds=9.0) -> dict:
    from simulation.runner import SimulationRunner, load_config

    r = SimulationRunner(load_config(config))
    r.run_headless(seconds)
    ups = r.controller.stats.updates
    statuses = {}
    for u in ups:
        statuses[u.status] = statuses.get(u.status, 0) + 1
    prim = np.array([u.prim_res for u in ups if np.isfinite(u.prim_res)])
    dual = np.array([u.dual_res for u in ups if np.isfinite(u.dual_res)])
    return {"config": config, "updates": len(ups), "statuses": statuses,
            "fallbacks": int(sum(u.fallback for u in ups)),
            "prim_res_max": float(prim.max()) if prim.size else None,
            "dual_res_max": float(dual.max()) if dual.size else None,
            "prim_res_median": float(np.median(prim)) if prim.size else None,
            "max_iterations": int(max(u.iterations for u in ups))}


def full_size_checks() -> dict:
    """Objective and constraint rows on the real 7-joint controller mid-motion."""
    from simulation.runner import SimulationRunner, load_config

    cfg = load_config()
    cfg["controller"]["type"] = "torque_mpc"
    r = SimulationRunner(cfg)
    while r.engine.time < 1.2:
        r.step()
    mpc = r.controller
    p = mpc.last_prediction
    tau_hold = p["gravity"] - p["disturbance"]
    tau_prev = mpc._tau_prev_at_update
    h = _linear_cost(mpc, p["references"], tau_hold, tau_prev)
    out = {"objective_mismatch": objective_mismatch(mpc, h, p["references"], tau_hold, tau_prev),
           "nz": mpc.nz, "nc": mpc.nc}
    lin = p["model"]
    out.update(constraint_rows(mpc, lin, p["x0"], p["disturbance"]))
    P, _ = captured_qp(mpc)
    ev = np.linalg.eigvalsh(P.toarray())
    out["P_eig_min"], out["P_eig_max"] = float(ev.min()), float(ev.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="results/v0.6/mpc_audit.json")
    args = ap.parse_args()
    report = {}
    print("1-2. hand-checkable 1-joint N=2 case")
    report["small_unconstrained"] = run_small_case(tau_limit=50.0)
    report["small_bounded"] = run_small_case(tau_limit=3.2)
    report["small_with_disturbance"] = run_small_case(tau_limit=50.0, d_hat=1.0)
    for k in ("small_unconstrained", "small_bounded", "small_with_disturbance"):
        print(f"  {k}: {json.dumps(report[k])}")
    print("1+3. full-size objective and constraint rows")
    report["full_size"] = full_size_checks()
    print(f"  {json.dumps(report['full_size'])}")
    print("4. disturbance signs")
    report["feedthrough"] = feedthrough_sign()
    report["plant_observer"] = plant_observer_sign()
    print(f"  feedthrough: {report['feedthrough']}\n  plant observer: {report['plant_observer']}")
    print("5. prediction vs nonlinear rollouts")
    report["prediction"] = prediction_agreement()
    for row in report["prediction"]:
        print(f"  t={row['time']:.2f}: 1-interval q err {row['one_interval_q_err_rad']:.2e} rad, horizon q err "
              f"{row['horizon_q_err_rad']:.2e} rad, re-linearized {row.get('relinearized_horizon_q_err_rad', 0):.2e} rad")
    print("6. statuses and residuals (wall demo, 9 s)")
    report["statuses"] = closed_loop_statuses()
    print(f"  {report['statuses']}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
