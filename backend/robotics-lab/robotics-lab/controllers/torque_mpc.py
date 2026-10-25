"""Receding-horizon joint-space MPC for torque-actuated arms (OSQP).

Decision variables are the joint torques tau[k] (Nm), applied directly by
MuJoCo torque motors and held for one control interval Ts (zero-order hold).
At every update:

  1. measure x0 = [q, qd];
  2. linearize MuJoCo's dynamics over one interval around (x0, held torque):
         x[k+1] = A x[k] + B tau[k] + c        (controllers/prediction.py)
     A, B from mjd_transitionFD composed over the interval; c makes the model
     exact at the linearization point (gravity, Coriolis, damping included).
     The same (A, B, c) is used for every stage: a linear time-invariant
     approximation of the nonlinear arm dynamics. This is linear MPC, not
     nonlinear MPC.
  3. solve
       min  sum_k ||x[k] - r[k]||^2_Q  (terminal x w_N)
          + w_tau  sum_k ||(tau[k] - tau_g) / tau_max||^2         effort above gravity hold
          + w_dtau sum_k ||(tau[k] - tau[k-1]) / tau_max||^2       smoothness
          + slack penalties
       s.t. dynamics; tau_min <= tau[k] <= tau_max                 (hard: exact physical limit)
            q_min + margin <= q[k] <= q_max - margin              (soft)
            |qd[k]| <= v_max                                       (soft)
  4. apply tau[0] and hold it until the next update.

Because the actuator applies exactly the commanded torque (no servo, no
clip unless the limit is exceeded, which the hard bound prevents), the torque
constraint is enforced predictively and exactly at every instant, which the
position-servo MPC could not do. The main model error is the LTI
approximation over the horizon; it grows with horizon length and with Ts.

Numerics: the QP optimizes normalized torques v = tau / tau_max in [-1, 1].
With raw torques (Nm) the diagonal of P spanned ~10 orders of magnitude
(effort weight 1/87^2 next to position weight 5e5), OSQP's solution polishing
failed, and the at-rest plan drifted 2.7 mrad at eps 1e-5 (measured).

On solver failure the controller hands over to a torque-level fallback
(computed-torque control) until the next successful update.

v0.4 additions (both off by default, so the v0.3 controller is unchanged):

* velocity_margin: constraint tightening, |qd[k]| <= (1 - margin) v_max in the
  QP, so the one-interval prediction error and the motion between knots stay
  inside the real limit (see docs/research/v0.4-results.md for the sizing).
* observer (controllers/disturbance.py): offset-free MPC. The estimated input
  disturbance d_hat (Nm) enters the prediction as a known constant input over
  the horizon, x[k+1] = A x[k] + B (tau[k] + d_hat) + c, and the effort term is
  measured from the torque that holds the arm against gravity *and* d_hat.
  The observer updates every physics step from the measured motion and the
  torque actually applied. With disturbance_feedthrough, the held torque is
  also corrected between MPC updates by the change of d_hat since the update
  (tau = tau_mpc - (d_hat_now - d_hat_update)), a 500 Hz inner compensation.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from controllers.base import Controller
from controllers.disturbance import JointDisturbanceObserver
from controllers.mpc import MPCStats, MPCUpdate, _per_joint, plan_violations
from controllers.prediction import MuJoCoPredictionModel

ReferenceFn = Callable[[float], tuple[np.ndarray, np.ndarray]]
_ACCEPTED = {osqp.SolverStatus.OSQP_SOLVED, osqp.SolverStatus.OSQP_SOLVED_INACCURATE}


@dataclass
class TorqueMPCWeights:
    position: float | np.ndarray = 1e5      # per rad^2
    velocity: float | np.ndarray = 10.0     # per (rad/s)^2
    terminal: float = 5.0
    effort: float | np.ndarray = 1.0        # per (torque above gravity / torque limit)^2
    torque_rate: float | np.ndarray = 10.0  # per (torque change / torque limit)^2
    slack_quadratic: float = 1e3
    slack_linear: float = 1e6               # exact penalty; 1e5 let a 2.5 rad step plan 3.8x v_max


class TorqueMPCController(Controller):
    def __init__(
        self,
        model: MuJoCoPredictionModel,
        fallback: Controller,
        horizon: int = 25,
        weights: TorqueMPCWeights | None = None,
        velocity_limits: np.ndarray | None = None,
        joint_limits: np.ndarray | None = None,
        joint_limit_margin: float = 0.02,
        soft_constraints: bool = True,
        reference_preview: ReferenceFn | None = None,
        solver_settings: dict | None = None,
        velocity_margin: float = 0.0,
        observer: JointDisturbanceObserver | None = None,
        disturbance_feedthrough: bool = False,
    ):
        if model.actuation != "torque":
            raise ValueError("TorqueMPCController needs torque-motor actuators (Panda(actuation='torque'))")
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.model = model
        self.fallback = fallback
        self.N = horizon
        self.dof = n = model.dof
        self.nx, self.nu = 2 * n, n
        self.substeps = model.substeps
        self.control_period = model.dt
        self.w = weights or TorqueMPCWeights()
        self.soft = soft_constraints
        self.reference_preview = reference_preview
        self.torque_limits = model.force_limits
        self.tau_scale = np.maximum(np.abs(self.torque_limits).max(axis=1), 1e-9)
        jl = joint_limits if joint_limits is not None else np.column_stack([np.full(n, -np.inf), np.full(n, np.inf)])
        self.joint_limits = np.asarray(jl, dtype=float)
        self.q_bounds = np.column_stack([jl[:, 0] + joint_limit_margin, jl[:, 1] - joint_limit_margin])
        vmax = _per_joint(np.inf if velocity_limits is None else velocity_limits, n)
        self.v_max = vmax
        if not 0.0 <= velocity_margin < 1.0:
            raise ValueError("velocity_margin must be in [0, 1)")
        self.velocity_margin = velocity_margin
        self.v_bounds = np.column_stack([-vmax, vmax]) * (1.0 - velocity_margin)
        self.observer = observer
        self.disturbance_feedthrough = disturbance_feedthrough
        self.settings = {
            "verbose": False,
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 500,
            "polishing": True,
            "warm_starting": True,
        } | (solver_settings or {})
        self._build_structure()
        self.reset()

    # z = [x[1..N] (N*nx) | tau[0..N-1] (N*nu) | s[1..N] (N*nx, soft only)]
    def _ix(self, k):
        return slice((k - 1) * self.nx, k * self.nx)

    def _iu(self, k):
        o = self.N * self.nx
        return slice(o + k * self.nu, o + (k + 1) * self.nu)

    def _build_structure(self) -> None:
        N, nx, nu, n = self.N, self.nx, self.nu, self.dof
        nX, nU = N * nx, N * nu
        nS = nX if self.soft else 0
        self.nz = nX + nU + nS

        # Cost: J = z'Hz + 2h'z, H constant (P = 2H). Diagonal except the torque-rate band.
        qx = np.concatenate([_per_joint(self.w.position, n), _per_joint(self.w.velocity, n)])
        self._Qx = qx
        diag = np.zeros(self.nz)
        for k in range(1, N + 1):
            diag[self._ix(k)] = qx * (self.w.terminal if k == N else 1.0)
        # Decision variables are v = tau / tau_scale, so these weights are dimensionless.
        self._we = _per_joint(self.w.effort, n)
        self._wr = _per_joint(self.w.torque_rate, n)
        U = np.zeros(nU)
        for k in range(N):
            U[k * nu:(k + 1) * nu] = self._we + self._wr * (2 if k < N - 1 else 1)
        diag[nX:nX + nU] = U
        if self.soft:
            diag[nX + nU:] = self.w.slack_quadratic
        off = np.zeros(nU - nu)
        for k in range(1, N):
            off[(k - 1) * nu:k * nu] = -self._wr
        H = sp.diags(diag) + sp.diags(np.r_[np.zeros(nX), off, np.zeros(nS + nu)][: self.nz - nu], nu) \
            + sp.diags(np.r_[np.zeros(nX), off, np.zeros(nS + nu)][: self.nz - nu], -nu)
        self._P = sp.triu(2 * H, format="csc")

        # Constraints, assembled in COO order with dense A/B blocks (fixed pattern).
        rows, cols, vals = [], [], []
        a_slots, b_slots = [], []  # (offset in COO value vector) per stage

        def add(r, c, v):
            rows.append(np.asarray(r)), cols.append(np.asarray(c)), vals.append(np.asarray(v, float))

        def dense(r0, c0, h, w):
            rr, cc = np.meshgrid(np.arange(r0, r0 + h), np.arange(c0, c0 + w), indexing="ij")
            add(rr.ravel(), cc.ravel(), np.zeros(h * w))

        r = 0
        for k in range(N):  # x[k+1] - A x[k] - B tau[k] = c   (A x0 moved to the bound for k = 0)
            add(np.arange(r, r + nx), np.arange(self._ix(k + 1).start, self._ix(k + 1).stop), np.ones(nx))
            if k >= 1:
                a_slots.append(len(vals))
                dense(r, self._ix(k).start, nx, nx)
            b_slots.append(len(vals))
            dense(r, self._iu(k).start, nx, nu)
            r += nx
        self._row_u = r  # torque limits (hard)
        add(np.arange(r, r + nU), np.arange(nX, nX + nU), np.ones(nU))
        r += nU
        self._row_x = r  # state limits
        if self.soft:
            add(np.arange(r, r + nX), np.arange(nX), np.ones(nX))          # x + s >= lower
            add(np.arange(r, r + nX), np.arange(nX + nU, self.nz), np.ones(nX))
            add(np.arange(r + nX, r + 2 * nX), np.arange(nX), np.ones(nX))  # x - s <= upper
            add(np.arange(r + nX, r + 2 * nX), np.arange(nX + nU, self.nz), -np.ones(nX))
            add(np.arange(r + 2 * nX, r + 3 * nX), np.arange(nX + nU, self.nz), np.ones(nX))  # s >= 0
            r += 3 * nX
        else:
            add(np.arange(r, r + nX), np.arange(nX), np.ones(nX))
            r += nX
        self.nc = r

        lengths = np.array([len(v) for v in vals])
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        rows, cols, base_vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        order = sp.coo_matrix((np.arange(1, len(rows) + 1, dtype=float), (rows, cols)), shape=(self.nc, self.nz)).tocsc()
        if order.nnz != len(rows):
            raise RuntimeError("duplicate entries in torque MPC constraint pattern")
        perm = order.data.astype(int) - 1          # CSC position -> COO position
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))           # COO position -> CSC position
        self._Ax = base_vals[perm]
        # Where each stage's A and B entries live in the CSC data array (row-major blocks).
        self._a_idx = np.concatenate([inv[starts[s]:starts[s] + nx * nx] for s in a_slots]) if a_slots else np.zeros(0, int)
        self._b_idx = np.concatenate([inv[starts[s]:starts[s] + nx * nu] for s in b_slots])
        self._csc_indices, self._csc_indptr = order.indices.copy(), order.indptr.copy()

        # Constant bounds.
        self._lo, self._hi = np.empty(self.nc), np.empty(self.nc)
        self._lo[self._row_u:self._row_u + nU] = np.tile(self.torque_limits[:, 0] / self.tau_scale, N)
        self._hi[self._row_u:self._row_u + nU] = np.tile(self.torque_limits[:, 1] / self.tau_scale, N)
        xlb = np.tile(np.r_[self.q_bounds[:, 0], self.v_bounds[:, 0]], N)
        xub = np.tile(np.r_[self.q_bounds[:, 1], self.v_bounds[:, 1]], N)
        r = self._row_x
        if self.soft:
            self._lo[r:r + nX], self._hi[r:r + nX] = xlb, np.inf
            self._lo[r + nX:r + 2 * nX], self._hi[r + nX:r + 2 * nX] = -np.inf, xub
            self._lo[r + 2 * nX:r + 3 * nX], self._hi[r + 2 * nX:r + 3 * nX] = 0.0, np.inf
        else:
            self._lo[r:r + nX], self._hi[r:r + nX] = xlb, xub
        self._h_slack = 0.5 * self.w.slack_linear if self.soft else 0.0

    # ------------------------------------------------------------- interface

    def reset(self) -> None:
        self.fallback.reset()
        if self.observer is not None:
            self.observer.reset()
        self._d_update = np.zeros(self.dof)  # d_hat used by the last successful update
        self._step = 0
        self._t = 0.0
        self._command: np.ndarray | None = None
        self._applied: np.ndarray | None = None
        self._using_fallback = False
        self._solver: osqp.OSQP | None = None
        self._z_prev: np.ndarray | None = None
        self.last_prediction: dict | None = None
        self.stats = MPCStats()

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def disturbance_estimate(self) -> np.ndarray:
        return self.observer.estimate.copy() if self.observer is not None else np.zeros(self.dof)

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        t = self._t if t is None else t
        if self.observer is not None:
            self.observer.update(q, qd, dt)
        if self._step % self.substeps == 0:
            self._update(np.asarray(q, float), np.asarray(qd, float), q_ref, qd_ref, t)
        self._step += 1
        self._t = t + dt
        if self._using_fallback or self._command is None:
            out = self.fallback.compute(q, qd, q_ref, qd_ref, dt, t=t, qdd_ref=qdd_ref)
        else:
            out = self._command.copy()
            if self.observer is not None and self.disturbance_feedthrough:
                out = np.clip(out - (self.observer.estimate - self._d_update),
                              self.torque_limits[:, 0], self.torque_limits[:, 1])
        self._applied = np.array(out, dtype=float)
        if self.observer is not None:
            self.observer.applied(self._applied)
        return out

    def _references(self, t, q_ref, qd_ref) -> np.ndarray:
        Ts, n = self.control_period, self.dof
        refs = np.empty((self.N, self.nx))
        for k in range(1, self.N + 1):
            if self.reference_preview is not None:
                rq, rqd = self.reference_preview(t + k * Ts)
            else:
                rq, rqd = np.asarray(q_ref) + np.asarray(qd_ref) * k * Ts, np.asarray(qd_ref)
            refs[k - 1, :n], refs[k - 1, n:] = rq, rqd
        return refs

    def _update(self, q, qd, q_ref, qd_ref, t) -> None:
        start = time.perf_counter()
        status, iterations, solve_time, max_slack = "not run", 0, 0.0, 0.0
        prim_res = dual_res = float("nan")
        v_ratio, q_excess = 0.0, 0.0
        N, nx, nu = self.N, self.nx, self.nu
        nX, nU = N * nx, N * nu
        try:
            x0 = np.concatenate([q, qd])
            if not np.all(np.isfinite(x0)):
                raise FloatingPointError("non-finite measured state")
            tau_g = self.model.gravity_torque(q)
            tau_prev = self._applied if self._applied is not None else tau_g
            self._tau_prev_at_update = tau_prev
            lin = self.model.linearize(q, qd, tau_prev)
            if not lin.plausible:
                raise FloatingPointError("implausible linearization")
            # Offset-free MPC: the estimated input disturbance is a known input over the horizon.
            d_hat = self.disturbance_estimate
            c = lin.c + lin.B @ d_hat
            tau_hold = tau_g - d_hat  # torque that holds the arm still against gravity and d_hat
            refs = self._references(t, q_ref, qd_ref)

            # Constraint values: only the A/B blocks change.
            Ax = self._Ax
            if len(self._a_idx):
                Ax[self._a_idx] = np.tile(-lin.A.ravel(), N - 1)
            Ax[self._b_idx] = np.tile(-(lin.B * self.tau_scale).ravel(), N)  # B acting on v
            lo, hi = self._lo, self._hi
            lo[:nx] = hi[:nx] = lin.A @ x0 + c
            lo[nx:nX] = hi[nx:nX] = np.tile(c, N - 1)
            # Linear cost.
            h = np.zeros(self.nz)
            w = np.ones(N)
            w[-1] = self.w.terminal
            h[:nX] = -(w[:, None] * self._Qx[None, :] * refs).ravel()
            h[nX:nX + nU] = -np.tile(self._we * tau_hold / self.tau_scale, N)
            h[nX:nX + nu] -= self._wr * tau_prev / self.tau_scale
            h[nX + nU:] = self._h_slack
            if not (np.all(np.isfinite(Ax)) and np.all(np.isfinite(h)) and np.all(lo <= hi)):
                raise FloatingPointError("non-finite or inconsistent QP data")

            t_solve = time.perf_counter()
            if self._solver is None:
                self._solver = osqp.OSQP()
                A = sp.csc_matrix((Ax.copy(), self._csc_indices, self._csc_indptr), shape=(self.nc, self.nz))
                self._solver.setup(self._P, 2 * h, A, lo.copy(), hi.copy(), **self.settings)
            else:
                self._solver.update(q=2 * h, l=lo, u=hi, Ax=Ax)
                if self._z_prev is not None:
                    self._solver.warm_start(x=self._shifted(self._z_prev))
            res = self._solver.solve(raise_error=False)
            solve_time = time.perf_counter() - t_solve
            status, iterations = res.info.status, int(res.info.iter)
            prim_res, dual_res = float(res.info.prim_res), float(res.info.dual_res)
            z = res.x
            if res.info.status_val not in _ACCEPTED or z is None or not np.all(np.isfinite(z)):
                raise RuntimeError(f"OSQP status: {status}")
            taus = z[nX:nX + nU].reshape(N, nu) * self.tau_scale
            self._command = np.clip(taus[0], self.torque_limits[:, 0], self.torque_limits[:, 1])
            self._z_prev = z
            self._using_fallback = False
            self._d_update = d_hat
            if self.soft:
                max_slack = float(z[nX + nU:].max())
            v_ratio, q_excess = plan_violations(z[:nX].reshape(N, nx), self.v_max, self.joint_limits)
            self.last_prediction = {
                "time": t, "x0": x0, "states": z[:nX].reshape(N, nx), "commands": taus,
                "references": refs, "model": lin, "gravity": tau_g, "disturbance": d_hat,
            }
        except Exception as exc:  # noqa: BLE001 - any failure hands over to the fallback
            if status == "not run":
                status = f"error: {exc}"
            self._using_fallback = True
            self._z_prev = None
        self.stats.updates.append(MPCUpdate(
            time=t, status=status, iterations=iterations, solve_time=solve_time,
            update_time=time.perf_counter() - start, fallback=self._using_fallback, max_slack=max_slack,
            predicted_velocity_ratio=v_ratio, predicted_position_excess=q_excess,
            prim_res=prim_res, dual_res=dual_res))

    def _shifted(self, z: np.ndarray) -> np.ndarray:
        N, nx, nu = self.N, self.nx, self.nu
        zs = z.copy()
        blocks = [(0, nx), (N * nx, nu)] + ([(N * (nx + nu), nx)] if self.soft else [])
        for start, width in blocks:
            b = z[start:start + N * width].reshape(N, width)
            zs[start:start + N * width] = np.vstack([b[1:], b[-1:]]).ravel()
        return zs

