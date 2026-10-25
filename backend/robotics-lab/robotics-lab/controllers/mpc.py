"""Receding-horizon joint-space MPC for position-servo arms (OSQP).

At every control update (default 50 Hz) the controller:
  1. takes the latest *measured* joint state x0 = [q, qd],
  2. relinearizes MuJoCo's dynamics over one control interval around x0 and
     the command currently held (controllers/prediction.py):
       x[k+1] = A x[k] + B u[k] + c
  3. updates and solves the finite-horizon QP below with OSQP (warm-started
     from the previous solution, shifted by one interval),
  4. applies u[0] and holds it until the next update.

The decision variables u are position-servo setpoints (rad), not torques.

    minimize   sum_{k=1..N}   ||q[k] - q_ref[k]||^2_Wq + ||qd[k] - qd_ref[k]||^2_Wv   (x terminal at k = N)
             + sum_{k=0..N-1} w_tau * ||(tau[k] - tau_g) / tau_max||^2               effort
             + sum_{k=0..N-1} ||u[k] - u[k-1]||^2_Wdu                               smoothness
             + rho_2 * ||s||^2 + rho_1 * sum(s)                                     slack (soft mode)
    subject to x[k+1] = A x[k] + B u[k] + c + d_hat                      prediction model
               u_min <= u[k] <= u_max                                    command limits (ctrlrange), hard
               |u[k] - u[k-1]| <= r * v_max * Ts                         setpoint slew limit, hard
               -f <= tau[k] / tau_max <= f                               sustained torque limits
               q_min + margin <= q[k+1] <= q_max - margin                joint position limits
               -v_max <= qd[k+1] <= v_max                                joint velocity limits

In soft mode (default) the torque, position and velocity limits get
non-negative slack variables with a heavy penalty, so the QP is feasible for
any measured state; they are hard otherwise. Command and slew limits are
always hard (holding the previous command always satisfies them).

The slew limit stops the setpoint from moving faster than the joint may move
(r = command_slew_scale, default 1). Without it, a step in the reference makes
the optimizer jump the setpoint far ahead to reach full speed within one
interval; every servo then saturates for the whole interval, the unsaturated
model is badly wrong, and the next solves start from states the plan never
predicted. The limit also bounds each command-step torque spike to
kp * v_max * Ts.

tau[k] = kp (u[k] - q[k+1]) - kd qd[k+1] is the *sustained* servo torque: the
torque at the end of interval k, after the joint has responded to the held
setpoint u[k]. tau_g is the gravity torque at the measured configuration, so
the effort term penalizes torque beyond what holding the arm still requires.

Why not the torque at the start of the interval? The servos are stiff and
heavily damped relative to link inertia (joint 7: I/kd ~ 0.5 ms), so each
setpoint step du produces a ~1 ms torque spike of kp * du that MuJoCo clips to
the force range. Bounding that spike would cap the change of setpoint per
interval at tau_max / kp (0.0057 rad on the 12 Nm wrist joints, i.e. 0.29
rad/s at 50 Hz). The spikes are left to MuJoCo's physical force clip; the
benchmark counts them (torque saturation). The prediction model is linearized
where the servo has settled (the held command), so it does not see the spikes;
that mismatch is the main reason MPC tracks less tightly than PID here (see
README "MPC design notes" for the measured alternatives).

d_hat (optional, disturbance_gain > 0) is a filtered estimate of the
one-interval prediction error, for offset-free tracking under model mismatch.

When an update fails (solver status other than solved, non-finite data or an
exception) the controller hands over to a JointPIDController until the next
successful update. The fallback PID runs in the background every physics
step so its integrator is warm when it takes over.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp

from controllers.base import Controller
from controllers.prediction import AffineModel, MuJoCoPredictionModel

ReferenceFn = Callable[[float], tuple[np.ndarray, np.ndarray]]

_ACCEPTED = {osqp.SolverStatus.OSQP_SOLVED, osqp.SolverStatus.OSQP_SOLVED_INACCURATE}


@dataclass
class MPCWeights:
    position: float | np.ndarray = 1e4      # per rad^2
    velocity: float | np.ndarray = 10.0     # per (rad/s)^2
    terminal: float = 5.0                   # multiplier on the last stage's tracking weights
    effort: float | np.ndarray = 1.0        # per (torque / torque limit)^2
    command_rate: float | np.ndarray = 1e3  # per rad^2 change of setpoint between intervals
    slack_quadratic: float = 1e3            # soft constraints: per (violation)^2
    slack_linear: float = 1e5               # soft constraints: per unit violation (exact penalty)


@dataclass
class MPCUpdate:
    """Record of one MPC update."""

    time: float
    status: str
    iterations: int
    solve_time: float     # OSQP update + solve wall time (s)
    update_time: float    # linearization + QP construction + solve wall time (s)
    fallback: bool
    max_slack: float = 0.0
    # What the accepted plan itself predicts, against the nominal (untightened) limits:
    predicted_velocity_ratio: float = 0.0   # max over knots and joints of |qd_pred| / v_max
    predicted_position_excess: float = 0.0  # max over knots of distance beyond a joint limit (rad), 0 if inside
    prim_res: float = float("nan")          # OSQP primal / dual residual of the returned solution (v0.6)
    dual_res: float = float("nan")


def plan_violations(states: np.ndarray, v_max: np.ndarray, joint_limits: np.ndarray | None) -> tuple[float, float]:
    """(max |qd| / v_max, max position excess beyond the joint limits) over predicted states (N, 2 * dof)."""
    n = states.shape[1] // 2
    ratio = float(np.max(np.abs(states[:, n:]) / v_max)) if np.all(np.isfinite(v_max)) else 0.0
    excess = 0.0
    if joint_limits is not None:
        q = states[:, :n]
        excess = float(max(np.max(joint_limits[:, 0] - q), np.max(q - joint_limits[:, 1]), 0.0))
    return ratio, excess


@dataclass
class MPCStats:
    updates: list[MPCUpdate] = field(default_factory=list)

    def summary(self, control_period: float) -> dict:
        if not self.updates:
            return {"updates": 0}
        solve = np.array([u.solve_time for u in self.updates])
        total = np.array([u.update_time for u in self.updates])
        return {
            "updates": len(self.updates),
            "fallbacks": int(sum(u.fallback for u in self.updates)),
            "inaccurate": int(sum(u.status == "solved inaccurate" for u in self.updates)),
            "solve_time_mean_ms": 1e3 * float(solve.mean()),
            "solve_time_p95_ms": 1e3 * float(np.percentile(solve, 95)),
            "solve_time_p99_ms": 1e3 * float(np.percentile(solve, 99)),
            "update_time_mean_ms": 1e3 * float(total.mean()),
            "update_time_p95_ms": 1e3 * float(np.percentile(total, 95)),
            "update_time_p99_ms": 1e3 * float(np.percentile(total, 99)),
            "update_time_max_ms": 1e3 * float(total.max()),
            "deadline_overruns": int(np.sum(total > control_period)),
            "mean_iterations": float(np.mean([u.iterations for u in self.updates])),
            "max_slack": float(max(u.max_slack for u in self.updates)),
            "max_predicted_velocity_ratio": float(max(u.predicted_velocity_ratio for u in self.updates)),
            "max_predicted_position_excess": float(max(u.predicted_position_excess for u in self.updates)),
        }


def _per_joint(value, dof: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), (dof,)).copy()


class JointMPCController(Controller):
    def __init__(
        self,
        model: MuJoCoPredictionModel,
        fallback: Controller,
        horizon: int = 15,
        weights: MPCWeights | None = None,
        velocity_limits: np.ndarray | None = None,
        joint_limits: np.ndarray | None = None,
        joint_limit_margin: float = 0.02,
        torque_limit_fraction: float = 0.95,
        soft_constraints: bool = True,
        command_slew_scale: float | None = 1.0,
        disturbance_gain: float = 0.0,
        velocity_margin: float = 0.0,
        reference_preview: ReferenceFn | None = None,
        solver_settings: dict | None = None,
    ):
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        if not 0.0 <= disturbance_gain <= 1.0:
            raise ValueError("disturbance_gain must be in [0, 1]")
        self.model = model
        self.fallback = fallback
        self.N = horizon
        self.dof = n = model.dof
        self.nx, self.nu = 2 * n, n
        self.substeps = model.substeps
        self.control_period = model.dt
        self.w = weights or MPCWeights()
        self.soft = soft_constraints
        self.disturbance_gain = disturbance_gain
        self.reference_preview = reference_preview

        servo = model.servo
        self.kp, self.kd = servo.kp, servo.kd
        self.ctrl_limits = servo.ctrl_limits
        self.force_limits = servo.force_limits
        self.torque_limit_fraction = torque_limit_fraction
        jl = joint_limits if joint_limits is not None else servo.ctrl_limits
        self.joint_limits = np.asarray(jl, dtype=float)
        self.q_bounds = np.column_stack([jl[:, 0] + joint_limit_margin, jl[:, 1] - joint_limit_margin])
        vmax = _per_joint(np.inf if velocity_limits is None else velocity_limits, n)
        self.v_max = vmax
        # Constraint tightening (v0.4): the QP keeps |qd| <= (1 - margin) v_max so that
        # prediction error and intersample motion stay inside the real limit.
        self.v_bounds = np.column_stack([-vmax, vmax]) * (1.0 - velocity_margin)
        self.slew = (
            command_slew_scale * vmax * self.control_period if command_slew_scale is not None else np.full(n, np.inf)
        )

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

    # ---------------------------------------------------------------- layout
    # z = [x[1..N] (N*nx) | u[0..N-1] (N*nu) | s[1..N] (N*ny, soft mode only)]
    # Per stage k = 1..N the constrained outputs are y[k] = [x[k]; tau[k-1] / tau_max].

    def _ix(self, k: int) -> slice:  # state x[k], k = 1..N
        return slice((k - 1) * self.nx, k * self.nx)

    def _iu(self, k: int) -> slice:  # input u[k], k = 0..N-1
        o = self.N * self.nx
        return slice(o + k * self.nu, o + (k + 1) * self.nu)

    def _is(self, k: int) -> slice:  # slack s[k], k = 1..N
        o = self.N * (self.nx + self.nu)
        return slice(o + (k - 1) * self.ny, o + k * self.ny)

    def _build_structure(self) -> None:
        """Fixed sparsity pattern for P and the constraint matrix (OSQP needs it constant)."""
        N, nx, nu, n = self.N, self.nx, self.nu, self.dof
        self.ny = nx + n
        self.nz = N * (nx + nu) + (N * self.ny if self.soft else 0)
        rho = self.kd / self.kp
        tau_max = np.maximum(np.abs(self.force_limits).max(axis=1), 1e-9)
        tau_max = np.where(np.isfinite(tau_max), tau_max, 1.0)
        self._tau_scale = self.kp / tau_max  # tau / tau_max = tau_scale * (u - q - rho qd)

        # ---- cost Hessian H (J = z'Hz + 2h'z, OSQP's P = 2H); constant across updates
        H = sp.lil_matrix((self.nz, self.nz))
        self._Qx = np.concatenate([_per_joint(self.w.position, n), _per_joint(self.w.velocity, n)])
        for k in range(1, N + 1):
            scale = self.w.terminal if k == N else 1.0
            H[self._ix(k), self._ix(k)] += sp.diags(scale * self._Qx)
        # Effort on the sustained torque of interval k (u[k] held, state x[k+1]).
        # Per-joint (diagonal) blocks: torque i depends only on u_i, q_i, qd_i.
        self._te2 = _per_joint(self.w.effort, n) * self._tau_scale**2
        te2 = self._te2
        for k in range(N):
            iu, xs = self._iu(k), self._ix(k + 1).start
            iq, iv = slice(xs, xs + n), slice(xs + n, xs + nx)
            H[iu, iu] += sp.diags(te2)
            for ix, coef in ((iq, -te2), (iv, -te2 * rho)):
                H[iu, ix] += sp.diags(coef)
                H[ix, iu] += sp.diags(coef)
            H[iq, iq] += sp.diags(te2)
            H[iv, iv] += sp.diags(te2 * rho**2)
            H[iq, iv] += sp.diags(te2 * rho)
            H[iv, iq] += sp.diags(te2 * rho)
        # Command-rate penalty: sum_k ||u[k] - u[k-1]||^2_Wdu, with u[-1] the held command.
        self._wdu = _per_joint(self.w.command_rate, n)
        wdu = sp.diags(self._wdu)
        for k in range(N):
            H[self._iu(k), self._iu(k)] += wdu * (2 if k < N - 1 else 1)
            if k >= 1:
                H[self._iu(k), self._iu(k - 1)] -= wdu
                H[self._iu(k - 1), self._iu(k)] -= wdu
        if self.soft:
            for k in range(1, N + 1):
                H[self._is(k), self._is(k)] = sp.eye(self.ny) * self.w.slack_quadratic
        self._P = sp.triu(2 * H.tocsc(), format="csc")

        # ---- constraint matrix, assembled in COO order; A/B blocks are dense so the
        # pattern does not depend on which Jacobian entries happen to be zero.
        rows, cols, vals = [], [], []
        self._dyn_slots = []  # ("A" | "B", index into vals)
        ar = np.arange

        def add(rr, cc, v):
            rows.append(np.asarray(rr)), cols.append(np.asarray(cc)), vals.append(np.asarray(v, float))

        def add_block(r0, c0, shape):
            rr, cc = np.meshgrid(ar(r0, r0 + shape[0]), ar(c0, c0 + shape[1]), indexing="ij")
            add(rr.ravel(), cc.ravel(), np.zeros(shape[0] * shape[1]))

        r = 0
        # 1. dynamics: x[k+1] - A x[k] - B u[k] = c (+ A x0 for k = 0)
        self._row_dyn = r
        for k in range(N):
            add(ar(r, r + nx), ar(self._ix(k + 1).start, self._ix(k + 1).stop), np.ones(nx))
            if k >= 1:
                self._dyn_slots.append(("A", len(vals)))
                add_block(r, self._ix(k).start, (nx, nx))
            self._dyn_slots.append(("B", len(vals)))
            add_block(r, self._iu(k).start, (nx, nu))
            r += nx
        # 2. command limits (hard)
        self._row_u = r
        add(ar(r, r + N * nu), ar(self._iu(0).start, self._iu(N - 1).stop), np.ones(N * nu))
        r += N * nu
        # 2b. setpoint slew limits (hard): u[0] - u_prev, u[k] - u[k-1]
        self._row_slew = r
        add(ar(r, r + N * nu), ar(self._iu(0).start, self._iu(N - 1).stop), np.ones(N * nu))
        add(ar(r + nu, r + N * nu), ar(self._iu(0).start, self._iu(N - 2).stop), -np.ones((N - 1) * nu))
        r += N * nu

        # 3. constrained outputs y[k] = [x[k]; tau[k-1]/tau_max], k = 1..N.
        def add_outputs(r0, slack_sign):
            for k in range(1, N + 1):
                rk = r0 + (k - 1) * self.ny
                xs = self._ix(k).start
                add(ar(rk, rk + nx), ar(xs, xs + nx), np.ones(nx))
                rt = ar(rk + nx, rk + self.ny)
                add(rt, ar(self._iu(k - 1).start, self._iu(k - 1).stop), self._tau_scale)
                add(rt, ar(xs, xs + n), -self._tau_scale)
                add(rt, ar(xs + n, xs + nx), -self._tau_scale * rho)
                if slack_sign:
                    add(ar(rk, rk + self.ny), ar(self._is(k).start, self._is(k).stop), slack_sign * np.ones(self.ny))

        self._row_y = r
        if self.soft:
            add_outputs(r, +1.0)        # y + s >= lower
            r += N * self.ny
            add_outputs(r, -1.0)        # y - s <= upper
            r += N * self.ny
            self._row_s = r             # s >= 0
            add(ar(r, r + N * self.ny), ar(self._is(1).start, self._is(N).stop), np.ones(N * self.ny))
            r += N * self.ny
        else:
            add_outputs(r, 0.0)
            r += N * self.ny
        self.nc = r

        self._const_vals = vals
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        nnz = len(rows)
        order = sp.coo_matrix((np.arange(1, nnz + 1, dtype=float), (rows, cols)), shape=(self.nc, self.nz)).tocsc()
        if order.nnz != nnz:
            raise RuntimeError("duplicate entries in MPC constraint pattern")
        self._perm = order.data.astype(int) - 1
        self._csc_indices, self._csc_indptr = order.indices.copy(), order.indptr.copy()
        # Fixed CSC data array; each update overwrites only the A and B slots.
        inv = np.empty_like(self._perm)
        inv[self._perm] = np.arange(len(self._perm))
        starts = np.r_[0, np.cumsum([len(v) for v in vals])[:-1]]
        self._Ax = np.concatenate(vals)[self._perm]
        a = [inv[starts[slot]:starts[slot] + nx * nx] for kind, slot in self._dyn_slots if kind == "A"]
        self._a_idx = np.concatenate(a) if a else np.zeros(0, dtype=int)
        self._b_idx = np.concatenate([inv[starts[slot]:starts[slot] + nx * nu]
                                      for kind, slot in self._dyn_slots if kind == "B"])

        # Output bounds (constant): positions, velocities, torque fractions.
        f = self.torque_limit_fraction
        y_lo = np.concatenate([self.q_bounds[:, 0], self.v_bounds[:, 0], f * self.force_limits[:, 0] / tau_max])
        y_hi = np.concatenate([self.q_bounds[:, 1], self.v_bounds[:, 1], f * self.force_limits[:, 1] / tau_max])
        self._y_lo, self._y_hi = np.tile(y_lo, N), np.tile(y_hi, N)

    def _constraint_values(self, lin: AffineModel) -> np.ndarray:
        if len(self._a_idx):
            self._Ax[self._a_idx] = np.tile(-lin.A.ravel(), self.N - 1)
        self._Ax[self._b_idx] = np.tile(-lin.B.ravel(), self.N)
        return self._Ax

    # ------------------------------------------------------------- interface

    def reset(self) -> None:
        self.fallback.reset()
        self._step = 0
        self._t = 0.0
        self._command: np.ndarray | None = None  # last MPC solution's first command
        self._applied: np.ndarray | None = None  # command actually sent last (MPC or fallback)
        self._using_fallback = False
        self._solver: osqp.OSQP | None = None
        self._z_prev: np.ndarray | None = None
        self._d_hat = np.zeros(self.nx)
        self._last_lin: AffineModel | None = None
        self._last_x: np.ndarray | None = None
        self.last_prediction: dict | None = None
        self.stats = MPCStats()

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        t = self._t if t is None else t
        # The fallback runs every physics step so it is warm when it takes over.
        fallback_ctrl = self.fallback.compute(q, qd, q_ref, qd_ref, dt, t=t)
        if self._step % self.substeps == 0:
            self._update(np.asarray(q, float), np.asarray(qd, float), q_ref, qd_ref, t)
        self._step += 1
        self._t = t + dt
        out = fallback_ctrl if (self._using_fallback or self._command is None) else self._command.copy()
        self._applied = np.array(out, dtype=float)
        return out

    # ----------------------------------------------------------------- update

    def _references(self, t: float, q_ref, qd_ref) -> np.ndarray:
        """Reference states r[1..N] at t + k Ts."""
        Ts = self.control_period
        refs = np.empty((self.N, self.nx))
        for k in range(1, self.N + 1):
            if self.reference_preview is not None:
                rq, rqd = self.reference_preview(t + k * Ts)
            else:  # no preview: constant-velocity extrapolation of the current reference
                rq, rqd = np.asarray(q_ref) + np.asarray(qd_ref) * k * Ts, np.asarray(qd_ref)
            refs[k - 1, : self.dof], refs[k - 1, self.dof :] = rq, rqd
        return refs

    def _update(self, q, qd, q_ref, qd_ref, t) -> None:
        start = time.perf_counter()
        x0 = np.concatenate([q, qd])
        status, iterations, solve_time, max_slack = "not run", 0, 0.0, 0.0
        v_ratio, q_excess = 0.0, 0.0
        try:
            if not np.all(np.isfinite(x0)):
                raise FloatingPointError("non-finite measured state")
            tau_g = self.model.gravity_torque(q)
            # The command the servo is holding right now: the last one actually sent,
            # which after a fallback is the PID's, not the stale MPC solution (v0.2
            # used the stale one, anchoring the slew limit to a setpoint the servo no
            # longer had). Before the first command, assume the servo has settled at
            # the current motion: gravity offset plus the damping lead kd/kp * qd.
            u_prev = self._applied if self._applied is not None else q + tau_g / self.kp + self.kd / self.kp * qd
            # Offset-free correction: filter the one-interval prediction error.
            if self.disturbance_gain > 0 and self._last_lin is not None and not self._using_fallback:
                pred = self._last_lin.A @ self._last_x + self._last_lin.B @ u_prev + self._last_lin.c + self._d_hat
                self._d_hat += self.disturbance_gain * (x0 - pred)
            lin = self.model.linearize(q, qd, u_prev)
            if not self._plausible(lin):
                raise FloatingPointError("implausible linearization (non-finite or exploding A/B)")
            refs = self._references(t, q_ref, qd_ref)
            Ax, h, lo, hi = self._qp_data(lin, x0, u_prev, tau_g, refs)
            # OSQP rejects inconsistent data in update() but still returns the old
            # solution as "solved", so validate before handing it over.
            if not (np.all(np.isfinite(Ax)) and np.all(np.isfinite(h)) and np.all(lo <= hi)):
                raise FloatingPointError("non-finite or inconsistent QP data")

            t_solve = time.perf_counter()
            if self._solver is None:
                self._solver = osqp.OSQP()
                A = sp.csc_matrix((Ax.copy(), self._csc_indices, self._csc_indptr), shape=(self.nc, self.nz))
                self._solver.setup(self._P, 2 * h, A, lo, hi, **self.settings)
            else:
                self._solver.update(q=2 * h, l=lo, u=hi, Ax=Ax)
                if self._z_prev is not None:
                    self._solver.warm_start(x=self._shifted(self._z_prev))
            res = self._solver.solve()
            solve_time = time.perf_counter() - t_solve
            status, iterations = res.info.status, int(res.info.iter)
            z = res.x
            if res.info.status_val not in _ACCEPTED or z is None or not np.all(np.isfinite(z)):
                raise RuntimeError(f"OSQP status: {status}")

            self._command = np.clip(z[self._iu(0)], self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
            self._z_prev = z
            self._using_fallback = False
            self._last_lin, self._last_x = lin, x0
            if self.soft:
                max_slack = float(np.max(z[self._is(1).start :]))
            v_ratio, q_excess = plan_violations(z[: self.N * self.nx].reshape(self.N, self.nx), self.v_max, self.joint_limits)
            self.last_prediction = {
                "time": t,
                "x0": x0,
                "states": z[: self.N * self.nx].reshape(self.N, self.nx),
                "commands": z[self._iu(0).start : self._iu(self.N - 1).stop].reshape(self.N, self.nu),
                "references": refs,
                "model": lin,
                "disturbance": self._d_hat.copy(),
            }
        except Exception as exc:  # noqa: BLE001 - any failure must hand over to the fallback
            if status == "not run":
                status = f"error: {exc}"
            self._using_fallback = True
            self._z_prev = None
            self._last_lin = None
        self.stats.updates.append(
            MPCUpdate(
                time=t,
                status=status,
                iterations=iterations,
                solve_time=solve_time,
                update_time=time.perf_counter() - start,
                fallback=self._using_fallback,
                max_slack=max_slack,
                predicted_velocity_ratio=v_ratio,
                predicted_position_excess=q_excess,
            )
        )

    @staticmethod
    def _plausible(lin: AffineModel) -> bool:
        """Reject linearizations no physical arm interval could produce (checked once, in the model)."""
        return lin.plausible

    def _qp_data(self, lin: AffineModel, x0, u_prev, tau_g, refs):
        N, nx, n = self.N, self.nx, self.dof
        rho = self.kd / self.kp

        # Linear cost term h (J = z'Hz + 2h'z).
        h = np.zeros(self.nz)
        for k in range(1, N + 1):
            scale = self.w.terminal if k == N else 1.0
            h[self._ix(k)] -= scale * self._Qx * refs[k - 1]
        # Effort: penalize (u - q - rho qd - tau_g / kp), scaled to the torque limit.
        off = self._te2 * tau_g / self.kp
        for k in range(N):
            xs = self._ix(k + 1).start
            h[self._iu(k)] -= off
            h[xs : xs + n] += off
            h[xs + n : xs + nx] += off * rho
        h[self._iu(0)] -= self._wdu * u_prev
        if self.soft:
            h[self._is(1).start :] += 0.5 * self.w.slack_linear

        # Bounds.
        lo, hi = np.empty(self.nc), np.empty(self.nc)
        r = self._row_dyn
        lo[r : r + nx] = hi[r : r + nx] = lin.A @ x0 + lin.c + self._d_hat
        lo[r + nx : r + N * nx] = hi[r + nx : r + N * nx] = np.tile(lin.c + self._d_hat, N - 1)
        r = self._row_u
        lo[r : r + N * n] = np.tile(self.ctrl_limits[:, 0], N)
        hi[r : r + N * n] = np.tile(self.ctrl_limits[:, 1], N)
        r = self._row_slew
        lo[r : r + N * n], hi[r : r + N * n] = np.tile(-self.slew, N), np.tile(self.slew, N)
        lo[r : r + n] += u_prev
        hi[r : r + n] += u_prev
        r, m = self._row_y, N * self.ny
        if self.soft:
            lo[r : r + m], hi[r : r + m] = self._y_lo, np.inf
            lo[r + m : r + 2 * m], hi[r + m : r + 2 * m] = -np.inf, self._y_hi
            lo[r + 2 * m : r + 3 * m], hi[r + 2 * m : r + 3 * m] = 0.0, np.inf
        else:
            lo[r : r + m], hi[r : r + m] = self._y_lo, self._y_hi
        return self._constraint_values(lin), h, lo, hi

    def _shifted(self, z: np.ndarray) -> np.ndarray:
        """Previous solution advanced one interval (last stage repeated) for warm starting."""
        N, nx, nu = self.N, self.nx, self.nu
        zs = z.copy()
        for start, width in ((0, nx), (N * nx, nu)) + (((N * (nx + nu)), self.ny),) * self.soft:
            blk = z[start : start + N * width].reshape(N, width)
            zs[start : start + N * width] = np.vstack([blk[1:], blk[-1:]]).ravel()
        return zs
