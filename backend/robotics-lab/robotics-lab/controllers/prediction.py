"""MuJoCo-backed prediction model for joint-space MPC (position-servo or torque arms).

State and input
---------------
    x = [q, qd]   arm joint positions (rad) and velocities (rad/s), 2 * dof
    u             arm actuator commands: position-servo *setpoints* (rad), or
                  joint torques (Nm) for torque-motor actuators (v0.3).
                  For servos:
                  They are not torques. MuJoCo turns them into a joint torque
                      tau = clip(kp * (u - q) - kd * qd, -f_max, f_max)

The controller runs at a lower rate than physics: each command is held for
`substeps` physics steps (zero-order hold). The model predicts one control
interval Ts = substeps * dt:

    x[k+1] = A x[k] + B u[k] + c

A, B and c come from MuJoCo itself, linearized around the measured state x0
and the last applied command u0:

* "power" (default): one-step Jacobians from mjd_transitionFD (forward
  differences by default, centered optional, of mj_step, including the implicitfast integrator, armature,
  joint damping and the servo), composed over the interval:
  A = A1^n, B = (A1^(n-1) + ... + I) B1. Exact to ~1e-6 rad while no actuator
  saturates, but wrong when the first substep is force-clipped (B1 is then ~0
  for that joint).
* "rollout": forward finite differences of the true n-step MuJoCo map (21
  perturbed rollouts of n mj_step calls each). ~5x more expensive; valid
  through the servo's force clip.

In both cases c is chosen so the model is exact at the linearization point:
c = f_n(x0, u_lin) - A x0 - B u_lin, where f_n is a true n-step MuJoCo
rollout. That carries gravity, Coriolis and servo terms at the measured state.

Why the linearization command matters: the servos are stiff (kp up to 4500
Nm/rad) and heavily damped relative to link inertia, so stepping the held
setpoint by du causes a ~1 ms torque spike of kp * du that MuJoCo clips to the
force range (12 Nm on joints 5-7). A linearization taken *through* that clip
(e.g. "rollout" at the next planned command) is locally unstable under
gravity, because a clipped joint gets constant torque with no servo stiffness
or damping. The MPC therefore linearizes at the held command, where the servo
has settled.

Limitations (validated in tests/test_mpc.py):
* Linear in (x, u) around the current state; gravity and inertia changes along
  the horizon are first-order only.
* Force-range saturation is a non-smooth clip. A, B are local to the
  linearization point (true model; force-unlimited backup at the clip boundary,
  see jac_model). Away from that point, clipping makes the model approximate.
* Finger joints are frozen at their current state ("power" composes the arm
  block of the one-step Jacobian only); contacts are whatever is active at the
  linearization point. External forces are not known to the model.

All computation happens on a private MjData. The live simulation is only read.
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

# State copied from the live simulation into the scratch data: positions,
# velocities, mocap/equality state, actuator commands and the solver
# warm start. Deliberately excludes applied external forces (xfrc_applied,
# qfrc_applied): a controller cannot measure those.
_COPY_SPEC = (
    mujoco.mjtState.mjSTATE_FULLPHYSICS
    | mujoco.mjtState.mjSTATE_CTRL
    | mujoco.mjtState.mjSTATE_WARMSTART
)


@dataclass(frozen=True)
class ServoParameters:
    """Per-joint parameters of MuJoCo position-servo actuators."""

    kp: np.ndarray            # stiffness (Nm/rad)
    kd: np.ndarray            # damping (Nm s/rad)
    force_limits: np.ndarray  # (dof, 2) physical torque limits (Nm)
    ctrl_limits: np.ndarray   # (dof, 2) command (setpoint) limits (rad)

    def torque(self, u: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Unclipped servo torque for setpoint u at state (q, qd)."""
        return self.kp * (u - q) - self.kd * qd


def servo_parameters(model: mujoco.MjModel, actuator_ids: np.ndarray) -> ServoParameters:
    """Read servo gains and limits; fail loudly if an actuator is not a position servo."""
    ids = np.asarray(actuator_ids)
    kp = model.actuator_gainprm[ids, 0].copy()
    bias = model.actuator_biasprm[ids, :3]
    ok = (
        np.all(model.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
        and np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
        and np.all(model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_NONE)
        and np.allclose(bias[:, 0], 0.0)
        and np.allclose(bias[:, 1], -kp)
        and np.all(model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
        and np.allclose(model.actuator_gear[ids, 0], 1.0)
    )
    if not ok:
        raise ValueError("Prediction model expects position-servo actuators on arm joints")
    force = model.actuator_forcerange[ids].copy()
    force[~model.actuator_forcelimited[ids].astype(bool)] = [-np.inf, np.inf]
    ctrl = model.actuator_ctrlrange[ids].copy()
    ctrl[~model.actuator_ctrllimited[ids].astype(bool)] = [-np.inf, np.inf]
    return ServoParameters(kp=kp, kd=-bias[:, 2].copy(), force_limits=force, ctrl_limits=ctrl)


def is_torque_motor(model: mujoco.MjModel, actuator_ids: np.ndarray) -> bool:
    """True if every actuator is a direct joint-torque motor (gain 1, no bias, no dynamics)."""
    ids = np.asarray(actuator_ids)
    return bool(
        np.all(model.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
        and np.allclose(model.actuator_gainprm[ids, 0], 1.0)
        and np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_NONE)
        and np.all(model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_NONE)
        and np.all(model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
        and np.allclose(model.actuator_gear[ids, 0], 1.0)
    )


def jacobians_plausible(A: np.ndarray, B: np.ndarray) -> bool:
    """False for Jacobians no physical arm interval could produce (finite
    differences straddling a discontinuity, NaN)."""
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        return False
    return np.abs(A).max() < 1e3 and np.abs(B).max() < 1e3 and np.abs(np.linalg.eigvals(A)).max() < 2.0


@dataclass
class AffineModel:
    """x[k+1] = A x[k] + B u[k] + c over one control interval."""

    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    x0: np.ndarray
    u0: np.ndarray
    dt: float
    plausible: bool = True  # Jacobians passed jacobians_plausible (possibly via the backup model)

    def predict(self, x0: np.ndarray, commands: np.ndarray) -> np.ndarray:
        """States x[1..N] for a command sequence (N, dof), starting from x0."""
        xs, x = [], np.asarray(x0, dtype=float)
        for u in commands:
            x = self.A @ x + self.B @ u + self.c
            xs.append(x)
        return np.array(xs)


class MuJoCoPredictionModel:
    def __init__(
        self,
        model: mujoco.MjModel,
        live_data: mujoco.MjData,
        qpos_adr: np.ndarray,
        dof_adr: np.ndarray,
        actuator_ids: np.ndarray,
        substeps: int,
        method: str = "power",
        eps: float = 1e-6,
        centered: bool = False,
    ):
        if method not in ("power", "rollout"):
            raise ValueError(f"Unknown linearization method '{method}'")
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.model = model
        self.live = live_data
        self.scratch = mujoco.MjData(model)
        # Backup Jacobian model: a private copy whose arm actuators have no force
        # limit (smooth servo). With implicitfast, MuJoCo drops the servo damping
        # from the implicit update while a force is clipped, so the true one-step
        # map jumps at the clip boundary and finite differences that straddle it
        # explode (~1/eps). Away from the boundary the true-model Jacobian is the
        # better local model (a clipped joint really has no servo authority), so
        # the backup is used only when the true one is implausible.
        self.jac_model = copy.deepcopy(model)
        self.jac_model.actuator_forcelimited[np.asarray(actuator_ids)] = 0
        self.jac_scratch = mujoco.MjData(self.jac_model)
        self.used_backup_jacobian = False
        self.qpos_adr = np.asarray(qpos_adr)
        self.dof_adr = np.asarray(dof_adr)
        self.actuator_ids = np.asarray(actuator_ids)
        self.dof = len(self.dof_adr)
        self.substeps = substeps
        self.method = method
        self.eps = eps
        # Forward differences: half the cost of centered (26 vs 52 perturbed steps);
        # arm Jacobians differ by <= 1e-6 (relative ~1e-7), measured in v0.3.
        self.centered = centered
        # Position servos: servo gains are needed by the servo MPC. Torque motors:
        # u is the joint torque itself, so only the limits matter.
        self.actuation = "torque" if is_torque_motor(model, self.actuator_ids) else "position"
        self.servo = servo_parameters(model, self.actuator_ids) if self.actuation == "position" else None
        force = model.actuator_forcerange[self.actuator_ids].copy()
        force[~model.actuator_forcelimited[self.actuator_ids].astype(bool)] = [-np.inf, np.inf]
        self.force_limits = force
        self.ctrl_limits = model.actuator_ctrlrange[self.actuator_ids].copy()

        nx = 2 * model.nv + model.na
        self._A1 = np.zeros((nx, nx))
        self._B1 = np.zeros((nx, model.nu))
        # Arm rows/columns inside MuJoCo's tangent state [dq (nv), dqd (nv), act (na)].
        self._x_idx = np.concatenate([self.dof_adr, model.nv + self.dof_adr])
        self._state = np.zeros(mujoco.mj_stateSize(model, _COPY_SPEC))

    @property
    def dt(self) -> float:
        return self.substeps * self.model.opt.timestep

    # -- scratch-data helpers --------------------------------------------------

    def _load(self, q: np.ndarray, qd: np.ndarray, u: np.ndarray, jac: bool = False) -> None:
        """Copy the live state into scratch, then override the arm state and command."""
        m, s = (self.jac_model, self.jac_scratch) if jac else (self.model, self.scratch)
        mujoco.mj_getState(self.model, self.live, self._state, _COPY_SPEC)
        mujoco.mj_setState(m, s, self._state, _COPY_SPEC)
        s.xfrc_applied[:] = 0.0
        s.qfrc_applied[:] = 0.0
        s.qpos[self.qpos_adr] = q
        s.qvel[self.dof_adr] = qd
        s.ctrl[self.actuator_ids] = u

    def _arm_state(self, jac: bool = False) -> np.ndarray:
        s = self.jac_scratch if jac else self.scratch
        return np.concatenate([s.qpos[self.qpos_adr], s.qvel[self.dof_adr]])

    def _rollout_from_loaded(self, u: np.ndarray, jac: bool = False) -> np.ndarray:
        m, s = (self.jac_model, self.jac_scratch) if jac else (self.model, self.scratch)
        s.ctrl[self.actuator_ids] = u
        mujoco.mj_step(m, s, self.substeps)
        return self._arm_state(jac)

    # -- public API ------------------------------------------------------------

    def rollout(self, q: np.ndarray, qd: np.ndarray, commands: np.ndarray) -> np.ndarray:
        """Nonlinear MuJoCo rollout on scratch data: states after each held command."""
        commands = np.atleast_2d(commands)
        self._load(q, qd, commands[0])
        return np.array([self._rollout_from_loaded(u) for u in commands])

    def linearize(
        self, q: np.ndarray, qd: np.ndarray, u_lin: np.ndarray, method: str | None = None
    ) -> AffineModel:
        """Affine model of one control interval around (x0, u_lin).

        A, B are the Jacobians of the interval map at (x0, u_lin); c makes the
        model exact there. Pick u_lin as the command you expect to apply: with
        "rollout" the force-clip spike of stepping to it is then part of both
        the drift and the sensitivities.
        """
        q, qd, u_lin = (np.asarray(v, dtype=float) for v in (q, qd, u_lin))
        x0 = np.concatenate([q, qd])
        jacobians = self._jacobians_power if (method or self.method) == "power" else self._jacobians_rollout
        A, B = jacobians(q, qd, u_lin, jac=False)
        plausible = jacobians_plausible(A, B)
        self.used_backup_jacobian = not plausible
        if self.used_backup_jacobian:
            A, B = jacobians(q, qd, u_lin, jac=True)
            plausible = jacobians_plausible(A, B)
        self._load(q, qd, u_lin)
        x1 = self._rollout_from_loaded(u_lin)
        c = x1 - A @ x0 - B @ u_lin
        plausible = plausible and bool(np.all(np.isfinite(c)))
        return AffineModel(A=A, B=B, c=c, x0=x0, u0=u_lin.copy(), dt=self.dt, plausible=plausible)

    def _jacobians_power(self, q, qd, u0, jac=False):
        m, s = (self.jac_model, self.jac_scratch) if jac else (self.model, self.scratch)
        self._load(q, qd, u0, jac=jac)
        mujoco.mjd_transitionFD(m, s, self.eps, self.centered, self._A1, self._B1, None, None)
        # Compose the arm block only. The finger joints rest on their joint limit
        # (gripper open), so finite differences across that limit give huge
        # finger-row entries; composing the full matrix leaks them into the arm
        # block (spectral radius ~1e22 observed). Fingers are frozen in this model.
        ix = self._x_idx
        A1 = self._A1[np.ix_(ix, ix)]
        B1 = self._B1[np.ix_(ix, self.actuator_ids)]
        A_n = np.eye(A1.shape[0])
        B_n = np.zeros_like(B1)
        for _ in range(self.substeps):
            B_n = A1 @ B_n + B1
            A_n = A1 @ A_n
        return A_n, B_n

    def _jacobians_rollout(self, q, qd, u0, jac=False):
        n, eps = self.dof, self.eps
        self._load(q, qd, u0, jac=jac)
        base = self._rollout_from_loaded(u0, jac=jac)
        A = np.zeros((2 * n, 2 * n))
        B = np.zeros((2 * n, n))
        for i in range(2 * n):
            dq, dqd = np.zeros(n), np.zeros(n)
            (dq if i < n else dqd)[i % n] = eps
            self._load(q + dq, qd + dqd, u0, jac=jac)
            A[:, i] = (self._rollout_from_loaded(u0, jac=jac) - base) / eps
        for j in range(n):
            du = np.zeros(n)
            du[j] = eps
            self._load(q, qd, u0 + du, jac=jac)
            B[:, j] = (self._rollout_from_loaded(u0 + du, jac=jac) - base) / eps
        return A, B

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Joint torque needed to hold configuration q at rest (gravity only)."""
        self._load(q, np.zeros(self.dof), self.scratch.ctrl[self.actuator_ids])
        self.scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.scratch)
        return self.scratch.qfrc_bias[self.dof_adr].copy()
