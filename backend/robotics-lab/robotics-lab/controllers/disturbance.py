"""Joint-torque disturbance observer for torque-actuated arms (v0.4).

The plant obeys (in the controller's model)

    M(q) qdd + h(q, qd) = tau_applied + d

where d lumps everything the model does not know: external pushes, payloads
and model error (wrong masses, wrong gravity torque). Each physics step the
observer measures the acceleration the last applied torque actually produced
and asks the model which torque would have produced it:

    d_raw[k] = ID(q[k-1], qd[k-1], (qd[k] - qd[k-1]) / dt) - tau_applied[k-1]
    d_hat[k] = d_hat[k-1] + (dt / T) (d_raw[k] - d_hat[k-1])      first-order filter
    d_hat    = clip(d_hat, -limit, limit)

ID is MuJoCo's inverse dynamics of the *discrete-time* step
(mjENBL_INVDISCRETE): with the implicitfast integrator the joint damping acts
implicitly, and continuous-time inverse dynamics would leave a residual of
dt * damping * qdd (~0.02 Nm here) in d_raw even with a perfect model.

Controllers use d_hat as a known input disturbance: computed-torque control
subtracts it from its torque, the torque MPC adds B d_hat to its prediction
model (offset-free MPC).

Anti-windup: d_raw is formed from the torque that was actually applied (after
clipping to the actuator limits), so the estimate does not grow while the
actuators saturate, unlike an integrator of the tracking error; the clip on
d_hat bounds it if the measurement is ever wrong. The observer only reads
measured joint states: it never sees the simulator's applied forces or its
true (possibly different) masses.
"""

import copy

import mujoco
import numpy as np

from controllers.dynamics import ArmInverseDynamics


class JointDisturbanceObserver:
    def __init__(
        self,
        model: mujoco.MjModel,
        live_data: mujoco.MjData,
        qpos_adr,
        dof_adr,
        time_constant: float = 0.02,
        limit: np.ndarray | float = np.inf,
    ):
        if time_constant <= 0:
            raise ValueError("time_constant must be > 0")
        inv_model = copy.deepcopy(model)  # private copy: the flag must not change anyone else's model
        inv_model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        self.dynamics = ArmInverseDynamics(inv_model, live_data, qpos_adr, dof_adr)
        self.dof = len(dof_adr)
        self.time_constant = time_constant
        self.limit = np.broadcast_to(np.asarray(limit, dtype=float), (self.dof,)).copy()
        self.reset()

    def reset(self) -> None:
        self.estimate = np.zeros(self.dof)
        self.raw = np.zeros(self.dof)
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._tau: np.ndarray | None = None

    def update(self, q: np.ndarray, qd: np.ndarray, dt: float) -> np.ndarray:
        """Fold in the newest measurement (state after the last applied torque); returns d_hat."""
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        if self._prev is not None and self._tau is not None:
            q0, qd0 = self._prev
            raw = self.dynamics.torque(q0, qd0, (qd - qd0) / dt) - self._tau
            if np.all(np.isfinite(raw)):
                self.raw = raw
                alpha = min(dt / self.time_constant, 1.0)
                self.estimate = np.clip(self.estimate + alpha * (raw - self.estimate), -self.limit, self.limit)
        self._prev = (q.copy(), qd.copy())
        self._tau = None
        return self.estimate

    def applied(self, tau: np.ndarray) -> None:
        """Record the torque actually sent to the actuators for the coming step."""
        self._tau = np.asarray(tau, float).copy()
