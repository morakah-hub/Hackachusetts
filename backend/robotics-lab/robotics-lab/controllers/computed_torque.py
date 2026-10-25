"""Computed-torque control (inverse dynamics + PD) for torque-actuated arms.

    qdd_des = qdd_ref + Kp (q_ref - q) + Kd (qd_ref - qd)
    tau     = InverseDynamics(q, qd, qdd_des)          (MuJoCo mj_inverse)
    tau     = clip(tau, torque limits)

With an exact model the closed-loop error obeys e'' + Kd e' + Kp e = 0, so
Kp = wn^2 and Kd = 2 zeta wn set the error dynamics directly. It runs every
physics step (500 Hz) and is the torque-level baseline for the torque MPC (and
the MPC's fallback). It uses the same MuJoCo model as the simulation, so it has
perfect model knowledge; it ignores velocity and joint-position limits.

v0.4 options (off by default, so the v0.3 baseline is unchanged):

* integral_gain ki: qdd_des += ki * integral(e) dt, i.e. error dynamics
  e''' + Kd e'' + Kp e' + ki e = 0, stable for ki < Kd Kp (Routh). Anti-windup:
  conditional integration (the integral is frozen on joints whose torque is
  clipped) plus a clamp |integral(e)| <= integral_limit.
* observer (controllers/disturbance.py): tau = ID(q, qd, qdd_des) - d_hat, the
  classic disturbance-observer-based computed-torque control. Saturation-safe
  by construction (the observer is driven by the applied torque).
"""

import numpy as np

from controllers.base import Controller
from controllers.disturbance import JointDisturbanceObserver
from controllers.dynamics import ArmInverseDynamics


class ComputedTorqueController(Controller):
    def __init__(
        self,
        dynamics: ArmInverseDynamics,
        torque_limits: np.ndarray,
        natural_frequency: float | np.ndarray = 25.0,
        damping_ratio: float | np.ndarray = 1.0,
        integral_gain: float = 0.0,
        integral_limit: float = 0.1,
        observer: JointDisturbanceObserver | None = None,
    ):
        self.dynamics = dynamics
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        dof = len(self.torque_limits)
        wn = np.broadcast_to(np.asarray(natural_frequency, dtype=float), (dof,))
        zeta = np.broadcast_to(np.asarray(damping_ratio, dtype=float), (dof,))
        self.kp = wn**2
        self.kd = 2 * zeta * wn
        self.ki = float(integral_gain)
        if np.any(self.ki >= self.kd * self.kp):
            raise ValueError(f"integral_gain {self.ki} must be < Kd * Kp = {np.min(self.kd * self.kp):.0f} for stability")
        self.integral_limit = float(integral_limit)
        self.observer = observer
        self.reset()

    def reset(self) -> None:
        self.integral = np.zeros(len(self.torque_limits))
        self._saturated = np.zeros(len(self.torque_limits), dtype=bool)
        if self.observer is not None:
            self.observer.reset()

    @property
    def disturbance_estimate(self) -> np.ndarray:
        return self.observer.estimate.copy() if self.observer is not None else np.zeros(len(self.torque_limits))

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        qdd_ff = np.zeros_like(q) if qdd_ref is None else qdd_ref
        e = np.asarray(q_ref) - q
        if self.ki > 0:
            # Conditional integration: hold the integral on joints that were clipped last step.
            self.integral = np.where(self._saturated, self.integral, self.integral + e * dt)
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        qdd_des = qdd_ff + self.kp * e + self.kd * (qd_ref - qd) + self.ki * self.integral
        tau = self.dynamics.torque(q, qd, qdd_des)
        if self.observer is not None:
            tau = tau - self.observer.update(q, qd, dt)
        out = np.clip(tau, self.torque_limits[:, 0], self.torque_limits[:, 1])
        self._saturated = out != tau
        if self.observer is not None:
            self.observer.applied(out)
        return out
