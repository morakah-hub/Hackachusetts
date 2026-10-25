"""Joint-space PID controller for position-servo actuators.

The Panda's Menagerie actuators are inner-loop PD servos inside MuJoCo
(tau = kp_servo * (ctrl - q) - kd_servo * qdot). On its own, that loop sags
under gravity. This controller closes an outer loop on the *measured* joint
state and shifts the servo setpoint:

    e    = q_ref - q
    ctrl = q_ref + kp * e + ki * integral(e) + kd * (qd_ref - qd) + kff * qd_ref

The feed-forward term kff * qd_ref cancels the servo's own damping lag
(kd_servo * qd_ref / kp_servo); for every Panda actuator kd_servo/kp_servo = 0.1.
The integral term removes the steady-state gravity error; the integrator is
clamped (anti-windup) so a blocked joint cannot wind the setpoint away.
"""

import numpy as np

from controllers.base import Controller


class JointPIDController(Controller):
    def __init__(
        self,
        dof: int,
        kp: float | np.ndarray = 0.5,
        ki: float | np.ndarray = 5.0,
        kd: float | np.ndarray = 0.0,
        velocity_feedforward: float | np.ndarray = 0.1,
        integral_limit: float = 0.05,
        ctrl_limits: np.ndarray | None = None,
    ):
        self.dof = dof
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (dof,)).copy()
        self.ki = np.broadcast_to(np.asarray(ki, dtype=float), (dof,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (dof,)).copy()
        self.kff = np.broadcast_to(np.asarray(velocity_feedforward, dtype=float), (dof,)).copy()
        self.integral_limit = integral_limit
        self.ctrl_limits = ctrl_limits
        self.integral = np.zeros(dof)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        error = q_ref - q
        # Clamp the integral's contribution (rad of setpoint offset) for anti-windup.
        self.integral += error * dt
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(self.ki > 0, self.integral_limit / self.ki, 0.0)
        self.integral = np.clip(self.integral, -bound, bound)

        ctrl = q_ref + self.kp * error + self.ki * self.integral + self.kd * (qd_ref - qd) + self.kff * qd_ref
        if self.ctrl_limits is not None:
            ctrl = np.clip(ctrl, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        return ctrl
