"""One-joint MPC learning sandbox (v0.7). Educational model, not a Panda experiment.

A single rotary joint with inertia I and no gravity or friction, torque held for Ts = 0.1 s:

    x = [theta, omega],  x[k+1] = A x[k] + B tau[k],  A = [[1, Ts], [0, 1]],  B = [Ts^2 / (2 I), Ts / I]

(exact for a constant torque over the interval). The controller is the project's own TorqueMPCController,
given this fixed linear model through a minimal prediction-model stand-in, so the sandbox exercises the same
QP code as the arm. `unconstrained_first_torque` solves the same cost in closed form (no torque limit) by
eliminating the states: it is the hand-checkable reference used in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.base import Controller
from controllers.torque_mpc import TorqueMPCController, TorqueMPCWeights

TS = 0.1
DURATION = 3.0


@dataclass
class _Lin:
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    plausible: bool = True


class _JointModel:
    actuation = "torque"
    dof = 1
    substeps = 1

    def __init__(self, inertia: float, torque_limit: float):
        self.dt = TS
        self.A = np.array([[1.0, TS], [0.0, 1.0]])
        self.B = np.array([[TS * TS / (2 * inertia)], [TS / inertia]])
        self.force_limits = np.array([[-torque_limit, torque_limit]])

    def gravity_torque(self, q):
        return np.zeros(1)

    def linearize(self, q, qd, u):
        return _Lin(self.A, self.B, np.zeros(2))


class _Zero(Controller):
    def reset(self):
        pass

    def compute(self, q, qd, q_ref, qd_ref, dt, t=None, qdd_ref=None):
        return np.zeros(1)


def _check(inertia, torque_limit, horizon, w_position, w_velocity, w_effort, target):
    if not 0.01 <= inertia <= 100 or not 0.01 <= torque_limit <= 1000:
        raise ValueError("inertia must be in [0.01, 100] kg m^2 and the torque limit in [0.01, 1000] Nm")
    if not 2 <= horizon <= 60 or int(horizon) != horizon:
        raise ValueError("horizon must be an integer 2-60")
    if min(w_position, w_velocity, w_effort) < 0 or w_position == 0:
        raise ValueError("weights must be >= 0 and the position weight > 0")
    if abs(target) > 10:
        raise ValueError("target must be within +-10 rad")


def controller(inertia, torque_limit, horizon, w_position, w_velocity, w_effort, target) -> TorqueMPCController:
    model = _JointModel(inertia, torque_limit)
    w = TorqueMPCWeights(position=w_position, velocity=w_velocity, terminal=1.0, effort=w_effort, torque_rate=0.0)
    return TorqueMPCController(model, _Zero(), horizon=int(horizon), weights=w, soft_constraints=False,
                               reference_preview=lambda t: (np.array([target]), np.zeros(1)),
                               solver_settings={"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iter": 20000})


def unconstrained_first_torque(inertia, torque_limit, horizon, w_position, w_velocity, w_effort, target,
                               x0=(0.0, 0.0)) -> float:
    """Closed form: x_k = A^k x0 + sum_i A^(k-1-i) B u_i; minimize sum_k (x_k - r)' Q (x_k - r) + w_e sum (u/tau_max)^2."""
    m = _JointModel(inertia, torque_limit)
    N = int(horizon)
    Q = np.diag([w_position, w_velocity])
    H = np.zeros((N, N))
    g = np.zeros(N)
    r = np.array([target, 0.0])
    for k in range(1, N + 1):
        Ak = np.linalg.matrix_power(m.A, k)
        G = np.zeros((2, N))
        for i in range(k):
            G[:, i] = (np.linalg.matrix_power(m.A, k - 1 - i) @ m.B)[:, 0]
        free = Ak @ np.asarray(x0, float) - r
        H += G.T @ Q @ G
        g += G.T @ Q @ free
    H += np.eye(N) * w_effort / torque_limit**2
    return float(np.linalg.solve(H, -g)[0])


def simulate(inertia=0.5, torque_limit=1.0, horizon=20, w_position=10.0, w_velocity=1.0, w_effort=0.1, target=1.0):
    _check(inertia, torque_limit, horizon, w_position, w_velocity, w_effort, target)
    mpc = controller(inertia, torque_limit, horizon, w_position, w_velocity, w_effort, target)
    plant = _JointModel(inertia, torque_limit)
    x = np.zeros(2)
    ts, theta, omega, taus = [0.0], [0.0], [0.0], []
    plan_t = plan_theta = None
    n = int(round(DURATION / TS))
    for k in range(n):
        u = mpc.compute(x[:1], x[1:], np.array([target]), np.zeros(1), TS, t=k * TS)
        if k == 0:
            p = mpc.last_prediction
            plan_t = (TS * np.arange(0, len(p["states"]) + 1)).tolist()
            plan_theta = [0.0] + p["states"][:, 0].tolist()
        x = plant.A @ x + plant.B[:, 0] * float(u[0])
        taus.append(float(u[0]))
        ts.append((k + 1) * TS)
        theta.append(float(x[0]))
        omega.append(float(x[1]))
    th = np.array(theta)
    tol = 0.01 * max(abs(target), 1e-9)
    inside = np.abs(th - target) <= tol
    settle = None
    for k in range(len(th)):
        if inside[k:].all():
            settle = ts[k]
            break
    over = float(max(0.0, (th - target).max() if target >= 0 else (target - th).max()))
    sat = float(np.mean(np.abs(np.array(taus)) >= torque_limit * (1 - 1e-6)))
    return {"t": ts, "theta": theta, "omega": omega, "tau": taus, "target": target, "plan_t": plan_t,
            "plan_theta": plan_theta, "tau0": taus[0],
            "tau0_unconstrained": unconstrained_first_torque(inertia, torque_limit, horizon, w_position, w_velocity,
                                                             w_effort, target),
            "settle_time": settle, "overshoot": over, "saturated_fraction": sat,
            "A": plant.A.tolist(), "B": plant.B[:, 0].tolist()}
