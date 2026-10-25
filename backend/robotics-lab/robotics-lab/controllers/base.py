"""Controller interface.

A controller maps a reference (desired joint positions/velocities) and the
measured robot state to actuator commands. It never writes simulation state.
"""

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (integrators, filters)."""

    @abstractmethod
    def compute(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        dt: float,
        t: float | None = None,
        qdd_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return actuator commands for the arm given measured and reference state.

        `t` is the current simulation time; controllers that schedule work or
        look ahead along the reference (MPC) use it, others may ignore it.
        `qdd_ref` is the reference acceleration, used by torque-level
        feedforward (computed-torque control); position-servo controllers ignore it.
        """
