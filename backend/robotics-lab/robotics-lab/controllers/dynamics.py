"""MuJoCo inverse dynamics for the arm, evaluated on private scratch data.

    tau = M(q) qdd + c(q, qd) + g(q) - tau_passive(q, qd)

is the joint torque the arm actuators must produce for acceleration qdd at
state (q, qd), computed exactly by mujoco.mj_inverse (including armature and
joint damping, which MuJoCo applies as a passive force). Finger joints are held
at their measured state with zero acceleration. The live simulation is only read.
"""

import mujoco
import numpy as np

# Same state copy as the prediction model: no external forces (unmeasurable).
_COPY_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS | mujoco.mjtState.mjSTATE_CTRL


class ArmInverseDynamics:
    def __init__(self, model: mujoco.MjModel, live_data: mujoco.MjData, qpos_adr, dof_adr):
        self.model = model
        self.live = live_data
        self.scratch = mujoco.MjData(model)
        self.qpos_adr = np.asarray(qpos_adr)
        self.dof_adr = np.asarray(dof_adr)
        self._state = np.zeros(mujoco.mj_stateSize(model, _COPY_SPEC))

    def _load(self, q: np.ndarray, qd: np.ndarray) -> None:
        mujoco.mj_getState(self.model, self.live, self._state, _COPY_SPEC)
        mujoco.mj_setState(self.model, self.scratch, self._state, _COPY_SPEC)
        self.scratch.qpos[self.qpos_adr] = q
        self.scratch.qvel[self.dof_adr] = qd

    def torque(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Arm joint torques that produce acceleration qdd at state (q, qd)."""
        self._load(q, qd)
        s = self.scratch
        s.qacc[:] = 0.0
        s.qacc[self.dof_adr] = qdd
        mujoco.mj_inverse(self.model, s)
        return s.qfrc_inverse[self.dof_adr].copy()

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Torque that holds configuration q at rest."""
        return self.torque(q, np.zeros(len(self.dof_adr)), np.zeros(len(self.dof_adr)))
