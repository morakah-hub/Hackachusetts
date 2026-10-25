"""Arm kinematics on private scratch data: FK, Jacobian, IK and collision checks.

Inverse kinematics (position only, 3D) uses damped least squares on MuJoCo's
site Jacobian J (3 x 7):

    dq = J^T (J J^T + lambda^2 I)^-1 e  +  k_null (I - J^+ J) (q_rest - q)

with e the end-effector position error. The step is limited, q is clamped to
the joint limits minus a margin after every step (joint-limit aware), and the
null-space term pulls the redundant 4 DOF towards a rest posture, away from the
limits. The end-effector orientation is left free.

Pose IK (v0.6, `solve_pose_ik`) adds the orientation: the 6-D error, in the world frame, is

    e = [ p_target - p ;  w * log(R_target R^T) ]

where log(.) is the rotation vector (axis * angle, |angle| <= pi) of the rotation that takes
the current orientation R to the target, expressed in world coordinates, and w (m/rad) weights
rotation against position. Its Jacobian is [J_p ; w J_r] with J_r MuJoCo's rotational site
Jacobian, which is also in world coordinates (d(rotation vector) = J_r dq to first order),
so the DLS step is the same formula with a 6 x 7 Jacobian and leaves 1 DOF to the null space.
`solve_ik` (position only) is unchanged.

A target is accepted only if IK converges (error < tol) and the robot is
collision-free at the solution and along the straight joint-space path from
the previous configuration (checked at sampled points). This is collision
*checking*, not planning: a target behind an obstacle is rejected rather than
reached by going around it.

Nothing here writes the live simulation state.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from simulation.engine import SimulationEngine


def rotation_angle(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (rad, in [0, pi]) of the relative rotation R_a^T R_b: the geodesic
    distance between two orientations. Computed through the unit quaternion of
    R_a^T R_b as 2 atan2(|v|, |w|), which stays accurate near 0 and near pi
    (arccos((trace - 1) / 2) loses precision at both ends)."""
    rel = np.asarray(R_a, float).reshape(3, 3).T @ np.asarray(R_b, float).reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rel.ravel())
    return float(2.0 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0])))


def rotation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World-frame rotation vector of R_target R^T (rotate R onto R_target), |angle| <= pi."""
    rel = np.asarray(R_target, float).reshape(3, 3) @ np.asarray(R, float).reshape(3, 3).T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rel.ravel())
    if quat[0] < 0:  # shortest rotation
        quat = -quat
    vec = np.zeros(3)
    mujoco.mju_quat2Vel(vec, quat, 1.0)
    return vec


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    error: float           # final end-effector position error (m)
    iterations: int
    reason: str = ""
    rotation_error: float = 0.0  # final orientation error (rad), pose IK only


class ArmKinematics:
    def __init__(
        self,
        engine: SimulationEngine,
        q_rest: np.ndarray | None = None,
        limit_margin: float = 0.05,
        damping: float = 0.05,
        max_step: float = 0.2,
        null_gain: float = 0.1,
    ):
        self.engine = engine
        self.model = engine.model
        self.scratch = mujoco.MjData(engine.model)
        self.qpos_adr, self.dof_adr = engine.arm_qpos_adr, engine.arm_dof_adr
        self.site_id = engine.ee_site_id
        lo, hi = engine.joint_limits[:, 0], engine.joint_limits[:, 1]
        self.q_min, self.q_max = lo + limit_margin, hi - limit_margin
        self.q_rest = np.asarray(engine.robot.home_qpos if q_rest is None else q_rest, dtype=float)
        self.damping = damping
        self.max_step = max_step
        self.null_gain = null_gain
        self._robot_bodies = set(engine._robot_body_ids())
        self._jacp = np.zeros((3, self.model.nv))

    def _set(self, q: np.ndarray) -> None:
        s = self.scratch
        s.qpos[:] = self.engine.data.qpos  # fingers etc. from the live state (read only)
        s.qpos[self.qpos_adr] = q

    def fk(self, q: np.ndarray) -> np.ndarray:
        self._set(q)
        mujoco.mj_kinematics(self.model, self.scratch)
        return self.scratch.site_xpos[self.site_id].copy()

    def fk_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """End-effector site position (3,) and rotation matrix (3, 3), world frame."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.scratch)
        return self.scratch.site_xpos[self.site_id].copy(), self.scratch.site_xmat[self.site_id].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """3 x 7 positional Jacobian of the end-effector site w.r.t. the arm joints."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        mujoco.mj_jacSite(self.model, self.scratch, self._jacp, None, self.site_id)
        return self._jacp[:, self.dof_adr].copy()

    def jacobian6(self, q: np.ndarray) -> np.ndarray:
        """6 x 7 site Jacobian [positional; rotational], both in world coordinates."""
        self._set(q)
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.scratch, self._jacp, jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

    def solve_pose_ik(self, position: np.ndarray, rotation: np.ndarray, seed: np.ndarray, tol: float = 1e-4,
                      rot_tol: float = 1e-3, max_iter: int = 300, rest: np.ndarray | None = None,
                      rot_weight: float = 0.2) -> IKResult:
        """DLS IK for a full pose (position (3,), rotation matrix (3, 3), world frame).
        Converged when the position error < tol (m) and the orientation error < rot_tol (rad)."""
        target = np.asarray(position, dtype=float)
        R_t = np.asarray(rotation, dtype=float).reshape(3, 3)
        q_rest = self.q_rest if rest is None else np.asarray(rest, dtype=float)
        q = np.clip(np.asarray(seed, dtype=float), self.q_min, self.q_max)
        lam2 = self.damping**2
        W = np.r_[np.ones(3), np.full(3, rot_weight)]
        err, rerr = np.inf, np.inf
        for it in range(1, max_iter + 1):
            p, R = self.fk_pose(q)
            ep, er = target - p, rotation_error(R_t, R)
            err, rerr = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if err < tol and rerr < rot_tol:
                return IKResult(q, True, err, it, rotation_error=rerr)
            J = W[:, None] * self.jacobian6(q)
            e = W * np.r_[ep, er]
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), e)
            J_pinv = J.T @ np.linalg.inv(JJt + 1e-9 * np.eye(6))
            dq += self.null_gain * (np.eye(len(q)) - J_pinv @ J) @ (q_rest - q)
            step = np.abs(dq).max()
            if step > self.max_step:
                dq *= self.max_step / step
            q = np.clip(q + dq, self.q_min, self.q_max)
        return IKResult(q, False, err, max_iter, f"pose IK did not converge (closest {1e3 * err:.1f} mm, "
                                                 f"{np.degrees(rerr):.2f} deg)", rotation_error=rerr)

    def contacts(self, q: np.ndarray) -> list[tuple[str, str]]:
        """Contacts involving the robot at configuration q (environment and self-collision)."""
        self._set(q)
        mujoco.mj_forward(self.model, self.scratch)
        m, d = self.model, self.scratch
        pairs = []
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self._robot_bodies or b2 in self._robot_bodies:
                pairs.append((m.body(b1).name, m.body(b2).name))
        return pairs

    def path_contacts(self, q0: np.ndarray, q1: np.ndarray, samples: int = 25) -> list[tuple[str, str]]:
        """First contacts found along the straight joint-space path q0 -> q1 (empty if free)."""
        for s in np.linspace(0.0, 1.0, samples):
            found = self.contacts(q0 + s * (q1 - q0))
            if found:
                return found
        return []

    def solve_ik(self, target: np.ndarray, seed: np.ndarray, tol: float = 1e-4, max_iter: int = 300,
                 rest: np.ndarray | None = None) -> IKResult:
        """DLS IK from `seed`. The null-space term pulls towards `rest` (default: the rest
        posture); passing the seed itself keeps solutions near different seeds distinct."""
        target = np.asarray(target, dtype=float)
        q_rest = self.q_rest if rest is None else np.asarray(rest, dtype=float)
        q = np.clip(np.asarray(seed, dtype=float), self.q_min, self.q_max)
        lam2 = self.damping**2
        err = np.inf
        for it in range(1, max_iter + 1):
            e = target - self.fk(q)
            err = float(np.linalg.norm(e))
            if err < tol:
                return IKResult(q, True, err, it)
            J = self.jacobian(q)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(3), e)
            J_pinv = J.T @ np.linalg.inv(JJt + 1e-9 * np.eye(3))
            dq += self.null_gain * (np.eye(len(q)) - J_pinv @ J) @ (q_rest - q)
            step = np.abs(dq).max()
            if step > self.max_step:
                dq *= self.max_step / step
            q = np.clip(q + dq, self.q_min, self.q_max)
        return IKResult(q, False, err, max_iter, f"IK did not converge (closest {1e3 * err:.1f} mm)")

    def plan_target(self, target: np.ndarray, q_from: np.ndarray, tol: float = 1e-4,
                    check_path: bool = True) -> IKResult:
        """IK (seeded from q_from, then from the rest posture) plus collision checks.
        check_path=False skips the straight-path check (explicit unsafe comparison mode)."""
        best = None
        for seed in (q_from, self.q_rest):
            res = self.solve_ik(target, seed, tol)
            if best is None or res.error < best.error:
                best = res
            if res.success:
                break
        if not best.success:
            best.reason = best.reason or "unreachable"
            return best
        hit = self.contacts(best.q)
        if hit:
            return IKResult(best.q, False, best.error, best.iterations, f"collision at goal: {hit[0][0]}-{hit[0][1]}")
        hit = self.path_contacts(q_from, best.q) if check_path else []
        if hit:
            return IKResult(best.q, False, best.error, best.iterations, f"collision on path: {hit[0][0]}-{hit[0][1]}")
        return best
