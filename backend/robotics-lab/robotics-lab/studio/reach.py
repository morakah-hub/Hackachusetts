"""Top-down grasp probes: pose IK plus the planner's collision rules, on a private scene copy.

Used to measure the work area (workspace.py) and for the pre-flight checks of task steps (planner.py).
A probe checks the end points of a pick or place the way PickPlaceEpisode will execute it: the pre-grasp
(approach height above the grasp, everything is an obstacle) and the grasp itself (open fingers around
the object, fingertips allowed close to the support surface). It does not plan the connecting motions;
the episode does that and reports its own failures.
"""

import numpy as np

from planning.collision import CollisionChecker
from simulation.engine import SimulationEngine
from simulation.kinematics import ArmKinematics
from studio.robot import wrap_angle
from studio.settings import StudioSettings
from tasks.pick_place import CartesianSegment, PathInfeasible, carry_arc_points, top_down_rotation
from tasks.trajectory import KinematicLimits

IK_ITER = 200


class GraspProbe:
    def __init__(self, settings: StudioSettings, engine: SimulationEngine):
        pp = settings.pick_place
        pl = pp["planning"]
        margin = pl.get("joint_limit_margin", 0.05)
        self.engine = engine
        self.kin = ArmKinematics(engine, limit_margin=margin)
        self.chk = CollisionChecker.from_engine(engine, clearance=pl["clearance"], cartesian_step=pl["cartesian_step"],
                                                max_joint_step=pl["max_joint_step"], limit_margin=margin,
                                                exact_box_box=pl.get("exact_box_box", True))
        col = pp["collision"]
        self.approach = float(pp["grasp"]["approach_height"])
        self.finger_support_min = col.get("finger_table_min")
        self.gripper_object_min = float(col["gripper_object_min"])
        self.object_support_min = float(col["object_table_min"])
        c = self.chk
        self.fingers = c.body_geoms("left_finger", "right_finger")
        self.gripper = c.body_geoms("hand", "left_finger", "right_finger")
        self.hand = c.body_geoms("hand")
        self.table = c.body_geoms("table")
        self.home = settings.home.copy()
        self.base_qpos = engine.data.qpos.copy()
        # Sign of the hand-yaw change per unit of joint 7 (top-down posture), for IK seeds.
        _, r0 = self.kin.fk_pose(self.home)
        q = self.home.copy()
        q[6] += 0.1
        _, r1 = self.kin.fk_pose(q)
        dyaw = wrap_angle(np.arctan2(r1[1, 1], r1[0, 1]) - np.arctan2(r0[1, 1], r0[0, 1]))
        self.q7_sign = 1.0 if dyaw > 0 else -1.0
        self._home_yaw = float(np.arctan2(r0[1, 1], r0[0, 1]))  # finger axis (site y) angle at the rest posture

    # ------------------------------------------------------------------ rules

    def _rules(self, mode: str, obj: str | None = None, supports: tuple[str, ...] = ()) -> None:
        """Collision exceptions of PickPlaceEpisode._configure_checker for the probed pose."""
        c = self.chk
        c.detach()
        c.clear_allowed()
        support = self.table if not supports else np.r_[self.table, c.body_geoms(*supports)].astype(int)
        if obj is None:
            if mode == "grasp" and self.finger_support_min is not None:
                c.allow(self.fingers, support, self.finger_support_min)
            return
        og = c.body_geoms(obj)
        if mode == "grasp":            # open fingers around the object (descend end / lift start)
            c.allow(self.gripper, og, self.gripper_object_min)
            if self.finger_support_min is not None:
                c.allow(self.fingers, support, self.finger_support_min)
        elif mode == "carry":          # object held: finger contact intended, palm keeps its distance
            c.allow(self.fingers, og, -np.inf)
            c.allow(self.hand, og, self.gripper_object_min)
        elif mode == "place":          # object held and resting on its support
            c.allow(self.fingers, og, -np.inf)
            c.allow(self.hand, og, self.gripper_object_min)
            c.allow(og, support, self.object_support_min)
            if self.finger_support_min is not None:
                c.allow(self.fingers, support, self.finger_support_min)
        elif mode != "free":
            raise ValueError(mode)

    # ------------------------------------------------------------------ IK

    def seeds(self, xy, yaw: float) -> list[np.ndarray]:
        """Rest posture turned towards the point, joint 7 set for the requested finger axis."""
        theta = float(np.arctan2(xy[1], xy[0]))
        out = []
        # Top-down: hand yaw ~ home yaw + (q1 - q1_home) + q7_sign * (q7 - q7_home). The second seed uses the
        # other sign in case the posture differs from the rest posture's.
        for sign in (self.q7_sign, -self.q7_sign):
            q = self.home.copy()
            q[0] = np.clip(theta, -2.75, 2.75)
            q[6] = np.clip(self.home[6] + sign * wrap_angle(yaw - self._home_yaw - (q[0] - self.home[0])),
                           -2.80, 2.80)
            out.append(q)
        return out

    def solve(self, position, yaw: float, seeds, qpos=None, mode: str = "free", obj=None, supports=()):
        """First pose-IK solution over the seeds that the checker accepts. Returns (q or None, reason)."""
        R = top_down_rotation(yaw)
        self.chk.set_fixed_qpos(self.base_qpos if qpos is None else qpos)
        self._rules(mode, obj, supports)
        reason, converged = "out of reach", False
        tried = []
        for s in seeds:
            if s is None or any(np.allclose(s, t, atol=1e-6) for t in tried):
                continue
            tried.append(s)
            res = self.kin.solve_pose_ik(np.asarray(position, float), R, s, max_iter=IK_ITER)
            if not res.success:
                continue
            converged = True
            rep = self.chk.check(res.q)
            if rep.valid:
                return res.q, ""
            reason = rep.reason
        return None, (reason if converged else "out of reach")

    def top_down(self, center_xy, z_grasp: float, yaw: float, warm=None, qpos=None, qpos_above=None, obj=None,
                 supports=(), holding: bool = False, above_height: float | None = None):
        """Pre-grasp/pre-place and grasp/place poses at (x, y).

        Pick (holding=False): the hand is empty; `qpos` has the object where it rests, and the open fingers
        close around it at `z_grasp`. Place (holding=True): `qpos` has the object at its placement and
        `qpos_above` has it raised by `above_height` (carried; default: the approach height).
        Returns (q or None, reason)."""
        x, y = float(center_xy[0]), float(center_xy[1])
        fresh = self.seeds((x, y), yaw)
        above = np.array([x, y, z_grasp + (self.approach if above_height is None else float(above_height))])
        q_pre, why = self.solve(above, yaw, [warm, *fresh], qpos_above if holding else qpos,
                                "carry" if holding else "free", obj, supports)
        if q_pre is None:
            return None, f"above the spot: {why}"
        q, why = self.solve(np.array([x, y, z_grasp]), yaw, [q_pre, warm, *fresh], qpos,
                            "place" if holding else "grasp", obj, supports)
        if q is None:
            return None, f"at the spot: {why}"
        return q, ""

    def carry_arc(self, q_lift: np.ndarray, end_position, yaw: float, qpos_lift: np.ndarray, obj: str,
                  limits: KinematicLimits, requested: float = 2.0):
        """The supervisor's arc transport (PickPlaceEpisode._carry_arc) from the lifted configuration: IK walk
        along the arc and a collision check with the object attached to the hand. Returns (ok, reason)."""
        p0, _ = self.kin.fk_pose(q_lift)
        center = self.engine.data.xpos[self.engine.model.body(self.engine.robot.base_body).id][:2]
        pts, length, _ = carry_arc_points(p0, end_position, center)
        if length < 0.02:
            return True, ""
        try:
            seg = CartesianSegment.plan_path(self.kin, q_lift, pts, top_down_rotation(yaw), requested, limits)
        except PathInfeasible as exc:
            return False, f"the wrist cannot keep the grip orientation along the way ({exc})"
        base = np.array(qpos_lift, float)
        base[self.engine.arm_qpos_adr] = q_lift
        self.chk.set_fixed_qpos(base)
        self._rules("carry", obj)
        self.chk.attach(obj, "hand")
        try:
            ok, why = self.chk.path_valid(seg.joint_samples())
        finally:
            self.chk.detach()
        return ok, "" if ok else f"blocked while carrying: {why}"
