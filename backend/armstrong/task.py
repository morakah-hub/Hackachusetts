"""One pick-and-place leg: a bounded, event-driven state machine (Armstrong's supervisor, generalised to any
box object and any placement height, so it can also stack).

    observe -> plan -> approach -> descend -> close -> lift -> transport -> lower -> release
            -> retreat -> verify

`program()` is a generator; each `yield` lets the simulation worker take one physics step. Phases end on
measured conditions (arm at rest, finger width, camera observation) or on a bounded timeout that fails the
leg with a stage and a reason.

Information boundary. In camera mode the handled object's pose comes only from `estimate_object` on a fresh
RGB-D frame; the scene layout only says where to look. Other objects and the table are known geometry for
collision checks. Known-world mode reads the simulator pose and is labelled as such. The live object is
moved only by contact: nothing here writes its state or attaches it to the hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import rot_z
from .perception import estimate_object
from .robot import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_Q
from .rules import grasp_depth
from .trajectory import JointMove, LineInfeasible, arc_carry, cartesian_line

PHASES = ("observe", "plan", "approach", "descend", "close", "lift", "transport", "lower", "release",
          "retreat", "verify")
PHASE_LABELS = {"observe": "Looking with the camera", "plan": "Planning the motion",
                "approach": "Moving above the object", "descend": "Reaching down", "close": "Closing the gripper",
                "lift": "Lifting", "transport": "Carrying", "lower": "Putting it down", "release": "Letting go",
                "retreat": "Moving back", "verify": "Checking the result"}
APPROACH_HEIGHT = 0.10
PLACE_CLEARANCE = 0.002
WIDTH_TOL = 0.007
VERIFY_TOL_XY = 0.015


class TaskFailure(Exception):
    def __init__(self, stage: str, code: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.code, self.reason = stage, code, reason


@dataclass
class Leg:
    """One pick-and-place move, as planned (planner.py) and executed (PickPlaceLeg)."""

    object_id: str
    object_name: str
    size: list[float]
    color: str
    start_pos: list[float]
    start_quat: list[float]
    target: list[float]                     # object centre at the placement
    yaw: float                              # object heading at the placement
    supports_from: list[str]
    supports_to: list[str]
    purpose: str = "move"                   # move | buffer
    label: str = ""
    grasp_rel: float | None = None          # closing axis relative to the object's x axis
    grasp_pick: float | None = None         # world angle of the closing axis at the pick
    grasp_place: float | None = None        # ... and at the placement
    width: float | None = None
    lift: float | None = None
    problems: list = field(default_factory=list)

    @property
    def start_yaw(self) -> float:
        w, x, y, z = self.start_quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "object_name": self.object_name, "purpose": self.purpose,
                "label": self.label, "from": [round(v, 4) for v in self.start_pos],
                "to": [round(v, 4) for v in self.target], "yaw": self.yaw, "ok": not self.problems,
                "problems": [p.to_dict() for p in self.problems],
                "grasp": None if self.grasp_pick is None else {
                    "yaw": self.grasp_pick, "width_mm": 1e3 * self.width,
                    "depth_mm": 1e3 * grasp_depth(self.size[2]), "lift_mm": 1e3 * self.lift}}


def hand_R(R_home: np.ndarray, angle: float) -> np.ndarray:
    """Top-down hand orientation whose finger closing axis points along world angle `angle` (at home it is +x)."""
    return rot_z(angle) @ R_home


def closest_rep(angle: float, ref: float, period: float) -> float:
    return float(ref + ((angle - ref + period / 2) % period) - period / 2)


class PickPlaceLeg:
    def __init__(self, sim, leg: Leg, mode: str):
        self.sim = sim
        self.leg = leg
        self.mode = mode
        self.oid = leg.object_id
        self.body = sim.ids.objects[leg.object_id].body
        self.qadr = sim.ids.objects[leg.object_id].qadr
        self.size = np.asarray(leg.size, float)
        self.phase = "observe"
        self.phase_log: list[dict] = []
        self.estimate: dict | None = None
        self.verify_estimate: dict | None = None
        self.plan_info: dict = {}
        self.grasp_width: float | None = None
        _, self.R_home = sim.kin.fk(HOME_Q)

    # ------------------------------------------------------------------ helpers

    def _enter(self, phase: str, note: str = "") -> None:
        t = self.sim.data.time
        if self.phase_log:
            self.phase_log[-1]["t_end"] = t
        self.phase = phase
        self.phase_log.append({"phase": phase, "t_start": t, "t_end": None, "note": note})
        self.sim.log(f"{self.leg.object_name}: {phase}" + (f" ({note})" if note else ""))

    def _wait(self, cond, timeout: float, stage: str, code: str, reason: str, check=None):
        deadline = self.sim.data.time + timeout
        while not cond():
            if self.sim.data.time >= deadline:
                raise TaskFailure(stage, code, reason)
            if check is not None:
                check()
            yield

    def _execute(self, seg, check=None):
        sim = self.sim
        sim.set_reference(seg)
        n = {"k": 0}

        def at_rest():
            if sim.reference_finished() and np.abs(sim.arm_qvel()).max() < 0.03:
                n["k"] += 1
            else:
                n["k"] = 0
            return n["k"] >= 20
        yield from self._wait(at_rest, seg.duration + 2.0, self.phase, "CONTROL_TIMEOUT",
                              "arm did not come to rest within 2 s after the planned motion", check)

    def _fingers_settled(self):
        st = {"last": self.sim.finger_width(), "n": 0, "t0": self.sim.data.time}

        def cond():
            w = self.sim.finger_width()
            st["n"] = st["n"] + 1 if abs(w - st["last"]) < 2e-5 else 0
            st["last"] = w
            return self.sim.data.time - st["t0"] > 0.15 and st["n"] >= 25
        return cond

    def _ik(self, p, R, what: str, seeds=None):
        kin = self.sim.kin
        seeds = seeds or [self.sim.arm_q(), HOME_Q]
        r = kin.solve_multi(p, R, seeds, rest=HOME_Q)
        if not r.ok:
            raise TaskFailure("plan", "PLAN_FAILED", f"{what} at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m is "
                                                     f"unreachable with a top-down grasp: {r.reason}")
        return r.q

    def _locate(self, stage: str, prior_xy, support_z: float, prior_yaw: float) -> dict:
        """Object pose for planning or verification, from the camera (or labelled simulator truth)."""
        sim = self.sim
        if self.mode == "known_world":
            p, yaw = sim.truth_pose(self.oid)
            return {"id": self.oid, "valid": True, "reasons": [], "position_m": p.tolist(), "yaw_rad": yaw,
                    "source": "known_world (simulator pose, development mode)"}
        obs = sim.observe()
        cat = {"id": self.oid, "name": self.leg.object_name, "color": self.leg.color, "size_m": list(self.size),
               "support_z_m": support_z, "prior_xy_m": list(prior_xy), "prior_yaw_rad": prior_yaw}
        est = estimate_object(obs, cat, current_revision=sim.scene_revision)
        sim.publish_estimate(est)
        if not est["valid"]:
            raise TaskFailure(stage, "PERCEPTION_UNAVAILABLE",
                              "camera observation invalid, motion not started: " + "; ".join(est["reasons"]))
        return est

    def _check(self, qs, what: str, held_ok=frozenset(), carry=None):
        kin = self.sim.kin
        ok_bodies = frozenset(held_ok)
        hit = kin.path_collision(qs, held=self.body, carry=carry, held_ok=ok_bodies, held_qadr=self.qadr)
        if hit:
            raise TaskFailure(self.phase, "PLAN_FAILED", f"{what} is not collision-free: {hit}")

    # ------------------------------------------------------------------ the leg

    def program(self):
        sim, leg = self.sim, self.leg
        h = float(self.size[2])
        depth = grasp_depth(h)
        ids = sim.ids
        support_from = [ids.objects[s].body for s in leg.supports_from if s in ids.objects]
        support_to = [ids.objects[s].body for s in leg.supports_to if s in ids.objects]
        start_bottom = float(leg.start_pos[2]) - h / 2

        self._enter("observe", "camera RGB-D frame" if self.mode == "camera_rgbd" else "known-world pose")
        n = {"k": 0}

        def still():
            n["k"] = n["k"] + 1 if np.abs(sim.arm_qvel()).max() < 0.02 else 0
            return n["k"] >= 10
        yield from self._wait(still, 2.0, "observe", "CONTROL_TIMEOUT", "arm not at rest before observing")
        est = self._locate("observe", leg.start_pos[:2], start_bottom, leg.start_yaw)
        self.estimate = est
        c = np.asarray(est["position_m"], float)
        yaw_est = float(est["yaw_rad"] or 0.0)
        yield

        self._enter("plan")
        sim.thinking = True
        a_pick = closest_rep(yaw_est + leg.grasp_rel, leg.grasp_pick, np.pi)
        R_pick = hand_R(self.R_home, a_pick)
        R_place = hand_R(self.R_home, leg.grasp_place)
        top = c[2] + h / 2
        p_grasp = np.array([c[0], c[1], top - depth])
        p_pre = p_grasp + [0, 0, APPROACH_HEIGHT]
        tgt = np.asarray(leg.target, float)
        p_place = np.array([tgt[0], tgt[1], tgt[2] + h / 2 - depth + PLACE_CLEARANCE])
        p_preplace = p_place + [0, 0, leg.lift]
        q_pre = self._ik(p_pre, R_pick, "pre-grasp pose")
        self._ik(p_grasp, R_pick, "grasp pose", seeds=[q_pre])
        q_preplace = self._ik(p_preplace, R_place, "pre-place pose", seeds=[q_pre, HOME_Q])
        self._ik(p_place, R_place, "place pose", seeds=[q_preplace])
        self.plan_info = {"object": leg.object_name, "grasp_angle_rad": a_pick, "estimate_m": c.tolist(),
                          "estimate_yaw_rad": yaw_est, "pre_grasp_m": p_pre.tolist(), "grasp_m": p_grasp.tolist(),
                          "place_m": p_place.tolist(), "target_m": tgt.tolist(), "lift_m": leg.lift,
                          "planning_world": "object from " + est.get("source", "camera_rgbd") +
                                            "; table and other objects are known geometry"}
        sim.publish_plan(self.plan_info)
        kin = sim.kin
        kin.load(sim.data.qpos)
        kin.set_object(self.oid, c, [np.cos(yaw_est / 2), 0, 0, np.sin(yaw_est / 2)])
        approach = JointMove(sim.arm_q(), q_pre, min_time=1.5)
        self._check(approach.samples(40), "the approach path", held_ok=[*support_from, ids.table_body])
        sim.thinking = False
        yield

        self._enter("approach")
        yield from self._execute(approach)

        self._enter("descend")
        try:
            seg = cartesian_line(kin, sim.reference_end(), p_grasp, R_pick, speed=0.10, min_time=1.0)
        except LineInfeasible as exc:
            raise TaskFailure("descend", "PLAN_FAILED", str(exc)) from None
        self._check(seg.samples(), "reaching down", held_ok=[*support_from, ids.table_body])
        yield from self._execute(seg)

        self._enter("close")
        sim.set_gripper(GRIPPER_CLOSED)
        yield from self._wait(self._fingers_settled(), 2.0, "close", "GRASP_FAILED", "fingers did not come to rest")
        width = sim.finger_width()
        self.grasp_width = width
        expected = float(leg.width)
        if width > expected + WIDTH_TOL:
            raise TaskFailure("close", "GRASP_FAILED", f"gripper did not close: fingers at {1e3 * width:.1f} mm "
                                                       f"({leg.object_name} is {1e3 * expected:.0f} mm wide)")
        if width < expected - WIDTH_TOL:
            raise TaskFailure("close", "GRASP_FAILED", f"fingers closed to {1e3 * width:.1f} mm: "
                                                       f"{leg.object_name} is not between the fingers "
                                                       f"(expected {1e3 * expected:.0f} mm)")

        def holding():
            w = sim.finger_width()
            if w < expected - WIDTH_TOL:
                raise TaskFailure(self.phase, "GRASP_FAILED", f"finger width dropped to {1e3 * w:.1f} mm: "
                                                              f"{leg.object_name} slipped out of the gripper")

        # Carried object in the hand frame (from the estimate and the planned grasp; no simulator truth).
        q_g = sim.reference_end()
        hp, hR = kin.hand_pose(q_g)
        off = hR.T @ (c - hp)
        Rrel = hR.T @ rot_z(yaw_est)
        carry = (off, Rrel)

        self._enter("lift")
        sim.thinking = True
        p_now, _ = kin.fk(q_g)
        try:
            seg = cartesian_line(kin, q_g, p_now + [0, 0, leg.lift], R_pick, speed=0.12, min_time=1.0)
        except LineInfeasible as exc:
            raise TaskFailure("lift", "PLAN_FAILED", str(exc)) from None
        self._check(seg.samples(), "lifting", held_ok=[*support_from, ids.table_body], carry=carry)
        sim.thinking = False
        yield from self._execute(seg, holding)

        self._enter("transport")
        sim.thinking = True
        try:
            move = arc_carry(kin, sim.reference_end(), p_preplace, a_pick, leg.grasp_place, self.R_home)
            self._check(move.samples()[::2], "the carry arc", carry=carry)
            self.plan_info["transport"] = "arc"
        except (LineInfeasible, TaskFailure) as exc:
            sim.log(f"carry arc not usable ({exc}); using a joint-space move", "warn")
            move = JointMove(sim.reference_end(), q_preplace, min_time=1.5)
            self._check(move.samples(50), "the carry path", carry=carry)
            self.plan_info["transport"] = "joint path"
        sim.thinking = False
        yield from self._execute(move, holding)

        self._enter("lower")
        sim.thinking = True
        try:
            seg = cartesian_line(kin, sim.reference_end(), p_place, R_place, speed=0.08, min_time=1.2)
        except LineInfeasible as exc:
            raise TaskFailure("lower", "PLAN_FAILED", str(exc)) from None
        self._check(seg.samples(), "putting it down", held_ok=[*support_to, ids.table_body], carry=carry)
        sim.thinking = False
        yield from self._execute(seg, holding)

        self._enter("release")
        sim.set_gripper(GRIPPER_OPEN)
        yield from self._wait(self._fingers_settled(), 2.0, "release", "GRASP_FAILED", "fingers did not open")
        if sim.finger_width() < expected + 0.015:
            raise TaskFailure("release", "GRASP_FAILED", f"fingers opened only to {1e3 * sim.finger_width():.1f} mm")

        self._enter("retreat")
        p_now, _ = kin.fk(sim.reference_end())
        try:
            seg = cartesian_line(kin, sim.reference_end(), p_now + [0, 0, APPROACH_HEIGHT], R_place, speed=0.14, min_time=0.8)
        except LineInfeasible as exc:
            raise TaskFailure("retreat", "PLAN_FAILED", str(exc)) from None
        yield from self._execute(seg)
        yield from self._execute(JointMove(sim.reference_end(), HOME_Q, min_time=1.5))

        self._enter("verify", "re-observe the placed object")
        t_end = sim.data.time + 0.3
        yield from self._wait(lambda: sim.data.time >= t_end, 1.0, "verify", "CONTROL_TIMEOUT", "never")
        est = self._locate("verify", tgt[:2], float(tgt[2]) - h / 2, leg.yaw)
        self.verify_estimate = est
        err = float(np.linalg.norm(np.asarray(est["position_m"][:2]) - tgt[:2]))
        est["placement_error_xy_m"] = err
        if err > VERIFY_TOL_XY:
            raise TaskFailure("verify", "PLACE_FAILED", f"{'camera' if self.mode == 'camera_rgbd' else 'known-world'} "
                                                        f"check: {leg.object_name} is {1e3 * err:.1f} mm from the target")
        self.phase_log[-1]["t_end"] = sim.data.time

