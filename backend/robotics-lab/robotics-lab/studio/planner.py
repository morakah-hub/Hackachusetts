"""Pre-written tasks: catalog, decomposition into pick-and-place legs, and pre-flight checks.

A task becomes an ordered list of legs; each leg is one physical PickPlaceEpisode (pick one object, carry it,
put it down). Rearrangements that would put an object where another one still stands use a temporary spot
first. Long swings around the base are split with a put-down in between, so the hand can change its grip
orientation (joint 7 has to counter-rotate by the swing while the object keeps its heading).

The pre-flight check runs on a private copy of the scene: payload and gripper limits, free target spots, the
grasp and placement poses (pose IK + the planner's collision rules), and the carry arc. It cannot promise
the physical outcome (slip, contact) which only the simulation decides.
"""

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from robots.panda import Panda
from simulation.engine import SimulationEngine
from studio.reach import GraspProbe
from studio.robot import Problem, RobotRules, quat_from_yaw
from studio.scene import (SceneObject, StudioStore, World, footprint, object_footprint, polygons_overlap,
                          supports_of)
from studio.settings import StudioSettings
from tasks.trajectory import KinematicLimits

TASKS = [
    {"id": "move", "title": "Move", "verb": "Move an object",
     "summary": "Pick up one object and put it down where you click.",
     "params": [{"key": "object", "type": "object", "label": "Object"},
                {"key": "target", "type": "point", "label": "Destination"}]},
    {"id": "hand_over", "title": "Bring to me", "verb": "Bring an object to the front",
     "summary": "Carry an object to the hand-over spot at the front edge of the table.",
     "params": [{"key": "object", "type": "object", "label": "Object"}]},
    {"id": "swap", "title": "Swap", "verb": "Swap two objects",
     "summary": "Exchange the places of two objects, using a free spot in between.",
     "params": [{"key": "a", "type": "object", "label": "First object"},
                {"key": "b", "type": "object", "label": "Second object"}]},
    {"id": "stack", "title": "Stack", "verb": "Stack one object on another",
     "summary": "Put one object on top of another one.",
     "params": [{"key": "object", "type": "object", "label": "Object to put on top"},
                {"key": "base", "type": "object", "label": "Object underneath"}]},
    {"id": "line_up", "title": "Line up", "verb": "Line objects up",
     "summary": "Arrange objects in a neat row across the front of the table.",
     "params": [{"key": "objects", "type": "objects", "label": "Objects (empty = all that can be lifted)"},
                {"key": "order", "type": "choice", "label": "Order (left to right)",
                 "choices": [["current", "Keep their order"], ["name", "By name"], ["height", "Shortest first"]],
                 "default": "current"}]},
    {"id": "sort_weight", "title": "Sort by weight", "verb": "Sort objects by weight",
     "summary": "Line objects up from the lightest (left) to the heaviest (right).",
     "params": [{"key": "objects", "type": "objects", "label": "Objects (empty = all that can be lifted)"}]},
]
TASK_IDS = {t["id"]: t for t in TASKS}
AT_TARGET = 0.005        # m: an object this close to its target counts as already there
# Half-extents of the open gripper around the grip centre, plus the planner's 20 mm clearance and a few mm of
# margin (measured from the Panda collision meshes: palm 208 x 63 mm, 37 mm above the grip centre; open
# fingers 133 x 21 mm). Along = the finger axis.
HAND_ALONG, HAND_ACROSS = 0.13, 0.06
FINGERS_ALONG, FINGERS_ACROSS = 0.092, 0.036
ROW_GAP = 0.03           # m: smallest gap between neighbours in a row (more where the open hand needs room)
LEG_SECONDS = 11.0       # typical simulated duration of one leg (v0.6: ~10-11 s)


class PlanError(Exception):
    def __init__(self, message: str, problems: list[Problem] | None = None):
        super().__init__(message)
        self.message, self.problems = message, problems or []


@dataclass
class Leg:
    object_id: str
    object_name: str
    start_pos: list[float]
    start_quat: list[float]
    target: list[float]               # object centre at the placement (x, y, z)
    yaw: float                        # object heading, kept while carried
    supports_from: list[str]
    supports_to: list[str]
    purpose: str = "move"             # move | buffer | via
    label: str = ""
    grasp_yaw: float | None = None
    grasp_width: float | None = None
    depth: float | None = None
    lift: float | None = None
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"object_id": self.object_id, "object_name": self.object_name, "purpose": self.purpose,
                "label": self.label, "from": [round(v, 4) for v in self.start_pos],
                "to": [round(v, 4) for v in self.target], "ok": not self.problems,
                "problems": [p.to_dict() for p in self.problems],
                "grasp": None if self.grasp_yaw is None else {
                    "yaw": self.grasp_yaw, "width_mm": 1e3 * self.grasp_width, "depth_mm": 1e3 * self.depth,
                    "lift_mm": 1e3 * self.lift}}


@dataclass
class TaskPlan:
    task_id: str
    title: str
    summary: str
    params: dict
    legs: list[Leg]
    problems: list[Problem]
    revision: int
    check_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.legs) and all(not leg.problems for leg in self.legs)

    def all_problems(self) -> list[Problem]:
        out = list(self.problems)
        for leg in self.legs:
            out += [p for p in leg.problems if p.code not in {q.code for q in out} or p.object_id]
        return out

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "title": self.title, "summary": self.summary, "params": self.params,
                "ok": self.ok, "revision": self.revision, "legs": [leg.to_dict() for leg in self.legs],
                "problems": [p.to_dict() for p in self.all_problems()], "check_s": round(self.check_s, 2),
                "estimated_s": round(LEG_SECONDS * len(self.legs) + (3.0 if self.legs else 0.0), 0)}


def scene_overrides(world: World, objects: list[SceneObject]) -> dict:
    """environment_overrides for the studio scene: table radius plus every object as a free box."""
    return {"geoms": {"table_top": {"size": [world.table_radius, 0.02, 0.0]}},
            "objects": [{"name": o.id, "pos": list(o.pos), "quat": list(o.quat), "half_size": (0.5 * np.asarray(o.size)).tolist(),
                         "mass": float(o.mass)} for o in objects]}


class Feasibility:
    """Pose IK and collision checks of legs on a private copy of the scene."""

    def __init__(self, settings: StudioSettings, world: World, rules: RobotRules, objects: list[SceneObject]):
        self.settings, self.world, self.rules = settings, world, rules
        self.engine = SimulationEngine(settings.environment, Panda(actuation="torque"), scene_overrides(world, objects))
        self.probe = GraspProbe(settings, self.engine)
        m = self.engine.model
        self.adr = {o.id: int(m.jnt_qposadr[m.joint(f"{o.id}_free").id]) for o in objects}
        self.base = self.engine.data.qpos.copy()
        self.base[self.engine.arm_qpos_adr] = settings.home
        lim = self.engine.joint_limits
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        r = self.engine.robot
        traj = settings.pick_place["trajectory"]
        self.kin_limits = KinematicLimits(velocity=np.asarray(r.velocity_limits) * traj["velocity_scale"],
                                          acceleration=np.asarray(r.acceleration_limits) * traj["acceleration_scale"])
        carry = settings.studio["carry"]
        self.clearance, self.min_lift, self.max_lift = carry["clearance_m"], carry["min_lift_m"], carry["max_lift_m"]

    def qpos(self, state: dict[str, SceneObject], moved: dict | None = None) -> np.ndarray:
        q = self.base.copy()
        for oid, o in state.items():
            a = self.adr[oid]
            pos, quat = (moved or {}).get(oid, (o.pos, o.quat))
            q[a:a + 3], q[a + 3:a + 7] = pos, quat
        return q

    def margin(self, q: np.ndarray) -> float:
        return float(np.minimum(q - self.q_lo, self.q_hi - q).min())

    def start_problem(self, state: dict[str, SceneObject], q_start: np.ndarray) -> Problem | None:
        self.probe.chk.set_fixed_qpos(self.qpos(state))
        self.probe._rules("free")
        rep = self.probe.chk.check(np.asarray(q_start, float))
        if rep.valid:
            return None
        return Problem("start_blocked", "An object is too close to the robot's rest position.",
                       f"Move objects away from the arm ({rep.reason}).")

    def lift_height(self, o: SceneObject, target_z: float, state: dict[str, SceneObject]) -> float:
        half_h = 0.5 * float(o.size[2])
        tops = [p.top for p in state.values() if p.id != o.id]
        need = (max(tops) if tops else self.world.table_height) + self.clearance - min(o.bottom, target_z - half_h)
        return float(np.clip(need, self.min_lift, self.max_lift))

    def check_leg(self, leg: Leg, state: dict[str, SceneObject]) -> tuple[bool, str]:
        """Fill the leg's grasp parameters, or its problems. Returns (needs_split, reason)."""
        o = state[leg.object_id]
        r = self.rules
        depth = r.depth(float(o.size[2]))
        half_h = 0.5 * float(o.size[2])
        z_pick = o.top - depth
        z_place = leg.target[2] + half_h - depth
        lift = self.lift_height(o, leg.target[2], state)
        tq = quat_from_yaw(leg.yaw)
        q_pick_scene = self.qpos(state)
        q_place_scene = self.qpos(state, {o.id: (leg.target, tq)})
        q_above_scene = self.qpos(state, {o.id: ([leg.target[0], leg.target[1], leg.target[2] + lift], tq)})
        reasons, candidates = [], []
        for opt in r.grasp_options(o.size, o.yaw):
            q_pick, why = self.probe.top_down(o.pos[:2], z_pick, opt.yaw, qpos=q_pick_scene, obj=o.id,
                                              supports=tuple(leg.supports_from))
            if q_pick is None:
                reasons.append(("pick", why))
                continue
            q_place, why = self.probe.top_down(leg.target[:2], z_place, opt.yaw, qpos=q_place_scene,
                                               qpos_above=q_above_scene, obj=o.id, supports=tuple(leg.supports_to),
                                               holding=True, above_height=lift)
            if q_place is None:
                reasons.append(("place", why))
                continue
            candidates.append((min(self.margin(q_pick), self.margin(q_place)), opt, q_pick))
        if not candidates:
            picks = all(k == "pick" for k, _ in reasons)
            detail = reasons[0][1] if reasons else "no grip orientation fits"
            names = {p.id: p.name for p in state.values()}
            if picks:
                leg.problems.append(Problem("pick_unreachable", f"The arm cannot pick up {o.name} where it stands.",
                                            _friendly(detail, names), o.id))
            else:
                leg.problems.append(Problem("place_unreachable", f"The arm cannot put {o.name} down there.",
                                            _friendly(detail, names), o.id))
            return False, ""
        # Comfortable joint margins first, then the narrower side (more room for the open fingers), then margin.
        candidates.sort(key=lambda c: (c[0] < 0.2, round(c[1].width, 3), -c[0]))
        arc_reason = ""
        for _, opt, q_pick in candidates:
            moved = {o.id: ([o.pos[0], o.pos[1], o.pos[2] + lift], o.quat)}
            q_lift_scene = self.qpos(state, moved)
            q_lift, why = self.probe.solve(np.r_[o.pos[0], o.pos[1], z_pick + lift], opt.yaw, [q_pick],
                                           q_lift_scene, "carry", o.id)
            if q_lift is None:
                arc_reason = f"lifting: {why}"
                continue
            ok, why = self.probe.carry_arc(q_lift, np.r_[leg.target[0], leg.target[1], z_place + lift], opt.yaw,
                                           q_lift_scene, o.id, self.kin_limits)
            if ok:
                leg.grasp_yaw, leg.grasp_width, leg.depth, leg.lift = float(opt.yaw), float(opt.width), depth, lift
                return False, ""
            arc_reason = _friendly(why, {p.id: p.name for p in state.values()})
        return True, arc_reason

    def pickable(self, o: SceneObject, state: dict[str, SceneObject]) -> bool:
        """Can the open gripper reach around `o` where it stands (any grip orientation)?"""
        z = o.top - self.rules.depth(float(o.size[2]))
        q_scene = self.qpos(state)
        supports = tuple(supports_of(o, list(state.values()), self.world.table_height))
        for opt in self.rules.grasp_options(o.size, o.yaw):
            if self.probe.top_down(o.pos[:2], z, opt.yaw, qpos=q_scene, obj=o.id, supports=supports)[0] is not None:
                return True
        return False


_BODY_WORDS = {"left_finger": "a finger", "right_finger": "a finger", "hand": "the hand", "link0": "the robot base",
               "table": "the table"}


def _friendly(reason: str, names: dict[str, str] | None = None) -> str:
    """A checker or planner reason in words: body ids become object names."""
    names = names or {}

    def word(body: str) -> str:
        body = body.strip()
        if body in names:
            return names[body]
        if body.startswith("link"):
            return "the arm"
        return _BODY_WORDS.get(body, body)

    if "within clearance" in reason or "collision" in reason:
        a, _, b = reason.split(":")[-1].strip().partition("-")
        return f"Not enough room: {word(a)} would come too close to {word(b)}. Leave more space around the object."
    if "out of reach" in reason:
        return "The spot is outside what the arm can reach with a top-down grip."
    for oid, name in names.items():
        reason = reason.replace(oid, name)
    return reason


class TaskPlanner:
    def __init__(self, settings: StudioSettings, store: StudioStore, world: World, rules: RobotRules):
        self.settings, self.store, self.world, self.rules = settings, store, world, rules
        self.max_swing = np.radians(float(settings.studio["carry"]["max_swing_deg"]))

    # ------------------------------------------------------------------ helpers

    def _obj(self, state: dict, oid, role: str) -> SceneObject:
        if not oid or oid not in state:
            raise PlanError(f"Choose the {role}.")
        return state[oid]

    def _liftable(self, o: SceneObject, objs: list[SceneObject]) -> list[Problem]:
        out = self.rules.lift_problems(o.name, o.size, o.mass, o.id)
        st, notes = self.store.object_state(o, objs)
        if st in ("tipped", "fallen", "unsupported"):
            out.append(Problem(f"object_{st}", notes[0], "", o.id))
        carrying = [p.name for p in objs if o.id in supports_of(p, objs, self.world.table_height)]
        if carrying:
            out.append(Problem("supporting", f"{carrying[0]} is standing on {o.name}.", "Move it off first.", o.id))
        return out

    def _upright_at(self, o: SceneObject, x: float, y: float, z: float | None = None) -> SceneObject:
        c = copy.deepcopy(o)
        c.pos = [float(x), float(y), self.world.table_height + 0.5 * float(o.size[2]) if z is None else float(z)]
        c.quat = quat_from_yaw(o.yaw)
        return c

    def _leg(self, o: SceneObject, state: dict, target_xyz, purpose: str, label: str, supports_to=()) -> Leg:
        objs = list(state.values())
        return Leg(o.id, o.name, list(o.pos), list(o.quat), [float(v) for v in target_xyz], o.yaw,
                   supports_of(o, objs, self.world.table_height), list(supports_to), purpose, label)

    def _apply(self, state: dict, leg: Leg) -> None:
        o = state[leg.object_id]
        o.pos = list(leg.target)
        o.quat = quat_from_yaw(leg.yaw)

    def blocks_pick(self, x: SceneObject, y: SceneObject) -> bool:
        """Geometric test: would `x` (where it is) leave the open hand no room around `y` in every grip
        orientation? The palm (200 x 63 mm, 37 mm above the grip centre) counts if `x` is taller than its bottom,
        otherwise only the fingers (133 x 21 mm); 20 mm clearance either way."""
        if x.id == y.id or x.bottom > y.top + 0.12 or x.top < y.bottom:
            return False
        opts = self.rules.grasp_options(y.size, y.yaw)
        if not opts:
            return False
        palm_bottom = y.top - self.rules.depth(float(y.size[2])) + 0.037
        along, across = (HAND_ALONG, HAND_ACROSS) if x.top > palm_bottom - 0.005 else (FINGERS_ALONG, FINGERS_ACROSS)
        fp = object_footprint(x)
        for g in opts:
            hand = footprint(y.pos[0], y.pos[1], g.yaw, (2 * along, 2 * across, 0.0))
            if not polygons_overlap(fp, hand):
                return False
        return True

    def free_spot(self, state: dict, o: SceneObject, near_xy, reserved: list[np.ndarray] = (),
                  exclude: set[str] = frozenset(), keep_pickable: list[SceneObject] = (),
                  future: list[SceneObject] = ()) -> tuple[float, float] | None:
        """A free, in-zone spot for `o`, close to `near_xy`, not on any reserved footprint, leaving the objects in
        `keep_pickable` room for the gripper, and where `o` stays pickable next to the `future` placements."""
        w = self.world
        half_diag = 0.5 * float(np.hypot(o.size[0], o.size[1]))
        cands = []
        for r in np.arange(w.r_in + 0.02, w.r_out - 0.01, 0.035):
            for th in np.radians(np.arange(-w.theta_max_deg, w.theta_max_deg + 1e-6, 7.5)):
                cands.append((float(r * np.cos(th)), float(r * np.sin(th))))
        near = np.asarray(near_xy, float)
        cands.sort(key=lambda c: float(np.hypot(c[0] - near[0], c[1] - near[1])) + 0.15 * abs(np.arctan2(c[1], c[0])))
        others = [p for p in state.values() if p.id != o.id and p.id not in exclude]
        for x, y in cands:
            cand = self._upright_at(o, x, y)
            if self.store.placement_problems(cand, others):
                continue
            fp = object_footprint(cand, 0.5 * self.store.gap + 0.01)
            if any(polygons_overlap(fp, r_) for r_ in reserved):
                continue
            if np.hypot(x, y) + half_diag > w.table_radius:
                continue
            if any(self.blocks_pick(cand, k) for k in keep_pickable if k.id != o.id):
                continue
            if any(self.blocks_pick(f, cand) for f in future if f.id != o.id):
                continue
            return x, y
        return None

    # ------------------------------------------------------------------ tasks -> legs

    def compose(self, task_id: str, params: dict, state: dict) -> tuple[list[Leg], list[Problem]]:
        objs = list(state.values())
        if task_id == "move":
            o = self._obj(state, params.get("object"), "object to move")
            target = params.get("target")
            if not target or len(target) < 2:
                raise PlanError("Click on the table to choose where it goes.")
            problems = self._liftable(o, objs)
            dest = self._upright_at(o, target[0], target[1])
            problems += self.store.placement_problems(dest, objs)
            if np.hypot(dest.pos[0] - o.pos[0], dest.pos[1] - o.pos[1]) < AT_TARGET and abs(dest.pos[2] - o.pos[2]) < AT_TARGET:
                problems.append(Problem("already_there", f"{o.name} is already there."))
            if problems:
                return [], problems
            return [self._leg(o, state, dest.pos, "move", f"Move {o.name}")], []

        if task_id == "hand_over":
            o = self._obj(state, params.get("object"), "object to bring")
            problems = self._liftable(o, objs)
            if problems:
                return [], problems
            spot = self._hand_over_spot(state, o)
            if spot is None:
                return [], [Problem("no_room", "The front edge of the table is full.",
                                    "Clear some space at the front of the table.")]
            dest = self._upright_at(o, *spot)
            if np.hypot(spot[0] - o.pos[0], spot[1] - o.pos[1]) < 0.03:
                return [], [Problem("already_there", f"{o.name} is already at the front.")]
            return [self._leg(o, state, dest.pos, "move", f"Bring {o.name} to the front")], []

        if task_id == "swap":
            a = self._obj(state, params.get("a"), "first object")
            b = self._obj(state, params.get("b"), "second object")
            if a.id == b.id:
                raise PlanError("Choose two different objects.")
            problems = self._liftable(a, objs) + self._liftable(b, objs)
            for o in (a, b):
                if abs(o.bottom - self.world.table_height) > 0.008:
                    problems.append(Problem("not_on_table", f"{o.name} must stand on the table to swap it.", "", o.id))
            a_at_b, b_at_a = self._upright_at(a, b.pos[0], b.pos[1]), self._upright_at(b, a.pos[0], a.pos[1])
            problems += self.store.placement_problems(a_at_b, objs, {b.id})
            problems += self.store.placement_problems(b_at_a, objs, {a.id})
            if problems:
                return [], problems
            return self.rearrange(state, {a.id: a_at_b.pos, b.id: b_at_a.pos}, labels={
                a.id: f"Move {a.name} to {b.name}'s place", b.id: f"Move {b.name} to {a.name}'s place"})

        if task_id == "stack":
            o = self._obj(state, params.get("object"), "object to put on top")
            base = self._obj(state, params.get("base"), "object underneath")
            if o.id == base.id:
                raise PlanError("Choose two different objects.")
            problems = self._liftable(o, objs)
            st, notes = self.store.object_state(base, objs)
            if st in ("tipped", "fallen", "unsupported"):
                problems.append(Problem(f"base_{st}", notes[0], "", base.id))
            if any(base.id in supports_of(p, objs, self.world.table_height) for p in objs):
                problems.append(Problem("base_occupied", f"Something is already standing on {base.name}.", "", base.id))
            if self.world.zone_problem(base.pos[0], base.pos[1], base.name):
                problems.append(Problem("base_unreachable", f"{base.name} is out of the arm's work area.", "", base.id))
            top_z = base.top + 0.5 * float(o.size[2])
            if top_z + 0.5 * float(o.size[2]) - self.world.table_height > 0.40:
                problems.append(Problem("stack_too_tall", "The stack would be taller than 400 mm."))
            area_o, area_b = float(o.size[0] * o.size[1]), float(base.size[0] * base.size[1])
            if area_o > 2.5 * area_b:
                problems.append(Problem("unstable", f"{o.name} is much larger than {base.name} and would not stand "
                                                    f"safely on it.", "Stack the smaller object on the larger one."))
            if problems:
                return [], problems
            return [self._leg(o, state, [base.pos[0], base.pos[1], top_z], "move", f"Put {o.name} on {base.name}",
                              supports_to=[base.id])], []

        if task_id in ("line_up", "sort_weight"):
            ids = [i for i in (params.get("objects") or []) if i in state]
            if params.get("objects") and not ids:
                raise PlanError("Choose the objects to line up.")
            pool = [state[i] for i in ids] if ids else [o for o in objs if not self._liftable(o, objs)
                                                         and abs(o.bottom - self.world.table_height) <= 0.008]
            problems = []
            for o in pool:
                problems += self._liftable(o, objs)
                if abs(o.bottom - self.world.table_height) > 0.008:
                    problems.append(Problem("not_on_table", f"{o.name} must stand on the table.", "", o.id))
            if len(pool) < 2:
                problems.append(Problem("too_few", "A row needs at least two objects that the arm can lift."))
            if problems:
                return [], problems
            order = "weight" if task_id == "sort_weight" else (params.get("order") or "current")
            key = {"weight": lambda o: (o.mass, o.name), "name": lambda o: (o.name.lower(), o.pos[1]),
                   "height": lambda o: (o.size[2], o.name), "current": lambda o: o.pos[1]}.get(order)
            if key is None:
                raise PlanError(f"Unknown order '{order}'.")
            pool.sort(key=key)
            slots = self._row_slots(state, pool)
            if slots is None:
                return [], [Problem("no_row", "There is no free strip across the table for the row.",
                                    "Move other objects away from the front of the table, or choose fewer objects.")]
            verb = "Sort" if task_id == "sort_weight" else "Line up"
            return self.rearrange(state, {o.id: s for o, s in zip(pool, slots)},
                                  labels={o.id: f"{verb}: {o.name} to place {k + 1}" for k, o in enumerate(pool)})
        raise PlanError(f"Unknown task '{task_id}'.")

    def _hand_over_spot(self, state: dict, o: SceneObject):
        w = self.world
        half_diag = 0.5 * float(np.hypot(o.size[0], o.size[1]))
        others = [p for p in state.values() if p.id != o.id]
        for dth in (0, 8, -8, 16, -16, 24, -24, 32, -32):
            for dr in (0.0, 0.04, 0.08, 0.12):
                r = min(w.r_out - 0.01, w.table_radius - half_diag - 0.01) - dr
                th = np.radians(dth)
                cand = self._upright_at(o, r * np.cos(th), r * np.sin(th))
                if not self.store.placement_problems(cand, others):
                    return cand.pos[0], cand.pos[1]
        return None

    def _hand_reach_y(self, o: SceneObject) -> float:
        """Half-extent along world y of the open hand gripping `o` (palm and clearance), for the grip orientation
        that keeps it narrowest along the row."""
        opts = self.rules.grasp_options(o.size, o.yaw)
        if not opts:
            return 0.0
        return min(HAND_ALONG * abs(np.sin(g.yaw)) + HAND_ACROSS * abs(np.cos(g.yaw)) for g in opts)

    def _row_slots(self, state: dict, pool: list[SceneObject]):
        """Slot centres of a row along y (left = -y as seen from the front), at the first x that is free. Neighbours
        are spaced so that the open hand still fits around every object of the row."""
        ext_y = [abs(0.5 * o.size[0] * np.sin(o.yaw)) + abs(0.5 * o.size[1] * np.cos(o.yaw)) for o in pool]
        reach = [self._hand_reach_y(o) for o in pool]
        spacing = [max(ext_y[i] + ext_y[i + 1] + ROW_GAP, reach[i] + ext_y[i + 1], ext_y[i] + reach[i + 1])
                   for i in range(len(pool) - 1)]
        offsets = np.r_[0.0, np.cumsum(spacing)]
        offsets -= 0.5 * ((offsets[0] - ext_y[0]) + (offsets[-1] + ext_y[-1]))  # centre the row's outer faces on y = 0
        members = {o.id for o in pool}
        others = [p for p in state.values() if p.id not in members]
        # First look for a strip that is completely free (no object has to wait or be set aside), then for one
        # that only the row's own objects stand on.
        for blocking in (list(state.values()), others):
            for x in (0.50, 0.58, 0.42, 0.65, 0.38, 0.70):
                slots = [[x, float(y), self.world.table_height + 0.5 * float(o.size[2])] for o, y in zip(pool, offsets)]
                ok = True
                for o, s in zip(pool, slots):
                    cand = self._upright_at(o, s[0], s[1])
                    rest = [p for p in blocking if p.id != o.id]
                    if self.store.placement_problems(cand, rest) or any(
                            self.blocks_pick(cand, p) for p in rest if p.id not in members):
                        ok = False
                        break
                if ok:
                    return slots
        return None

    def rearrange(self, state: dict, targets: dict[str, list[float]], labels: dict[str, str]):
        """Order moves so that no object is put down where another one still stands; break cycles with a
        temporary spot. Objects already at their target are left alone."""
        legs: list[Leg] = []
        pending = {oid: t for oid, t in targets.items()
                   if np.hypot(t[0] - state[oid].pos[0], t[1] - state[oid].pos[1]) > AT_TARGET}
        reserved = {oid: object_footprint(self._upright_at(state[oid], t[0], t[1]), 0.5 * self.store.gap)
                    for oid, t in targets.items()}
        def obstacles(oid):
            """Objects standing on oid's target, and pending objects whose pick oid's placement would block."""
            dest = self._upright_at(state[oid], pending[oid][0], pending[oid][1])
            over = [p for p in state.values() if self.store.overlaps(dest, p)]
            blocked = [state[y] for y in pending if y != oid and self.blocks_pick(dest, state[y])]
            return over, blocked

        for _ in range(4 * len(targets) + 4):
            if not pending:
                return legs, []
            progressed = False
            for oid in list(pending):
                over, blocked = obstacles(oid)
                if not over and not blocked:
                    o = state[oid]
                    leg = self._leg(o, state, self._upright_at(o, *pending[oid][:2]).pos, "move",
                                    labels.get(oid, f"Move {o.name}"))
                    legs.append(leg)
                    self._apply(state, leg)
                    del pending[oid]
                    progressed = True
            if progressed:
                continue
            # Nothing can go yet: set one object aside (one that stands in the way, or one whose pick would be
            # blocked) on a spot that blocks nobody else.
            for oid in list(pending):
                over, blocked = obstacles(oid)
                fixed = [p for p in over if p.id not in pending]
                if fixed:
                    return [], [Problem("target_blocked", f"{fixed[0].name} is in the way of {state[oid].name}.",
                                        "Move it aside first.", fixed[0].id)]
                b = (over or blocked)[0]
                keep = [state[y] for y in pending if y != b.id]
                future = [self._upright_at(state[y], *pending[y][:2]) for y in pending if y != b.id]
                spot = self.free_spot(state, b, b.pos[:2], [fp for k, fp in reserved.items() if k != b.id],
                                      keep_pickable=keep, future=future)
                if spot is None:
                    return [], [Problem("no_room", "There is no free spot to set an object down in between.",
                                        "Make some room on the table.")]
                leg = self._leg(b, state, self._upright_at(b, *spot).pos, "buffer",
                                f"Set {b.name} aside for a moment")
                legs.append(leg)
                self._apply(state, leg)
                break
        return [], [Problem("no_order", "Could not find an order for these moves.")]

    # ------------------------------------------------------------------ splitting and checking

    def _swing(self, leg: Leg) -> float:
        a0 = np.arctan2(leg.start_pos[1], leg.start_pos[0])
        a1 = np.arctan2(leg.target[1], leg.target[0])
        return float(a1 - a0)  # front path: azimuths in (-pi, pi], never across the back

    def _via_spots(self, leg: Leg, state: dict, parts: int):
        o = state[leg.object_id]
        a0 = np.arctan2(leg.start_pos[1], leg.start_pos[0])
        sweep = self._swing(leg)
        r0, r1 = np.hypot(*leg.start_pos[:2]), np.hypot(*leg.target[:2])
        w = self.world
        r_mid = float(np.clip(0.5 * (r0 + r1), w.r_in + 0.08, w.r_out - 0.08))
        spots = []
        for k in range(1, parts):
            th = a0 + sweep * k / parts
            found = None
            for dr in (0.0, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12):
                for dth in (0.0, 5.0, -5.0, 10.0, -10.0):
                    r, t = r_mid + dr, th + np.radians(dth)
                    cand = self._upright_at(o, r * np.cos(t), r * np.sin(t))
                    others = [p for p in state.values() if p.id != o.id]
                    if not self.store.placement_problems(cand, others):
                        found = cand.pos
                        break
                if found:
                    break
            if found is None:
                return None
            spots.append(found)
        return spots

    def _split(self, leg: Leg, state: dict, parts: int) -> list[Leg] | None:
        spots = self._via_spots(leg, state, parts)
        if spots is None:
            return None
        out, o = [], copy.deepcopy(state[leg.object_id])
        stops = [*spots, leg.target]
        for k, s in enumerate(stops):
            last = k == len(stops) - 1
            sub = Leg(o.id, o.name, list(o.pos), list(o.quat), list(s), leg.yaw,
                      leg.supports_from if k == 0 else [], leg.supports_to if last else [],
                      leg.purpose if last else "via",
                      leg.label if last else f"{leg.label}: set it down halfway to turn the grip")
            out.append(sub)
            o.pos, o.quat = list(s), quat_from_yaw(leg.yaw)
        return out

    def plan(self, task_id: str, params: dict, q_start=None) -> TaskPlan:
        t0 = time.perf_counter()
        meta = TASK_IDS.get(task_id)
        if meta is None:
            raise PlanError(f"Unknown task '{task_id}'.")
        snap = self.store.snapshot()
        state = {o.id: o for o in snap["objects"]}
        work = copy.deepcopy(state)
        legs, problems = self.compose(task_id, params or {}, work)
        plan = TaskPlan(task_id, meta["title"], self._summary(meta, params or {}, state), dict(params or {}),
                        [], problems, snap["revision"])
        if problems or not legs:
            if not problems:
                plan.problems.append(Problem("nothing_to_do", "Nothing to do: everything is already in place."))
            plan.check_s = time.perf_counter() - t0
            return plan
        feas = Feasibility(self.settings, self.world, self.rules, list(state.values()))
        sim = copy.deepcopy(state)
        start = feas.start_problem(sim, self.settings.home if q_start is None else q_start)
        if start is not None:
            plan.problems.append(start)
        queue = list(legs)
        while queue:
            leg = queue.pop(0)
            leg.start_pos, leg.start_quat = list(sim[leg.object_id].pos), list(sim[leg.object_id].quat)
            leg.supports_from = supports_of(sim[leg.object_id], list(sim.values()), self.world.table_height)
            parts = int(np.ceil(abs(self._swing(leg)) / self.max_swing - 1e-9))
            if parts > 1 and leg.purpose != "via":
                split = self._split(leg, sim, parts)
                if split is None:
                    leg.problems.append(Problem("no_room", f"No free spot to set {leg.object_name} down halfway.",
                                                "The arm turns its grip on long moves around the base."))
                    plan.legs.append(leg)
                    break
                queue = split + queue
                continue
            needs_split, why = feas.check_leg(leg, sim)
            if needs_split and leg.purpose != "via" and abs(self._swing(leg)) > np.radians(30):
                split = self._split(leg, sim, 2)
                if split is not None:
                    queue = split + queue
                    continue
            if needs_split:
                leg.problems.append(Problem("carry_blocked", f"The arm cannot carry {leg.object_name} there.",
                                            why, leg.object_id))
            if not leg.problems and leg.purpose == "move":
                # Every nearby object must stay pickable (also those picked later in this task, from where they
                # stand now); the object it is stacked on does not count.
                self._check_neighbours(feas, leg, sim, set(leg.supports_to))
            plan.legs.append(leg)
            if leg.problems:
                break
            self._apply(sim, leg)
        plan.check_s = time.perf_counter() - t0
        return plan

    def _check_neighbours(self, feas: Feasibility, leg: Leg, sim: dict, skip: set[str]) -> None:
        """A final placement must not leave a nearby object without room for the gripper's fingers."""
        after = copy.deepcopy(sim)
        self._apply(after, leg)
        for n in sim.values():
            if n.id == leg.object_id or n.id in skip:
                continue
            if np.hypot(n.pos[0] - leg.target[0], n.pos[1] - leg.target[1]) > 0.18:
                continue
            if self._liftable(n, list(sim.values())):
                continue
            if feas.pickable(n, sim) and not feas.pickable(after[n.id], after):
                leg.problems.append(Problem("blocks_neighbour", f"Putting {leg.object_name} there would leave no room "
                                                                f"for the gripper to pick up {n.name} later.",
                                            "Choose a spot a little further away from it.", n.id))
                return

    def _summary(self, meta: dict, params: dict, state: dict) -> str:
        name = lambda k: state[params[k]].name if params.get(k) in state else "?"  # noqa: E731
        tid = meta["id"]
        if tid == "move":
            return f"Move {name('object')} to the chosen spot"
        if tid == "hand_over":
            return f"Bring {name('object')} to the front"
        if tid == "swap":
            return f"Swap {name('a')} and {name('b')}"
        if tid == "stack":
            return f"Put {name('object')} on {name('base')}"
        if tid == "sort_weight":
            return "Sort objects by weight"
        return "Line objects up"
