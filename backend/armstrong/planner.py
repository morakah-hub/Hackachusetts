"""Pre-written tasks: catalog, decomposition into pick-and-place legs, and pre-flight checks.

Task composition (move, bring to me, swap, stack, line up, sort by weight) follows Workcell Studio's
planner; "Assemble" builds a small structure (a bridge of two pillars and a beam, or a three-block tower).
A task becomes an ordered list of legs, each one physical pick-and-place (task.PickPlaceLeg). Rearrangements
that would put an object where another one still stands use a temporary spot first.

The pre-flight check runs on a private copy of the scene: payload and gripper limits, free target spots, and
for every leg the grasp and placement poses (pose IK) and the collision checks of the approach, descent,
lift, carry arc and placement, with the same rules the executor uses. It cannot promise the physical outcome
(slip, contact), which only the simulation decides.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from . import scene as sc
from .kinematics import Kinematics
from .robot import HOME_Q
from .rules import Problem, RobotRules, World, grasp_depth, quat_from_yaw, wrap_angle
from .store import SceneObject, StudioStore, footprint, object_footprint, polygons_overlap, supports_of
from .task import APPROACH_HEIGHT, PLACE_CLEARANCE, Leg, closest_rep, hand_R
from .trajectory import JointMove, LineInfeasible, arc_carry, cartesian_line

TASKS = [
    {"id": "assemble", "title": "Assemble", "verb": "Assemble a structure",
     "summary": "Build a small structure from the assembly parts on the table: a bridge (two pillars and a beam "
                "laid across them) or a tower of three blocks.",
     "params": [{"key": "recipe", "type": "choice", "label": "What to build",
                 "choices": [["bridge", "Bridge"], ["tower", "Tower"]], "default": "bridge"},
                {"key": "target", "type": "point", "label": "Build spot (optional)", "optional": True}]},
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
AT_TARGET = 0.005
HAND_ALONG, HAND_ACROSS = 0.13, 0.06          # open hand half-extents (with clearance) around the grip centre
FINGERS_ALONG, FINGERS_ACROSS = 0.092, 0.036
ROW_GAP = 0.03
LEG_SECONDS = 15.0
CARRY_CLEARANCE, MIN_LIFT, MAX_LIFT = 0.04, 0.10, 0.25
BUILD_SPOT = (0.50, 0.0)
BRIDGE_HALF_SPAN = 0.05                        # pillar centres +-50 mm from the bridge centre


class PlanError(Exception):
    def __init__(self, message: str, problems: list[Problem] | None = None):
        super().__init__(message)
        self.message, self.problems = message, problems or []


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
                "estimated_s": round(LEG_SECONDS * len(self.legs), 0)}


def scene_config(objects: list[SceneObject]) -> sc.SceneConfig:
    return sc.SceneConfig(objects=[sc.ObjectSpec(o.id, tuple(o.size), float(o.mass), o.color, tuple(o.pos),
                                                 tuple(o.quat)) for o in objects])


_BODY_WORDS = {"left_finger": "a finger", "right_finger": "a finger", "hand": "the hand", "link0": "the robot base",
               "table": "the table"}


def friendly(reason: str, names: dict[str, str]) -> str:
    """A checker reason in words: body ids become object names."""
    for oid, name in names.items():
        reason = reason.replace(oid, name)
    for body, word in _BODY_WORDS.items():
        reason = reason.replace(f"{body} would", f"{word} would")
    for i in range(1, 8):
        reason = reason.replace(f"link{i} would", "the arm would")
    return reason


class Feasibility:
    """Pose IK and collision checks of legs on a private copy of the scene (no rendering, any thread)."""

    def __init__(self, objects: list[SceneObject], rules: RobotRules):
        self.rules = rules
        cfg = scene_config(objects)
        self.model, self.ids = sc.build_model(cfg)
        data = mujoco.MjData(self.model)
        sc.set_initial_state(self.model, data, self.ids, cfg)
        self.base = data.qpos.copy()
        self.kin = Kinematics(self.model, self.ids)
        _, self.R_home = self.kin.fk(HOME_Q)

    def load(self, state: dict[str, SceneObject]) -> None:
        self.kin.load(self.base, HOME_Q)
        for o in state.values():
            if o.id in self.ids.objects:
                self.kin.set_object(o.id, o.pos, o.quat)

    def lift_height(self, o: SceneObject, target_z: float, state: dict[str, SceneObject]) -> float:
        half_h = 0.5 * float(o.size[2])
        tops = [p.top for p in state.values() if p.id != o.id]
        need = (max(tops) if tops else sc.TABLE_TOP_Z) + CARRY_CLEARANCE - min(o.bottom, target_z - half_h)
        return float(np.clip(need, MIN_LIFT, MAX_LIFT))

    def check_leg(self, leg: Leg, state: dict[str, SceneObject]) -> None:
        """Choose the grasp (closing axis and hand representation) or record why the leg cannot run."""
        o = state[leg.object_id]
        h = float(o.size[2])
        depth = grasp_depth(h)
        kin, ids = self.kin, self.ids
        body, qadr = ids.objects[o.id].body, ids.objects[o.id].qadr
        table = ids.table_body
        sup_from = [ids.objects[s].body for s in leg.supports_from if s in ids.objects] + [table]
        sup_to = [ids.objects[s].body for s in leg.supports_to if s in ids.objects] + [table]
        lift = self.lift_height(o, leg.target[2], state)
        names = {p.id: p.name for p in state.values()}
        p_grasp = np.array([o.pos[0], o.pos[1], o.top - depth])
        tgt = np.asarray(leg.target, float)
        p_place = np.array([tgt[0], tgt[1], tgt[2] + h / 2 - depth + PLACE_CLEARANCE])
        reasons: list[tuple[str, str]] = []
        best = None
        for opt in self.rules.grasp_options(o.size, o.yaw):
            # The object turns by the smallest rotation that gives its target heading (mod pi).
            turn = closest_rep(leg.yaw - o.yaw, 0.0, np.pi)
            a_pick, a_place = opt.angle, opt.angle + turn
            R_pick, R_place = hand_R(self.R_home, a_pick), hand_R(self.R_home, a_place)
            self.load(state)
            try:
                r_pre = kin.solve_multi(p_grasp + [0, 0, APPROACH_HEIGHT], R_pick, [HOME_Q], rest=HOME_Q)
                if not r_pre.ok:
                    reasons.append(("pick", "the arm cannot reach above it with this grip"))
                    continue
                r_g = kin.solve(p_grasp, R_pick, r_pre.q, rest=HOME_Q)
                if not r_g.ok:
                    reasons.append(("pick", "the arm cannot reach down to it with this grip"))
                    continue
                r_pp = kin.solve_multi(p_place + [0, 0, lift], R_place, [r_pre.q, HOME_Q], rest=HOME_Q)
                r_pl = kin.solve(p_place, R_place, r_pp.q, rest=HOME_Q) if r_pp.ok else r_pp
                if not (r_pp.ok and r_pl.ok):
                    reasons.append(("place", "the arm cannot put it down there with this grip"))
                    continue
                approach = JointMove(HOME_Q, r_pre.q, min_time=1.5)
                hit = kin.path_collision(approach.samples(30), held=body, held_ok=frozenset(sup_from))
                if hit:
                    reasons.append(("pick", hit))
                    continue
                down = cartesian_line(kin, r_pre.q, p_grasp, R_pick)
                hit = kin.path_collision(down.samples()[::2], held=body, held_ok=frozenset(sup_from))
                if hit:
                    reasons.append(("pick", hit))
                    continue
                q_g = down.end
                hp, hR = kin.hand_pose(q_g)
                carry = (hR.T @ (np.asarray(o.pos) - hp), hR.T @ _rotz(o.yaw))
                up = cartesian_line(kin, q_g, p_grasp + [0, 0, lift], R_pick)
                hit = kin.path_collision(up.samples()[::3], held=body, carry=carry, held_ok=frozenset(sup_from),
                                         held_qadr=qadr)
                if hit:
                    reasons.append(("lift", hit))
                    continue
                try:
                    move = arc_carry(kin, up.end, p_place + [0, 0, lift], a_pick, a_place, self.R_home)
                    qs = move.samples()[::3]
                except LineInfeasible:
                    move = JointMove(up.end, r_pp.q, min_time=1.5)
                    qs = move.samples(40)
                hit = kin.path_collision(qs, held=body, carry=carry, held_qadr=qadr)
                if hit:
                    reasons.append(("carry", hit))
                    continue
                lower = cartesian_line(kin, move.end, p_place, R_place)
                hit = kin.path_collision(lower.samples()[::2], held=body, carry=carry, held_ok=frozenset(sup_to),
                                         held_qadr=qadr)
                if hit:
                    reasons.append(("place", hit))
                    continue
            except LineInfeasible as exc:
                reasons.append(("pick", str(exc)))
                continue
            margin = min(kin.margin(r_g.q), kin.margin(lower.end))
            score = (margin < 0.25, round(opt.width, 3), -margin)
            if best is None or score < best[0]:
                best = (score, opt, a_pick, a_place, turn)
            if margin > 0.4:
                break
        if best is None:
            stage, why = reasons[0] if reasons else ("pick", "no grip orientation fits")
            text = friendly(why, names)
            if stage == "place":
                leg.problems.append(Problem("place_unreachable", f"The arm cannot put {o.name} down there.",
                                            text, o.id))
            elif stage == "carry":
                leg.problems.append(Problem("carry_blocked", f"The arm cannot carry {o.name} there.", text, o.id))
            else:
                leg.problems.append(Problem("pick_unreachable", f"The arm cannot pick up {o.name} where it stands.",
                                            text, o.id))
            return
        _, opt, a_pick, a_place, turn = best
        leg.grasp_rel, leg.grasp_pick, leg.grasp_place = opt.rel, a_pick, a_place
        leg.width, leg.lift = opt.width, lift
        leg.yaw = float(o.yaw + turn)


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class TaskPlanner:
    def __init__(self, store: StudioStore, world: World, rules: RobotRules):
        self.store, self.world, self.rules = store, world, rules

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
        carrying = [p.name for p in objs if o.id in supports_of(p, objs)]
        if carrying:
            out.append(Problem("supporting", f"{carrying[0]} is standing on {o.name}.", "Move it off first.", o.id))
        return out

    def _upright_at(self, o: SceneObject, x: float, y: float, z: float | None = None,
                    yaw: float | None = None) -> SceneObject:
        c = copy.deepcopy(o)
        c.pos = [float(x), float(y), self.world.table_height + 0.5 * float(o.size[2]) if z is None else float(z)]
        c.quat = quat_from_yaw(o.yaw if yaw is None else yaw)
        return c

    def _leg(self, o: SceneObject, state: dict, target_xyz, purpose: str, label: str, supports_to=(),
             yaw: float | None = None) -> Leg:
        objs = list(state.values())
        return Leg(o.id, o.name, list(o.size), o.color, list(o.pos), list(o.quat), [float(v) for v in target_xyz],
                   o.yaw if yaw is None else float(yaw), supports_of(o, objs), list(supports_to), purpose, label)

    def _apply(self, state: dict, leg: Leg) -> None:
        o = state[leg.object_id]
        o.pos = list(leg.target)
        o.quat = quat_from_yaw(leg.yaw)

    def blocks_pick(self, x: SceneObject, y: SceneObject) -> bool:
        """Would `x` (where it is) leave the open hand no room around `y` in every grip orientation?"""
        if x.id == y.id or x.bottom > y.top + 0.12 or x.top < y.bottom:
            return False
        opts = self.rules.grasp_options(y.size, y.yaw)
        if not opts:
            return False
        palm_bottom = y.top - grasp_depth(float(y.size[2])) + 0.037
        along, across = (HAND_ALONG, HAND_ACROSS) if x.top > palm_bottom - 0.005 else (FINGERS_ALONG, FINGERS_ACROSS)
        fp = object_footprint(x)
        for g in opts:
            hand = footprint(y.pos[0], y.pos[1], g.angle, (2 * along, 2 * across, 0.0))
            if not polygons_overlap(fp, hand):
                return False
        return True

    def free_spot(self, state: dict, o: SceneObject, near_xy, reserved=(), exclude=frozenset(),
                  keep_pickable=(), future=()) -> tuple[float, float] | None:
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
            # ... and it must leave the open hand room to put the other objects down at their targets.
            if any(self.blocks_pick(cand, f) for f in future if f.id != o.id):
                continue
            return x, y
        return None

    # ------------------------------------------------------------------ tasks -> legs

    def compose(self, task_id: str, params: dict, state: dict) -> tuple[list[Leg], list[Problem]]:
        objs = list(state.values())
        if task_id == "assemble":
            return self._assemble(params, state)

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
            problems = self._liftable(o, objs) + self._stack_problems(o, base, objs)
            if problems:
                return [], problems
            return [self._leg(o, state, [base.pos[0], base.pos[1], base.top + 0.5 * float(o.size[2])], "move",
                              f"Put {o.name} on {base.name}", supports_to=[base.id])], []

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

    def _stack_problems(self, o: SceneObject, base: SceneObject, objs) -> list[Problem]:
        problems = []
        st, notes = self.store.object_state(base, objs)
        if st in ("tipped", "fallen", "unsupported"):
            problems.append(Problem(f"base_{st}", notes[0], "", base.id))
        if any(base.id in supports_of(p, objs) for p in objs if p.id != o.id):
            problems.append(Problem("base_occupied", f"Something is already standing on {base.name}.", "", base.id))
        if self.world.zone_problem(base.pos[0], base.pos[1], base.name):
            problems.append(Problem("base_unreachable", f"{base.name} is out of the arm's work area.", "", base.id))
        if base.top + float(o.size[2]) - self.world.table_height > 0.30:
            problems.append(Problem("stack_too_tall", "The stack would be taller than 300 mm."))
        if float(o.size[0] * o.size[1]) > 2.5 * float(base.size[0] * base.size[1]):
            problems.append(Problem("unstable", f"{o.name} is much larger than {base.name} and would not stand "
                                                f"safely on it.", "Stack the smaller object on the larger one."))
        return problems

    # ------------------------------------------------------------------ assembly

    def _parts(self, state: dict, template: str) -> list[SceneObject]:
        objs = list(state.values())
        return [o for o in objs if o.template_id == template and not self._liftable(o, objs)]

    def _assemble(self, params: dict, state: dict) -> tuple[list[Leg], list[Problem]]:
        recipe = params.get("recipe") or "bridge"
        target = params.get("target")
        if target is not None and len(target) >= 2:
            bx, by = float(target[0]), float(target[1])
        else:
            bx, by = BUILD_SPOT
        zone = self.world.zone_problem(bx, by, "The build spot")
        if zone:
            return [], [zone]
        if recipe == "bridge":
            return self._bridge(state, bx, by)
        if recipe == "tower":
            return self._tower(state, bx, by)
        raise PlanError(f"Unknown structure '{recipe}'.")

    def _missing(self, what: str) -> Problem:
        return Problem("missing_parts", f"Missing parts: the {what} needs {self._kit_text(what)} on the table.",
                       "Press 'Assembly kit' in the left panel to lay out all parts, or add them from the library.")

    @staticmethod
    def _kit_text(what: str) -> str:
        return "two Pillars and a Beam" if what == "bridge" else "a Base block, a Middle block and a Top block"

    def _bridge(self, state: dict, bx: float, by: float):
        pillars = self._parts(state, "lib_pillar")
        beams = self._parts(state, "lib_beam")
        if len(pillars) < 2 or not beams:
            return [], [self._missing("bridge")]
        th = float(np.arctan2(by, bx))
        t = np.array([-np.sin(th), np.cos(th)])          # bridge axis: across the robot's line of sight
        spots = [np.array([bx, by]) - BRIDGE_HALF_SPAN * t, np.array([bx, by]) + BRIDGE_HALF_SPAN * t]
        # The two pillars closest to the spots, matched to minimise the total distance.
        best = None
        for i, a in enumerate(pillars):
            for j, b in enumerate(pillars):
                if i == j:
                    continue
                d = np.hypot(*(np.asarray(a.pos[:2]) - spots[0])) + np.hypot(*(np.asarray(b.pos[:2]) - spots[1]))
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, pa, pb = best
        beam = min(beams, key=lambda o: np.hypot(o.pos[0] - bx, o.pos[1] - by))
        yaw_axis = float(np.arctan2(t[1], t[0]))
        pillar_yaw = yaw_axis
        problems = []
        targets = {}
        for p, s in ((pa, spots[0]), (pb, spots[1])):
            dest = self._upright_at(p, s[0], s[1], yaw=pillar_yaw)
            others = [q for q in state.values() if q.id not in (pa.id, pb.id)]
            problems += self.store.placement_problems(dest, others)
            targets[p.id] = dest.pos
        # The beam's current spot must not be in the way of the bridge.
        foot = footprint(bx, by, yaw_axis, (0.20, 0.08, 0.0))
        blockers = [q for q in state.values() if q.id not in (pa.id, pb.id)
                    and polygons_overlap(object_footprint(q), foot)]
        legs: list[Leg] = []
        for q in blockers:
            if q.id == beam.id or not self._liftable(q, list(state.values())):
                spot = self.free_spot(state, q, q.pos[:2], reserved=[foot])
                if spot is None:
                    return [], [Problem("no_room", f"{q.name} is in the way of the bridge and there is no free spot "
                                                   f"to move it to.", "Clear the build spot first.", q.id)]
                leg = self._leg(q, state, self._upright_at(q, *spot).pos, "buffer", f"Clear the build spot: move {q.name}")
                legs.append(leg)
                self._apply(state, leg)
            else:
                return [], [Problem("target_blocked", f"{q.name} stands on the build spot and cannot be moved.",
                                    "Choose another build spot.", q.id)]
        moved = {q.id for q in blockers}
        problems = [p for p in problems if p.code != "overlap" or p.object_id not in moved]
        if problems:
            return [], problems
        more, problems = self.rearrange(state, {pa.id: targets[pa.id], pb.id: targets[pb.id]},
                                        labels={pa.id: f"Place {pa.name} (left side of the bridge)",
                                                pb.id: f"Place {pb.name} (right side of the bridge)"},
                                        yaws={pa.id: pillar_yaw, pb.id: pillar_yaw})
        if problems:
            return [], problems
        legs += more
        beam = state[beam.id]
        z = max(state[pa.id].top, state[pb.id].top) + 0.5 * float(beam.size[2])
        legs.append(self._leg(beam, state, [bx, by, z], "move", f"Lay {beam.name} across the pillars",
                              supports_to=[pa.id, pb.id], yaw=yaw_axis))
        return legs, []

    def _tower(self, state: dict, bx: float, by: float):
        names = ("lib_block_l", "lib_block_m", "lib_block_s")
        parts = [self._parts(state, n) for n in names]
        if not all(parts):
            return [], [self._missing("tower")]
        base, mid, top = (min(p, key=lambda o: np.hypot(o.pos[0] - bx, o.pos[1] - by)) for p in parts)
        legs: list[Leg] = []
        th = float(np.arctan2(by, bx))
        if np.hypot(base.pos[0] - bx, base.pos[1] - by) > AT_TARGET:
            dest = self._upright_at(base, bx, by, yaw=th)
            more, problems = self.rearrange(state, {base.id: dest.pos}, labels={base.id: f"Put {base.name} on the build spot"},
                                            yaws={base.id: th})
            if problems:
                return [], problems
            legs += more
        b = state[base.id]
        for upper, lower in ((mid, b), (top, None)):
            lower = lower or state[mid.id]
            upper = state[upper.id]
            problems = self._stack_problems(upper, lower, list(state.values()))
            if problems:
                return [], problems
            leg = self._leg(upper, state, [lower.pos[0], lower.pos[1], lower.top + 0.5 * float(upper.size[2])],
                            "move", f"Stack {upper.name} on {lower.name}", supports_to=[lower.id], yaw=lower.yaw)
            legs.append(leg)
            self._apply(state, leg)
        return legs, []

    # ------------------------------------------------------------------ rearranging

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
        opts = self.rules.grasp_options(o.size, o.yaw)
        if not opts:
            return 0.0
        return min(HAND_ALONG * abs(np.sin(g.angle)) + HAND_ACROSS * abs(np.cos(g.angle)) for g in opts)

    def _row_slots(self, state: dict, pool: list[SceneObject]):
        ext_y = [abs(0.5 * o.size[0] * np.sin(o.yaw)) + abs(0.5 * o.size[1] * np.cos(o.yaw)) for o in pool]
        reach = [self._hand_reach_y(o) for o in pool]
        spacing = [max(ext_y[i] + ext_y[i + 1] + ROW_GAP, reach[i] + ext_y[i + 1], ext_y[i] + reach[i + 1])
                   for i in range(len(pool) - 1)]
        offsets = np.r_[0.0, np.cumsum(spacing)]
        offsets -= 0.5 * ((offsets[0] - ext_y[0]) + (offsets[-1] + ext_y[-1]))
        members = {o.id for o in pool}
        others = [p for p in state.values() if p.id not in members]
        for blocking in (list(state.values()), others):
            for x in (0.50, 0.58, 0.42, 0.62, 0.38):
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

    def rearrange(self, state: dict, targets: dict[str, list[float]], labels: dict[str, str], yaws=None):
        """Order moves so that no object is put down where another one still stands; break cycles with a
        temporary spot. Objects already at their target are left alone."""
        yaws = yaws or {}
        legs: list[Leg] = []
        pending = {oid: t for oid, t in targets.items()
                   if np.hypot(t[0] - state[oid].pos[0], t[1] - state[oid].pos[1]) > AT_TARGET}
        reserved = {oid: object_footprint(self._upright_at(state[oid], t[0], t[1], yaw=yaws.get(oid)),
                                          0.5 * self.store.gap) for oid, t in targets.items()}

        def dest_of(oid):
            return self._upright_at(state[oid], pending[oid][0], pending[oid][1], yaw=yaws.get(oid))

        def obstacles(oid):
            dest = dest_of(oid)
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
                    leg = self._leg(o, state, dest_of(oid).pos, "move", labels.get(oid, f"Move {o.name}"),
                                    yaw=yaws.get(oid))
                    legs.append(leg)
                    self._apply(state, leg)
                    del pending[oid]
                    progressed = True
            if progressed:
                continue
            for oid in list(pending):
                over, blocked = obstacles(oid)
                fixed = [p for p in over if p.id not in pending]
                if fixed and not self._liftable(fixed[0], list(state.values())):
                    b = fixed[0]
                elif fixed:
                    return [], [Problem("target_blocked", f"{fixed[0].name} is in the way of {state[oid].name}.",
                                        "Move it aside first.", fixed[0].id)]
                else:
                    b = (over or blocked)[0]
                keep = [state[y] for y in pending if y != b.id]
                future = [dest_of(y) for y in pending if y != b.id]
                spot = self.free_spot(state, b, b.pos[:2], [fp for k, fp in reserved.items() if k != b.id],
                                      keep_pickable=keep, future=future)
                if spot is None:
                    return [], [Problem("no_room", "There is no free spot to set an object down in between.",
                                        "Make some room on the table.")]
                leg = self._leg(b, state, self._upright_at(b, *spot).pos, "buffer", f"Set {b.name} aside for a moment")
                legs.append(leg)
                self._apply(state, leg)
                break
        return [], [Problem("no_order", "Could not find an order for these moves.")]

    # ------------------------------------------------------------------ plan

    def plan(self, task_id: str, params: dict) -> TaskPlan:
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
        feas = Feasibility(list(state.values()), self.rules)
        sim = copy.deepcopy(state)
        for leg in legs:
            o = sim[leg.object_id]
            leg.start_pos, leg.start_quat = list(o.pos), list(o.quat)
            leg.supports_from = supports_of(o, list(sim.values()))
            feas.check_leg(leg, sim)
            plan.legs.append(leg)
            if leg.problems:
                break
            self._apply(sim, leg)
        plan.check_s = time.perf_counter() - t0
        return plan

    def _summary(self, meta: dict, params: dict, state: dict) -> str:
        name = lambda k: state[params[k]].name if params.get(k) in state else "?"  # noqa: E731
        tid = meta["id"]
        if tid == "assemble":
            return "Build a bridge from two pillars and a beam" if (params.get("recipe") or "bridge") == "bridge" \
                else "Build a tower of three blocks"
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


__all__ = ["TASKS", "TaskPlanner", "TaskPlan", "PlanError", "Feasibility", "scene_config", "wrap_angle"]
