"""Motion-planning facade used by the tasks: joint goal or Cartesian goal -> valid joint path.

    plan_joint(q_start, q_goal)        RRT-Connect + shortcutting
    plan_to_position(xyz, q_from)      IK candidates (several postures) + the above, per candidate

Both return a MotionPlan. It has the fields of an IKResult (q, success, error,
reason) so CartesianReachTask can use it in place of the IK-only planner, plus
the joint path (`path`, start ... goal, every edge collision-checked), the raw
RRT path before shortcutting, a status code and statistics.

Planning runs once, when a task is built; the controllers only ever see the
resulting time-parameterized joint reference.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from planning.collision import CollisionChecker
from planning.ik import ik_candidates
from planning.rrt_connect import PlanStatus, RRTConnect, RRTSettings
from planning.shortcut import shortcut
from simulation.kinematics import ArmKinematics

# Status codes for Cartesian queries in addition to PlanStatus.
UNREACHABLE = "unreachable"
NO_VALID_IK = "no_valid_ik"


@dataclass
class PlanningSettings:
    seed: int = 0
    clearance: float = 0.02            # m kept from the environment (robot-environment pairs)
    cartesian_step: float = 0.01       # m: max motion of any robot point between edge samples
    max_joint_step: float = 0.05       # rad: max joint motion between edge samples
    joint_limit_margin: float = 0.05   # rad kept from the joint limits
    exact_box_box: bool = True         # box-box robot-environment pairs via mj_geomDistance (v0.5.2)
    rrt: RRTSettings = field(default_factory=RRTSettings)
    shortcut: bool = True
    shortcut_iterations: int = 100
    ik_random_seeds: int = 12
    ik_local_seeds: int = 8            # IK seeds perturbed around the current configuration
    ik_max_candidates: int = 4         # IK postures attempted per Cartesian target
    ik_tolerance: float = 1e-4
    ik_time_split: bool = True         # each IK posture gets a fair share of the remaining time

    @classmethod
    def from_config(cls, cfg: dict | None) -> "PlanningSettings":
        """From a config section like configs/planning_demo.yaml `task.planning`."""
        cfg = dict(cfg or {})
        cfg.pop("enabled", None)
        cfg.pop("segment_time", None)
        rrt = RRTSettings(**(cfg.pop("rrt", None) or {}))
        sc = cfg.pop("shortcut", None) or {}
        ik = cfg.pop("ik", None) or {}
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown planning settings: {sorted(unknown)}")
        return cls(rrt=rrt, shortcut=sc.get("enabled", True), shortcut_iterations=sc.get("iterations", 100),
                   ik_random_seeds=ik.get("random_seeds", 12), ik_local_seeds=ik.get("local_seeds", 8),
                   ik_max_candidates=ik.get("max_candidates", 4),
                   ik_tolerance=ik.get("tolerance", 1e-4), ik_time_split=ik.get("time_split", True), **cfg)


@dataclass
class MotionPlan:
    success: bool
    status: str                         # PlanStatus value, "unreachable" or "no_valid_ik"
    reason: str = ""
    q: np.ndarray | None = None         # goal configuration
    error: float = 0.0                  # IK position error at the goal (m), 0 for joint goals
    path: list[np.ndarray] = field(default_factory=list)      # executed path (shortcut if enabled)
    raw_path: list[np.ndarray] = field(default_factory=list)  # RRT output
    stats: dict = field(default_factory=dict)


class MotionPlanner:
    def __init__(self, checker: CollisionChecker, kinematics: ArmKinematics | None = None,
                 settings: PlanningSettings | None = None):
        self.checker = checker
        self.kin = kinematics
        self.settings = settings or PlanningSettings()
        self.rrt = RRTConnect(checker, self.settings.rrt)
        self._queries = 0

    @classmethod
    def from_engine(cls, engine, settings: PlanningSettings | None = None, kinematics: ArmKinematics | None = None):
        s = settings or PlanningSettings()
        checker = CollisionChecker.from_engine(engine, clearance=s.clearance, cartesian_step=s.cartesian_step,
                                               max_joint_step=s.max_joint_step, limit_margin=s.joint_limit_margin,
                                               exact_box_box=s.exact_box_box)
        return cls(checker, kinematics or ArmKinematics(engine, limit_margin=s.joint_limit_margin), s)

    def _next_seed(self) -> int:
        """Deterministic per-query seed: the n-th query of a planner always gets the same seed."""
        self._queries += 1
        return self.settings.seed + 1000 * (self._queries - 1)

    def _finish_path(self, raw: list[np.ndarray], seed: int, stats: dict) -> list[np.ndarray]:
        stats["raw_length_rad"] = stats.get("path_length_rad", 0.0)
        stats["raw_vertices"] = len(raw)
        if self.settings.shortcut and len(raw) > 2:
            t0 = time.perf_counter()
            # Unweighted joint-space length: the metric that is reported and minimized.
            res = shortcut(raw, self.checker, self.settings.shortcut_iterations, seed)
            stats.update(smoothing_time_s=time.perf_counter() - t0, shortcuts_applied=res.shortcuts_applied,
                         smoothing_fell_back=res.used_original)
            path = res.path
        else:
            path = [q.copy() for q in raw]
        stats["smoothed_length_rad"] = self.rrt.path_length(path)
        stats["smoothed_vertices"] = len(path)
        return path

    def plan_joint(self, q_start: np.ndarray, q_goal: np.ndarray, seed: int | None = None) -> MotionPlan:
        seed = self._next_seed() if seed is None else seed
        t0, checks0 = time.perf_counter(), self.checker.config_checks
        res = self.rrt.plan(q_start, q_goal, seed)
        stats = dict(res.stats)
        if not res.success:
            stats["total_time_s"] = time.perf_counter() - t0
            return MotionPlan(False, res.status.value, res.reason, np.asarray(q_goal, float), stats=stats)
        path = self._finish_path(res.path, seed, stats)
        stats["collision_checks"] = self.checker.config_checks - checks0
        stats["total_time_s"] = time.perf_counter() - t0
        return MotionPlan(True, res.status.value, "", path[-1].copy(), 0.0, path, res.path, stats)

    def plan_to_position(self, target: np.ndarray, q_from: np.ndarray, seed: int | None = None,
                         rotation: np.ndarray | None = None, accept=None) -> MotionPlan:
        """Reach an end-effector position: try collision-free IK postures in order of
        joint distance from q_from until one is connected by a valid path. The time
        budget rrt.max_time covers IK, all RRT attempts and shortcutting. With `rotation`
        (3 x 3, world frame; v0.6) the goal is the full end-effector pose. `accept(q)` (optional,
        studio) rejects goal postures that do not suit what follows, e.g. a straight descent."""
        if self.kin is None:
            raise ValueError("plan_to_position needs ArmKinematics")
        s = self.settings
        seed = self._next_seed() if seed is None else seed
        t0, checks0 = time.perf_counter(), self.checker.config_checks
        stats = {"seed": seed}

        def done(plan: MotionPlan) -> MotionPlan:
            plan.stats = {**stats, **plan.stats, "total_time_s": time.perf_counter() - t0,
                          "collision_checks": self.checker.config_checks - checks0}
            return plan

        rep = self.checker.check(q_from)
        if not rep.valid:
            return done(MotionPlan(False, PlanStatus.INVALID_START.value, f"start configuration invalid: {rep.reason}"))
        ik = ik_candidates(self.kin, self.checker, target, q_from, s.ik_random_seeds, s.ik_tolerance, seed,
                           self.rrt.weights, max_candidates=s.ik_max_candidates, local_seeds=s.ik_local_seeds,
                           rotation=rotation)
        stats.update(ik_time_s=time.perf_counter() - t0, ik_seeds=ik.seeds_tried, ik_converged=ik.converged,
                     ik_valid=len(ik.candidates), ik_best_error_m=ik.best_error)
        if not ik.reachable:
            return done(MotionPlan(False, UNREACHABLE, f"Cartesian target unreachable: IK did not converge from "
                                   f"{ik.seeds_tried} seeds (closest {1e3 * ik.best_error:.1f} mm)", error=ik.best_error))
        if not ik.candidates:
            return done(MotionPlan(False, NO_VALID_IK, f"no collision-free IK configuration: all {ik.converged} "
                                   f"solutions rejected (e.g. {ik.rejected[0]})", error=ik.best_error))
        candidates, errors = ik.candidates, ik.errors
        if accept is not None:
            kept = [(q, e) for q, e in zip(candidates, errors) if accept(q)]
            stats["ik_accepted"] = len(kept)
            if not kept:
                return done(MotionPlan(False, NO_VALID_IK, f"none of the {len(candidates)} collision-free IK "
                                       f"postures suits the next motion", error=ik.best_error))
            candidates, errors = [q for q, _ in kept], [e for _, e in kept]

        attempts, budget, n_cand = [], s.rrt.max_time, len(candidates)
        for i, (q_goal, err) in enumerate(zip(candidates, errors)):
            remaining = budget - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            # Fair share of the remaining time, so one hard posture cannot starve the others
            # (unused time carries over to the next posture).
            share = remaining / (n_cand - i) if s.ik_time_split else remaining
            self.rrt.settings = RRTSettings(**{**vars(s.rrt), "max_time": share})
            try:
                res = self.rrt.plan(q_from, q_goal, seed + i)
            finally:
                self.rrt.settings = s.rrt
            attempts.append(res)
            if res.success:
                pst = dict(res.stats)
                path = self._finish_path(res.path, seed + i, pst)
                pst.update(candidate_index=i, candidates_tried=len(attempts),
                           nodes_all_attempts=sum(a.stats["nodes"] for a in attempts),
                           planning_time_all_attempts_s=sum(a.stats["planning_time_s"] for a in attempts))
                return done(MotionPlan(True, PlanStatus.SUCCESS.value, "", path[-1].copy(), err, path, res.path, pst))
        last = attempts[-1] if attempts else None
        status = last.status.value if last else PlanStatus.TIMEOUT.value
        reasons = "; ".join(f"posture {k + 1}: {a.reason}" for k, a in enumerate(attempts)) or "no time left"
        pst = dict(last.stats) if last else {}
        pst.update(candidates_tried=len(attempts), nodes_all_attempts=sum(a.stats["nodes"] for a in attempts))
        return done(MotionPlan(False, status, f"no path to any of {len(candidates)} IK postures ({reasons})",
                               error=errors[0], stats=pst))
