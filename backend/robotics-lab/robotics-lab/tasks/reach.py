"""Reach a sequence of Cartesian end-effector targets (x, y, z).

The task resolves each target to a joint configuration through an injected
planner (inverse kinematics + collision checks, see simulation/kinematics.py),
then produces the same smooth joint references as JointTargetTask (with the
same optional feasibility-checked time scaling): move, hold, next target. Targets that cannot be reached
(out of reach, IK failure, collision at the goal or along the path) are
skipped and recorded with the reason; the arm never moves towards them.
The task produces references only; it never moves the robot itself.

With a motion planner (v0.5, planning/motion_planner.py) the plan result also
carries a collision-checked joint path. Its intermediate vertices become via
points: the reference stops at each (rest to rest along every straight,
checked edge) without holding, then holds at the target as before.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tasks.joint_target import JointTargetTask, Reference


@dataclass
class TargetPlan:
    name: str
    position: np.ndarray        # requested end-effector position (m, world frame)
    reachable: bool
    q: np.ndarray | None        # joint solution if reachable
    ik_error: float             # IK end-effector error (m)
    reason: str = ""
    waypoint_index: int | None = None  # index in the joint waypoints if reachable
    status: str = ""                    # planner status code (v0.5), "" for IK-only planning
    path: list | None = None            # planned joint path start ... q (v0.5), None for IK-only
    raw_path: list | None = None        # planner output before shortcutting
    stats: dict | None = None           # planner statistics


class CartesianReachTask:
    def __init__(
        self,
        targets: list[tuple[str, np.ndarray]],
        start_q: np.ndarray,
        plan: Callable,  # plan(target_xyz, q_from) -> IKResult-like (q, success, error, reason)
        move_time: float = 3.0,
        hold_time: float = 1.0,
        joint_limits: np.ndarray | None = None,
        path_move_time: float | None = None,  # requested time per planned-path edge (default move_time)
        **timing,  # profile, limits, torque_fn, torque_limits: see JointTargetTask
    ):
        if not targets:
            raise ValueError("Need at least one Cartesian target")
        self.plans: list[TargetPlan] = []
        waypoints = [np.asarray(start_q, dtype=float)]
        moves, holds = [], []
        # End-effector goal shown for each joint waypoint (index 0 = start; via points show their target).
        self.waypoint_targets: list[np.ndarray | None] = [None]
        edge_time = move_time if path_move_time is None else path_move_time
        for name, pos in targets:
            pos = np.asarray(pos, dtype=float)
            if pos.shape != (3,):
                raise ValueError(f"Target '{name}' must be [x, y, z], got {pos.tolist()}")
            res = plan(pos, waypoints[-1])
            path = getattr(res, "path", None) or None
            tp = TargetPlan(name, pos, bool(res.success), res.q if res.success else None, res.error, res.reason,
                            status=getattr(res, "status", ""), stats=getattr(res, "stats", None))
            if res.success:
                detour = path is not None and len(path) > 2  # a free direct edge keeps move_time
                if path is not None:
                    if not np.allclose(path[0], waypoints[-1]) or not np.allclose(path[-1], res.q):
                        raise ValueError(f"planned path for '{name}' does not run from the current waypoint to its goal")
                    tp.path, tp.raw_path = [np.asarray(q, float) for q in path], getattr(res, "raw_path", None)
                    for q in path[1:-1]:  # via points: stop, no hold
                        waypoints.append(np.asarray(q, dtype=float))
                        moves.append(edge_time)
                        holds.append(0.0)
                        self.waypoint_targets.append(pos)
                tp.waypoint_index = len(waypoints)
                waypoints.append(np.asarray(res.q, dtype=float))
                moves.append(edge_time if detour else move_time)
                holds.append(hold_time)
                self.waypoint_targets.append(pos)
            self.plans.append(tp)
        if len(waypoints) == 1:  # nothing reachable: hold the start configuration
            waypoints.append(waypoints[0].copy())
            moves.append(move_time)
            holds.append(hold_time)
            self.waypoint_targets.append(None)
        self.joint_task = JointTargetTask(waypoints, move_time, hold_time, joint_limits, loop=False,
                                          move_times=moves, hold_times=holds, **timing)
        self.waypoints = self.joint_task.waypoints
        self.move_time, self.hold_time = move_time, hold_time

    @property
    def reachable(self) -> list[TargetPlan]:
        return [p for p in self.plans if p.reachable]

    @property
    def rejected(self) -> list[TargetPlan]:
        return [p for p in self.plans if not p.reachable]

    @property
    def planned_paths(self) -> list[tuple[list, list]]:
        """(raw RRT path, executed path) of every target reached through the motion planner."""
        return [(p.raw_path or p.path, p.path) for p in self.reachable if p.path is not None]

    @property
    def segment_time(self) -> float:
        return self.joint_task.segment_time

    @property
    def num_segments(self) -> int:
        return self.joint_task.num_segments

    @property
    def segments(self):
        return self.joint_task.segments

    def segment_bounds(self, k: int) -> tuple[float, float, float]:
        return self.joint_task.segment_bounds(k)

    def timing_report(self) -> list[dict]:
        return self.joint_task.timing_report()

    @property
    def duration(self) -> float:
        return self.joint_task.duration

    def hold_end_time(self, plan: TargetPlan) -> float:
        """Time at the end of the hold at this target (end of its segment)."""
        return self.joint_task.segment_bounds(plan.waypoint_index - 1)[2]

    def reference(self, t: float) -> Reference:
        return self.joint_task.reference(t)
