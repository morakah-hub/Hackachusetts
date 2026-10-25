"""Joint waypoints connected by planned collision-free paths (v0.5).

Like JointTargetTask, but each move from one key waypoint to the next follows
the joint path returned by an injected planner (plan(q_a, q_b) -> MotionPlan,
see planning/motion_planner.py) instead of the straight line. Path vertices
become via points: the reference stops at each one (rest to rest along every
checked straight edge) without holding, and holds only at the key waypoints.
If any move cannot be planned the task cannot be built: PlanningFailed says
which move and why. The task produces references only.
"""

from collections.abc import Callable

import numpy as np

from tasks.joint_target import JointTargetTask, Reference


class PlanningFailed(ValueError):
    """A required move has no collision-free path (the message says which and why)."""


class PlannedJointTask:
    def __init__(
        self,
        waypoints: list[list[float]] | np.ndarray,
        plan: Callable,
        move_time: float = 3.0,
        hold_time: float = 1.0,
        joint_limits: np.ndarray | None = None,
        loop: bool = True,
        path_move_time: float | None = None,
        names: list[str] | None = None,
        **timing,
    ):
        keys = np.asarray(waypoints, dtype=float)
        if keys.ndim != 2 or len(keys) < 2:
            raise ValueError("Need at least two waypoints, each a list of joint angles")
        names = names or [f"waypoint{i}" for i in range(len(keys))]
        n = len(keys)
        edge_time = move_time if path_move_time is None else path_move_time
        self.plans = []
        expanded, moves, holds = [keys[0]], [], []
        self.key_index = [0]       # index in the expanded waypoints of each key waypoint
        self.goal_index = [0]      # for every expanded waypoint: the key waypoint it leads to
        for k in range(n if loop else n - 1):
            a, b = keys[k], keys[(k + 1) % n]
            res = plan(a, b)
            if not res.success:
                raise PlanningFailed(f"move {names[k]} -> {names[(k + 1) % n]}: {res.status}: {res.reason}")
            self.plans.append(res)
            target_key = (k + 1) % n
            # A free direct move keeps move_time (identical to JointTargetTask); detours use edge_time per edge.
            t_edge = edge_time if len(res.path) > 2 else move_time
            for q in res.path[1:-1]:
                expanded.append(np.asarray(q, float))
                moves.append(t_edge)
                holds.append(0.0)
                self.goal_index.append(None)
            if target_key != 0:
                expanded.append(b)
                self.key_index.append(len(expanded) - 1)
                self.goal_index.append(None)
            moves.append(t_edge)
            holds.append(hold_time)
        # Each via point leads to the next key waypoint (via points of a loop's closing move to waypoint 0).
        keys_set, nxt = set(self.key_index), 0
        for i in range(len(expanded) - 1, -1, -1):
            if i in keys_set:
                nxt = i
            self.goal_index[i] = nxt
        self.joint_task = JointTargetTask(expanded, move_time, hold_time, joint_limits, loop=loop,
                                          move_times=moves, hold_times=holds, **timing)
        self.waypoints = self.joint_task.waypoints
        self.key_waypoints = keys
        self.move_time, self.hold_time, self.loop = move_time, hold_time, loop

    @property
    def planned_paths(self) -> list[tuple[list, list]]:
        return [(p.raw_path or p.path, p.path) for p in self.plans]

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

    def reference(self, t: float) -> Reference:
        return self.joint_task.reference(t)
