"""Bidirectional RRT-Connect in the arm's 7-D joint space.

Kuffner & LaValle, "RRT-Connect: An Efficient Approach to Single-Query Path
Planning", ICRA 2000. Two trees grow from the start and the goal. Every
iteration samples a configuration uniformly inside the (margin-shrunk) joint
limits, EXTENDs tree A one step towards it, and, if that added a node, tries to
CONNECT tree B to the new node (repeated EXTEND steps until it reaches it or is
blocked). The trees then swap roles. A path is found when CONNECT reaches.

Distances use a weighted Euclidean metric d(a, b) = ||W (a - b)||, W = diag(w).
By default w_j is joint j's lever arm (CollisionChecker.lever_arms) relative to
joint 1, floored at 0.2: a radian of shoulder motion sweeps the arm much
further than a radian of wrist roll, so nearest neighbours are judged by how
much the arm moves rather than by raw angles. The extension step is measured in
this metric. Every new node is checked as a configuration and the edge from its
parent is checked with CollisionChecker.edge_valid (so the tree contains only
valid edges and the reconstructed path is valid edge by edge).

Budgets: wall-clock time, total nodes in both trees, and iterations. Each ends
the search with its own status. With a fixed seed and a run that ends on the
node or iteration budget (or succeeds), the result is deterministic; a time
budget depends on the machine.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from planning.collision import CollisionChecker


class PlanStatus(str, Enum):
    SUCCESS = "success"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    TIMEOUT = "timeout"
    NODE_BUDGET = "node_budget"
    NO_PATH = "no_path"  # iteration budget exhausted without connecting the trees


@dataclass
class RRTSettings:
    step_size: float = 0.3          # extension step in the weighted metric (~rad of shoulder motion)
    max_time: float = 5.0           # s, wall clock
    max_nodes: int = 20000          # both trees together
    max_iterations: int = 20000
    try_direct: bool = True         # return [start, goal] when that straight edge is valid
    joint_weights: tuple[float, ...] | None = None  # None: lever arms relative to joint 1


@dataclass
class RRTResult:
    status: PlanStatus
    path: list[np.ndarray]                       # start ... goal (empty unless success)
    reason: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == PlanStatus.SUCCESS


class _Tree:
    """Nodes in a preallocated array with parent indices."""

    def __init__(self, root: np.ndarray, capacity: int, weights: np.ndarray):
        self.q = np.empty((capacity, len(root)))
        self.parent = np.full(capacity, -1, dtype=int)
        self.q[0] = root
        self.n = 1
        self.w = weights

    def nearest(self, q: np.ndarray) -> int:
        diff = (self.q[: self.n] - q) * self.w
        return int(np.argmin(np.einsum("ij,ij->i", diff, diff)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.q[self.n] = q
        self.parent[self.n] = parent
        self.n += 1
        return self.n - 1

    def branch(self, i: int) -> list[np.ndarray]:
        """Configurations from node i back to the root."""
        out = []
        while i >= 0:
            out.append(self.q[i].copy())
            i = self.parent[i]
        return out


_TRAPPED, _ADVANCED, _REACHED = 0, 1, 2


class RRTConnect:
    def __init__(self, checker: CollisionChecker, settings: RRTSettings | None = None):
        self.checker = checker
        self.settings = settings or RRTSettings()
        if self.settings.joint_weights is not None:
            self.weights = np.asarray(self.settings.joint_weights, dtype=float)
        else:
            arms = checker.lever_arms
            self.weights = np.maximum(arms / arms[0], 0.2)
        if self.weights.shape != checker.q_min.shape or np.any(self.weights <= 0):
            raise ValueError("joint_weights must be positive, one per joint")

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) * self.weights))

    def path_length(self, path: list[np.ndarray], weighted: bool = False) -> float:
        if len(path) < 2:
            return 0.0
        d = np.diff(np.asarray(path), axis=0)
        if weighted:
            d = d * self.weights
        return float(np.linalg.norm(d, axis=1).sum())

    def _extend(self, tree: _Tree, target: np.ndarray, stats: dict) -> tuple[int, int]:
        """One step from the nearest node towards target: (_TRAPPED|_ADVANCED|_REACHED, new node)."""
        i = tree.nearest(target)
        q_near = tree.q[i]
        d = self.distance(q_near, target)
        if d <= self.settings.step_size:
            q_new, reached = target.copy(), True
        else:
            q_new, reached = q_near + (target - q_near) * (self.settings.step_size / d), False
        stats["extensions"] += 1
        if self.checker.in_collision(q_new):  # inside the sampling box, so only collisions can fail
            return _TRAPPED, -1
        stats["edge_checks"] += 1
        if not self.checker.edge_valid(q_near, q_new):
            return _TRAPPED, -1
        j = tree.add(q_new, i)
        return (_REACHED if reached else _ADVANCED), j

    def _connect(self, tree: _Tree, target: np.ndarray, stats: dict, budget_left) -> tuple[int, int]:
        while True:
            if not budget_left():
                return _TRAPPED, -1
            status, j = self._extend(tree, target, stats)
            if status != _ADVANCED:
                return status, j

    def plan(self, start: np.ndarray, goal: np.ndarray, seed: int = 0) -> RRTResult:
        s = self.settings
        start, goal = np.asarray(start, float).copy(), np.asarray(goal, float).copy()
        rng = np.random.default_rng(seed)
        checks0 = self.checker.config_checks
        t0 = time.perf_counter()
        stats = {"seed": seed, "iterations": 0, "nodes": 0, "extensions": 0, "edge_checks": 0,
                 "collision_checks": 0, "planning_time_s": 0.0, "direct_edge_valid": None}

        def finish(status, path=(), reason=""):
            stats["collision_checks"] = self.checker.config_checks - checks0
            stats["planning_time_s"] = time.perf_counter() - t0
            path = [np.asarray(p, float) for p in path]
            stats["path_vertices"] = len(path)
            stats["path_length_rad"] = self.path_length(path)
            stats["path_length_weighted"] = self.path_length(path, weighted=True)
            return RRTResult(status, path, reason, stats)

        rep = self.checker.check(start)
        if not rep.valid:
            return finish(PlanStatus.INVALID_START, reason=f"start configuration invalid: {rep.reason}")
        rep = self.checker.check(goal)
        if not rep.valid:
            return finish(PlanStatus.INVALID_GOAL, reason=f"goal configuration invalid: {rep.reason}")
        stats["edge_checks"] += 1
        direct = self.checker.edge_valid(start, goal)
        stats["direct_edge_valid"] = direct
        if direct and s.try_direct:
            stats["nodes"] = 2
            return finish(PlanStatus.SUCCESS, [start, goal])

        cap = s.max_nodes + 2
        ta, tb = _Tree(start, cap, self.weights), _Tree(goal, cap, self.weights)
        a_is_start = True
        lo, hi = self.checker.q_min, self.checker.q_max
        exhausted = {"why": None}

        def budget_left() -> bool:
            if ta.n + tb.n >= s.max_nodes:
                exhausted["why"] = PlanStatus.NODE_BUDGET
            elif time.perf_counter() - t0 > s.max_time:
                exhausted["why"] = PlanStatus.TIMEOUT
            return exhausted["why"] is None

        while stats["iterations"] < s.max_iterations and budget_left():
            stats["iterations"] += 1
            q_rand = rng.uniform(lo, hi)
            status, j = self._extend(ta, q_rand, stats)
            if status != _TRAPPED:
                status_b, k = self._connect(tb, ta.q[j], stats, budget_left)
                if status_b == _REACHED:
                    stats["nodes"] = ta.n + tb.n
                    # Node k of tree B sits exactly on node j of tree A: join the two branches there.
                    t_start, i_start, t_goal, i_goal = (ta, j, tb, k) if a_is_start else (tb, k, ta, j)
                    path = t_start.branch(i_start)[::-1] + t_goal.branch(i_goal)[1:]
                    return finish(PlanStatus.SUCCESS, path)
            ta, tb, a_is_start = tb, ta, not a_is_start
        stats["nodes"] = ta.n + tb.n
        why = exhausted["why"]
        if why == PlanStatus.TIMEOUT:
            return finish(why, reason=f"planner timed out after {s.max_time:g} s ({stats['nodes']} nodes)")
        if why == PlanStatus.NODE_BUDGET:
            return finish(why, reason=f"node budget of {s.max_nodes} exceeded")
        return finish(PlanStatus.NO_PATH, reason=f"no valid path found in {s.max_iterations} iterations "
                                                  f"({stats['nodes']} nodes)")
