"""Collision-checked shortcutting of joint-space paths.

RRT paths zig-zag. Two passes shorten them while keeping every edge valid:

1. Greedy vertex pruning: from each kept vertex, jump to the farthest later
   vertex that a valid straight edge reaches (deterministic).
2. Random shortcuts (seeded): pick two points uniformly by path length, possibly
   in the middle of edges; if the straight edge between them is valid, replace
   the path between them. New vertices lie on already-validated edges, inside
   the joint box (it is convex), but they and the shortened old edges are
   checked again: discrete checking only certifies points between samples to
   clearance - cartesian_step / 2, and the final re-validation samples them anew.
   Without this, long runs failed re-validation and fell back to the raw path.

Pruning runs again at the end. The result is re-validated vertex by vertex and
edge by edge; if that ever fails the original path is returned unchanged.
No splines: the output is a polyline whose every edge was collision-checked,
which is exactly what the straight-line trajectory segments execute.
"""

from dataclasses import dataclass

import numpy as np

from planning.collision import CollisionChecker


@dataclass
class ShortcutResult:
    path: list[np.ndarray]
    original_length: float      # rad (unweighted joint-space length)
    length: float
    original_vertices: int
    vertices: int
    shortcuts_applied: int
    used_original: bool         # True if the optimized path failed re-validation


def path_length(path: list[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def prune(path: list[np.ndarray], checker: CollisionChecker) -> list[np.ndarray]:
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if j == i + 1 or checker.edge_valid(path[i], path[j]):
                out.append(path[j])
                i = j
                break
    return out


def shortcut(
    path: list[np.ndarray],
    checker: CollisionChecker,
    iterations: int = 100,
    seed: int = 0,
    weights: np.ndarray | None = None,
    min_gain: float = 1e-6,
) -> ShortcutResult:
    """Shorten a valid path. `weights` (per joint) pick the metric used to sample
    points along the path; reported lengths are unweighted (rad)."""
    original = [np.asarray(q, float).copy() for q in path]
    w = np.ones(original[0].shape) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    p = prune(original, checker) if len(original) > 2 else list(original)
    applied = 0
    for _ in range(iterations):
        if len(p) < 3:
            break
        seg = np.linalg.norm(np.diff(np.asarray(p), axis=0) * w, axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        t1, t2 = np.sort(rng.uniform(0.0, cum[-1], 2))
        i1 = min(int(np.searchsorted(cum, t1, side="right")) - 1, len(p) - 2)
        i2 = min(int(np.searchsorted(cum, t2, side="right")) - 1, len(p) - 2)
        if i1 == i2:
            continue  # same edge: already straight
        a = p[i1] + (p[i1 + 1] - p[i1]) * ((t1 - cum[i1]) / seg[i1] if seg[i1] > 0 else 0.0)
        b = p[i2] + (p[i2 + 1] - p[i2]) * ((t2 - cum[i2]) / seg[i2] if seg[i2] > 0 else 0.0)
        old = (np.linalg.norm((a - p[i1 + 1]) * w) + (cum[i2] - cum[i1 + 1]) + np.linalg.norm((p[i2] - b) * w))
        if old - np.linalg.norm((b - a) * w) <= min_gain:
            continue
        # The new vertices and the shortened pieces of the old edges are re-sampled at new
        # positions, so check them like path_valid will (a point on a validated edge is only
        # certified to clearance - cartesian_step / 2, not to the full clearance).
        if not (checker.edge_valid(a, b) and checker.is_valid(a) and checker.is_valid(b)
                and checker.edge_valid(p[i1], a) and checker.edge_valid(b, p[i2 + 1])):
            continue
        new = p[: i1 + 1]
        if np.linalg.norm(a - p[i1]) > 1e-9:
            new.append(a)
        new.append(b)
        if np.linalg.norm(b - p[i2 + 1]) > 1e-9:
            new.append(p[i2 + 1])
        p = new + p[i2 + 2:]
        applied += 1
    p = prune(p, checker) if len(p) > 2 else p
    ok, _ = checker.path_valid(p)
    used_original = not ok
    if used_original:
        p = original
    return ShortcutResult(p, path_length(original), path_length(p), len(original), len(p), applied, used_original)
