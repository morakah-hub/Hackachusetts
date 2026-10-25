"""Planned-path markers drawn into a MuJoCo scene (the passive viewer's user_scn).

Visual only: the geoms live in an MjvScene, not in the model, so they never
collide or affect physics. Each joint path is drawn as the end-effector curve
it produces (forward kinematics of densely interpolated straight joint edges),
which is generally curved in Cartesian space.
"""

from collections.abc import Callable

import mujoco
import numpy as np


def ee_polyline(fk: Callable, path: list[np.ndarray], samples_per_edge: int = 12) -> np.ndarray:
    """End-effector points along the straight joint-space edges of `path`."""
    pts = [fk(np.asarray(path[0], float))]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        for s in np.linspace(0.0, 1.0, samples_per_edge + 1)[1:]:
            pts.append(fk(a + (b - a) * s))
    return np.array(pts)


def add_polyline(scene: mujoco.MjvScene, points: np.ndarray, rgba, width: float = 0.004) -> int:
    """Append capsule segments joining consecutive points; returns the number added
    (stops quietly when the scene is full)."""
    added = 0
    for p0, p1 in zip(points[:-1], points[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(p1 - p0) < 1e-9:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0, float), np.asarray(p1, float))
        scene.ngeom += 1
        added += 1
    return added


def add_points(scene: mujoco.MjvScene, points: np.ndarray, rgba, radius: float = 0.012) -> int:
    """Append small spheres (e.g. via points)."""
    added = 0
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                            np.asarray(p, float), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        scene.ngeom += 1
        added += 1
    return added


EXECUTED_RGBA = (0.1, 0.85, 0.3, 0.9)   # shortcut path the robot follows
RAW_RGBA = (1.0, 0.55, 0.1, 0.45)       # RRT-Connect output before shortcutting
VIA_RGBA = (0.1, 0.85, 0.3, 0.7)


def draw_planned_paths(scene: mujoco.MjvScene, fk: Callable, planned: list[tuple[list, list]],
                       show_raw: bool = True) -> int:
    """Draw (raw, executed) joint paths as end-effector curves; returns geoms added."""
    n = 0
    for raw, path in planned:
        if show_raw and raw is not None and len(raw) > 2:
            n += add_polyline(scene, ee_polyline(fk, raw), RAW_RGBA, 0.0025)
        n += add_polyline(scene, ee_polyline(fk, path), EXECUTED_RGBA, 0.004)
        n += add_points(scene, np.array([fk(np.asarray(q, float)) for q in path[1:-1]]), VIA_RGBA)
    return n
