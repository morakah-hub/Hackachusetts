"""Image-derived localization of one upright, known-size box object.

Inputs: a CameraObservation (RGB, metric depth, K, T_world_camera) and a catalog entry for the object: its
colour, full dimensions, the height of the surface it stands on and where the scene layout says it is (a
search window). This module never receives MjData, simulator poses or segmentation IDs: the pose it returns
is measured from the image and depth.

Pipeline: back-project every depth pixel -> keep pixels inside the search window and above the support
surface (and of the object's hue, for saturated colours) -> connected components in the image, take the one
nearest the window centre -> top-face points (highest surface) -> height consistency check against the known
height -> rectangle fit on the top-face points (yaw from convex-hull edge directions) -> center = rectangle
midpoint, z = top - height / 2. Orientation is constrained to upright (roll = pitch = 0).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .camera import CameraObservation, backproject

MIN_PIXELS = 20
MIN_TOP_POINTS = 10
TOP_BAND_M = 0.003   # wider bands pull in near side-face pixels and bias the center toward the camera
HEIGHT_TOL_M = 0.012
EXTENT_TOL_M = 0.016
WINDOW_MARGIN_M = 0.035
MAX_AGE_S = 2.0
SAT_MIN = 0.35       # colours at least this saturated are segmented by hue as well


def _hsv(rgb: np.ndarray):
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    s = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.zeros_like(mx)
    d = np.maximum(delta, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, h)
    h = np.where(mx == g, (b - r) / d + 2, h)
    h = np.where(mx == b, (r - g) / d + 4, h)
    return h * 60.0, s, mx


def hex_hsv(color: str):
    c = color.lstrip("#")
    rgb = np.array([[[int(c[i:i + 2], 16) for i in (0, 2, 4)]]], np.uint8)
    h, s, v = _hsv(rgb)
    return float(h[0, 0]), float(s[0, 0]), float(v[0, 0])


def _fit_rectangle(xy: np.ndarray):
    """Yaw (mod 90 deg), side extents and center of the top-face point set."""
    from scipy.spatial import ConvexHull

    hull = xy[ConvexHull(xy).vertices]
    e = np.roll(hull, -1, axis=0) - hull
    L = np.linalg.norm(e, axis=1)
    th = np.arctan2(e[:, 1], e[:, 0])
    yaw = float(np.arctan2(np.sum(L * np.sin(4 * th)), np.sum(L * np.cos(4 * th))) / 4)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s], [-s, c]])              # world -> box frame
    p = xy @ R.T
    lo, hi = np.percentile(p, 0.5, axis=0), np.percentile(p, 99.5, axis=0)
    return yaw, hi - lo, R.T @ ((lo + hi) / 2)


def _closest_rep(angle: float, ref: float, period: float) -> float:
    return float(ref + ((angle - ref + period / 2) % period) - period / 2)


def estimate_object(obs: CameraObservation, catalog: dict, current_revision: int | None = None) -> dict:
    """Return an ObservedObject dict: valid, reasons, pose (position, yaw, quaternion_wxyz) and diagnostics.

    catalog: id, name, color (#rrggbb), size_m [w, d, h], support_z_m, prior_xy_m, prior_yaw_rad."""
    reasons: list[str] = []
    size = np.asarray(catalog["size_m"], float)
    support_z = float(catalog["support_z_m"])
    prior = np.asarray(catalog["prior_xy_m"], float)
    name = catalog.get("name", catalog["id"])
    out = {"id": catalog["id"], "frame_id": obs.frame_id, "scene_revision": obs.scene_revision,
           "sim_time_s": obs.sim_time_s, "valid": False, "reasons": reasons, "position_m": None, "yaw_rad": None,
           "quaternion_wxyz": None, "size_m": size.tolist(), "pixels": 0, "top_points": 0,
           "top_z_m": None, "top_extent_m": None, "source": "camera_rgbd"}
    if current_revision is not None and obs.scene_revision != current_revision:
        reasons.append(f"stale frame: scene revision {obs.scene_revision}, current {current_revision}")
        return out
    if obs.age_s() > MAX_AGE_S:
        reasons.append(f"stale frame: {obs.age_s():.1f} s old")
        return out
    depth = obs.depth_m
    valid = depth > 0
    if valid.mean() < 0.5:
        reasons.append("camera image unavailable: less than half of the depth pixels are valid "
                       "(camera masked or covered)")
        return out
    H, W = depth.shape
    vs, us = np.nonzero(valid)
    pts = np.zeros((H, W, 3), np.float32)
    pts[vs, us] = backproject(depth, obs.K, obs.T_world_camera, us, vs)
    radius = 0.5 * float(np.hypot(size[0], size[1])) + WINDOW_MARGIN_M
    dxy = np.hypot(pts[..., 0] - prior[0], pts[..., 1] - prior[1])
    mask = valid & (dxy < radius) & (pts[..., 2] > support_z + 0.5 * size[2]) & \
        (pts[..., 2] < support_z + size[2] + 0.03)
    hue, sat, _ = hex_hsv(catalog.get("color", "#888888"))
    colorful = sat >= SAT_MIN
    if colorful:
        h, s, v = _hsv(obs.rgb)
        dh = np.abs((h - hue + 180) % 360 - 180)
        mask &= (dh < 16) & (s > 0.3) & (v > 0.12)
    labels, n = ndimage.label(mask)
    if n == 0:
        reasons.append(f"{name} not visible in the camera image near where it should be")
        return out
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    best, best_d = None, np.inf
    for k in range(n):
        if sizes[k] < MIN_PIXELS:
            continue
        sel = labels == k + 1
        c = pts[sel][:, :2].mean(0)
        dd = float(np.hypot(*(c - prior)))
        if dd < best_d:
            best, best_d = k, dd
    if best is None:
        reasons.append(f"{name}: visible region too small (occluded or out of view)")
        return out
    region = labels == best + 1
    out["pixels"] = int(region.sum())
    rp = pts[region].astype(float)
    z_hi = np.percentile(rp[:, 2], 95)
    top = rp[np.abs(rp[:, 2] - z_hi) < TOP_BAND_M]
    out["top_points"] = int(len(top))
    if len(top) < MIN_TOP_POINTS:
        reasons.append(f"{name}: top face not visible enough ({len(top)} points)")
        return out
    top_z = float(np.median(top[:, 2]))
    out["top_z_m"] = top_z
    expected_top = support_z + size[2]
    if abs(top_z - expected_top) > HEIGHT_TOL_M:
        reasons.append(f"{name}: visible top surface at z = {top_z:.3f} m, expected {expected_top:.3f} m "
                       f"(tilted, lifted or partly occluded)")
        return out
    yaw, ext, center = _fit_rectangle(top[:, :2])
    out["top_extent_m"] = [float(ext[0]), float(ext[1])]
    if np.any(np.abs(np.sort(ext) - np.sort(size[:2])) > EXTENT_TOL_M):
        reasons.append(f"{name}: top face measures {1e3 * ext[0]:.0f} x {1e3 * ext[1]:.0f} mm, expected "
                       f"{1e3 * size[0]:.0f} x {1e3 * size[1]:.0f} mm (partly occluded or merged with a neighbour)")
        return out
    prior_yaw = float(catalog.get("prior_yaw_rad", 0.0))
    if abs(size[0] - size[1]) > 0.010:
        # A rectangle: the fitted axis with the longer extent is the box's longer side.
        if (ext[0] > ext[1]) != (size[0] > size[1]):
            yaw += np.pi / 2
        yaw = _closest_rep(yaw, prior_yaw, np.pi)
    else:
        yaw = _closest_rep(yaw, prior_yaw, np.pi / 2)
    pos = np.array([center[0], center[1], top_z - size[2] / 2])
    out.update(valid=True, position_m=pos.tolist(), yaw_rad=float(yaw),
               quaternion_wxyz=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
               color_segmented=bool(colorful))
    return out
