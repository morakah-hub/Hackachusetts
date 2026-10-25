"""Joint-space reference segments with minimum-jerk timing.

Every segment returns (q, qd, qdd) at a time relative to its start. Cartesian straight lines
are converted to a dense joint path by sequential pose IK and interpolated with a cubic spline
over the path parameter, so the reference is smooth in position, velocity and acceleration.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .kinematics import Kinematics
from .robot import VELOCITY_LIMITS

SPEED_SCALE = 0.30          # fraction of the datasheet joint speed limits used for free moves


def min_jerk(tau: float) -> tuple[float, float, float]:
    """s, ds/dtau, d2s/dtau2 for tau in [0, 1]."""
    t = min(max(tau, 0.0), 1.0)
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    ds = 30 * t**2 - 60 * t**3 + 30 * t**4
    dds = 60 * t - 180 * t**2 + 120 * t**3
    return s, ds, dds


class Hold:
    def __init__(self, q: np.ndarray):
        self.q = np.asarray(q, float).copy()
        self.duration = 0.0

    def sample(self, t: float):
        z = np.zeros_like(self.q)
        return self.q, z, z

    @property
    def end(self) -> np.ndarray:
        return self.q


class JointMove:
    def __init__(self, q0: np.ndarray, q1: np.ndarray, min_time: float = 1.0):
        self.q0, self.q1 = np.asarray(q0, float).copy(), np.asarray(q1, float).copy()
        dq = np.abs(self.q1 - self.q0)
        vmax = VELOCITY_LIMITS * SPEED_SCALE
        self.duration = float(max(min_time, 1.875 * np.max(dq / vmax)))

    def sample(self, t: float):
        T = self.duration
        s, ds, dds = min_jerk(t / T)
        d = self.q1 - self.q0
        return self.q0 + s * d, d * ds / T, d * dds / T**2

    @property
    def end(self) -> np.ndarray:
        return self.q1

    def samples(self, n: int = 40):
        return [self.q0 + s * (self.q1 - self.q0) for s in np.linspace(0, 1, n)]


class PathMove:
    """Smooth timing along a dense joint path qs[0..N] (path parameter uniformly spaced)."""

    def __init__(self, qs: list[np.ndarray], duration: float):
        self.qs = np.asarray(qs, float)
        self.u = np.linspace(0.0, 1.0, len(self.qs))
        self.spline = CubicSpline(self.u, self.qs, axis=0, bc_type="clamped")
        self.d1 = self.spline.derivative(1)
        self.d2 = self.spline.derivative(2)
        self.duration = float(duration)

    def sample(self, t: float):
        T = self.duration
        s, ds, dds = min_jerk(t / T)
        sd, sdd = ds / T, dds / T**2
        q = self.spline(s)
        dq = self.d1(s)
        return q, dq * sd, self.d2(s) * sd**2 + dq * sdd

    @property
    def end(self) -> np.ndarray:
        return self.qs[-1]

    def samples(self):
        return list(self.qs)


class LineInfeasible(Exception):
    pass


def cartesian_line(kin: Kinematics, q_start: np.ndarray, p_end: np.ndarray, R: np.ndarray,
                   speed: float = 0.08, min_time: float = 1.0, step: float = 0.005) -> PathMove:
    """Straight end-effector line from FK(q_start) to p_end at a fixed orientation R."""
    p0, _ = kin.fk(q_start)
    dist = float(np.linalg.norm(p_end - p0))
    n = max(2, int(np.ceil(dist / step)) + 1)
    qs = [np.asarray(q_start, float)]
    q = qs[0]
    for s in np.linspace(0, 1, n)[1:]:
        r = kin.solve(p0 + s * (p_end - p0), R, q, rest=q)
        if not r.ok:
            raise LineInfeasible(f"straight line leaves the reachable workspace at {1e3 * s * dist:.0f} mm "
                                 f"({r.reason})")
        if np.abs(r.q - q).max() > 0.15:
            raise LineInfeasible("straight line requires a joint jump (near a singularity or joint limit)")
        q = r.q
        qs.append(q)
    duration = max(min_time, 1.875 * dist / speed)
    return PathMove(qs, duration)


def arc_carry(kin: Kinematics, q_start: np.ndarray, p_end: np.ndarray, a_start: float, a_end: float,
              R_home: np.ndarray, speed: float = 0.30, min_time: float = 1.5, step: float = 0.01) -> PathMove:
    """Carry along an arc around the base at a constant top-down hand orientation: the end-effector moves in
    polar coordinates (radius, azimuth and height interpolated linearly) while the finger axis turns from
    a_start to a_end. Keeps the carried object level (a joint-space swing tilts the hand)."""
    from .kinematics import rot_z

    p0, _ = kin.fk(q_start)
    r0, r1 = float(np.hypot(p0[0], p0[1])), float(np.hypot(p_end[0], p_end[1]))
    t0 = float(np.arctan2(p0[1], p0[0]))
    t1 = t0 + float((np.arctan2(p_end[1], p_end[0]) - t0 + np.pi) % (2 * np.pi) - np.pi)
    length = float(abs(t1 - t0) * 0.5 * (r0 + r1) + abs(r1 - r0) + abs(p_end[2] - p0[2]))
    n = max(2, int(np.ceil(length / step)) + 1)
    qs = [np.asarray(q_start, float)]
    q = qs[0]
    for s in np.linspace(0, 1, n)[1:]:
        r, th = r0 + s * (r1 - r0), t0 + s * (t1 - t0)
        p = np.array([r * np.cos(th), r * np.sin(th), p0[2] + s * (p_end[2] - p0[2])])
        R = rot_z(a_start + s * (a_end - a_start)) @ R_home
        res = kin.solve(p, R, q, rest=q)
        if not res.ok:
            raise LineInfeasible(f"carry arc leaves the reachable workspace ({res.reason})")
        if np.abs(res.q - q).max() > 0.15:
            raise LineInfeasible("carry arc requires a joint jump (near a singularity or joint limit)")
        q = res.q
        qs.append(q)
    qs_arr = np.asarray(qs)
    joint_dist = np.abs(np.diff(qs_arr, axis=0)).sum(axis=0)
    vmax = VELOCITY_LIMITS * SPEED_SCALE
    duration = max(min_time, 1.875 * length / speed, 1.875 * float(np.max(joint_dist / vmax)))
    return PathMove(qs, duration)
