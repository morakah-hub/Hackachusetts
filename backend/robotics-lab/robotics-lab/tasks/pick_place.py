"""Reference pieces for known-world pick-and-place (v0.6).

The pick-and-place supervisor (simulation/manipulation.py) builds the arm reference one motion
at a time, as phases complete; this module provides the motions. They only produce joint
references (q, qd, qdd); the arm controllers track them and the gripper is commanded separately.

* JointPathSegment: a collision-checked joint path from the motion planner, timed exactly like
  the v0.5 planned moves (JointTargetTask, rest to rest at every via point, v0.4 limits).
* CartesianSegment: a straight end-effector line at constant orientation (approach, lift,
  lower, retreat). Pose IK is solved at closely spaced points along the line, each seeded with
  the previous solution (a continuous branch, no posture jumps); a cubic spline through those
  joint samples, parameterized by the path fraction s, gives q(s), and a quintic timing law
  s(t) gives q(t), qd = q'(s) s', qdd = q''(s) s'^2 + q'(s) s''. The duration is the requested
  one, lengthened until velocity and acceleration stay within the scaled limits (bounded with
  max|q'| and max|q''| over the path). The end-effector pose along the *spline* (not only at the
  IK samples) is checked by `CartesianSegment.validate`.
* ManipulationReference: the reference the controllers see: hold a configuration, or play the
  current segment from its start time and hold its end afterwards. `preview(t)` serves the MPC.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from simulation.kinematics import ArmKinematics, rotation_angle
from tasks.joint_target import JointTargetTask, Reference
from tasks.trajectory import QUINTIC_PEAK_A, QUINTIC_PEAK_V, KinematicLimits, quintic


def carry_arc_points(p0: np.ndarray, p1: np.ndarray, center_xy, step: float = 0.005) -> tuple[np.ndarray, float, float]:
    """Dense polyline from p0 to p1 around a vertical axis through `center_xy` (the robot base): radius,
    azimuth and height interpolated linearly. Azimuths are measured from world x in (-pi, pi], so the arc
    always passes in front of the robot (it never crosses the azimuth +-pi behind the base, where joint 1
    runs out of range). Returns (points, length (m), sweep (rad))."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = np.asarray(center_xy, float)[:2]
    r0, r1 = (float(np.linalg.norm(p[:2] - c)) for p in (p0, p1))
    th0, th1 = (float(np.arctan2(p[1] - c[1], p[0] - c[0])) for p in (p0, p1))
    dth = th1 - th0
    length = float(np.sqrt((0.5 * (r0 + r1) * dth) ** 2 + (r1 - r0) ** 2 + (p1[2] - p0[2]) ** 2))
    s = np.linspace(0.0, 1.0, max(3, int(np.ceil(length / step))) + 1)
    th, r, z = th0 + s * dth, r0 + s * (r1 - r0), p0[2] + s * (p1[2] - p0[2])
    pts = np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), z])
    pts[0], pts[-1] = p0, p1
    return pts, length, dth


def top_down_rotation(yaw: float) -> np.ndarray:
    """End-effector orientation pointing straight down (site z = world -z) with the finger axis
    (site y) at angle `yaw` from world x. yaw = 0 is the Panda's home orientation."""
    c, s = np.cos(yaw), np.sin(yaw)
    y = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(y, z), y, z])


class PathInfeasible(ValueError):
    """A Cartesian line could not be followed (IK failure, posture jump, limits)."""


class JointPathSegment:
    """Planner path (start ... goal) as a rest-to-rest via-point reference (v0.4/v0.5 timing)."""

    def __init__(self, path: list[np.ndarray], edge_time: float, joint_limits: np.ndarray, **timing):
        path = [np.asarray(q, float) for q in path]
        if len(path) < 2:
            path = [path[0], path[0].copy()]
        n = len(path) - 1
        self.task = JointTargetTask(path, edge_time, 0.0, joint_limits, loop=False,
                                    move_times=[edge_time] * n, hold_times=[0.0] * n, **timing)
        self.path = path
        self.duration = self.task.duration
        self.end = path[-1].copy()

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.task.reference(min(t, self.duration))
        return r.q, r.qd, r.qdd


@dataclass
class LineValidation:
    max_position_deviation: float   # m, end-effector distance from the straight line (spline samples)
    max_rotation_error: float       # rad, from the commanded orientation
    max_ik_error: float             # m, at the IK samples
    max_joint_step: float           # rad, largest joint change between consecutive IK samples
    samples: int


@dataclass
class CartesianSegment:
    start_position: np.ndarray
    end_position: np.ndarray
    rotation: np.ndarray
    s_grid: np.ndarray
    q_grid: np.ndarray
    duration: float
    limited_by: str
    spline: CubicSpline = field(repr=False)
    validation: LineValidation | None = None
    path_points: np.ndarray | None = field(default=None, repr=False)  # plan_path only: the requested polyline

    @property
    def end(self) -> np.ndarray:
        return self.q_grid[-1].copy()

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau = min(max(t / self.duration, 0.0), 1.0)
        s, sd, sdd, _ = quintic(tau)
        sd, sdd = sd / self.duration, sdd / self.duration**2
        q = self.spline(s)
        dq, ddq = self.spline(s, 1), self.spline(s, 2)
        return q, dq * sd, ddq * sd * sd + dq * sdd

    @classmethod
    def plan(cls, kin: ArmKinematics, q_start: np.ndarray, end_position: np.ndarray, rotation: np.ndarray,
             requested_time: float, limits: KinematicLimits, step: float = 0.005, tol: float = 1e-4,
             rot_tol: float = 1e-3, max_joint_step: float = 0.1) -> "CartesianSegment":
        q_start = np.asarray(q_start, float)
        p0, R0 = kin.fk_pose(q_start)
        if rotation_angle(R0, rotation) > 0.01:
            raise PathInfeasible(f"start orientation is {np.degrees(rotation_angle(R0, rotation)):.1f} deg "
                                 f"from the line's orientation")
        p1 = np.asarray(end_position, float)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        s_grid = np.linspace(0.0, 1.0, n + 1)
        qs, worst_ik, worst_step = [q_start.copy()], 0.0, 0.0
        for s in s_grid[1:]:
            res = kin.solve_pose_ik(p0 + s * (p1 - p0), rotation, qs[-1], tol, rot_tol, rest=qs[-1])
            if not res.success:
                raise PathInfeasible(f"pose IK failed at s = {s:.3f}: {res.reason}")
            jump = float(np.abs(res.q - qs[-1]).max())
            if jump > max_joint_step:
                raise PathInfeasible(f"posture jump of {jump:.3f} rad at s = {s:.3f}")
            worst_ik, worst_step = max(worst_ik, res.error), max(worst_step, jump)
            qs.append(res.q)
        q_grid = np.array(qs)
        spline = CubicSpline(s_grid, q_grid, axis=0)  # not-a-knot: follows the IK samples closely
        dmax = np.abs(spline(np.linspace(0, 1, 201), 1)).max(axis=0)
        ddmax = np.abs(spline(np.linspace(0, 1, 201), 2)).max(axis=0)
        T, limited_by = float(requested_time), "requested"
        if limits.velocity is not None:
            t_v = float(np.max(dmax * QUINTIC_PEAK_V / np.asarray(limits.velocity)))
            if t_v > T:
                T, limited_by = t_v, "velocity"
        if limits.acceleration is not None:
            a = np.asarray(limits.acceleration)
            # |qdd| <= ddmax (1.875 / T)^2 + dmax 5.7735 / T^2: smallest T meeting every joint
            t_a = float(np.max(np.sqrt((ddmax * QUINTIC_PEAK_V**2 + dmax * QUINTIC_PEAK_A) / a)))
            if t_a > T:
                T, limited_by = t_a, "acceleration"
        seg = cls(p0, p1, np.asarray(rotation, float), s_grid, q_grid, T, limited_by, spline)
        seg.validation = seg.validate(kin, worst_ik, worst_step)
        return seg

    @classmethod
    def plan_path(cls, kin: ArmKinematics, q_start: np.ndarray, points: np.ndarray, rotation: np.ndarray,
                  requested_time: float, limits: KinematicLimits, tol: float = 1e-4, rot_tol: float = 1e-3,
                  max_joint_step: float = 0.1) -> "CartesianSegment":
        """Like plan(), but along a dense end-effector polyline (points[0] = the start position, spacing
        about 5 mm), e.g. an arc around the robot base at constant orientation (studio transport)."""
        q_start = np.asarray(q_start, float)
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
            raise PathInfeasible("a path needs at least 3 points")
        p0, R0 = kin.fk_pose(q_start)
        if rotation_angle(R0, rotation) > 0.01 or np.linalg.norm(p0 - pts[0]) > 1e-3:
            raise PathInfeasible("the path does not start at the current end-effector pose")
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_grid = np.r_[0.0, np.cumsum(seglen)] / max(float(seglen.sum()), 1e-9)
        qs, worst_ik, worst_step = [q_start.copy()], 0.0, 0.0
        for s, p in zip(s_grid[1:], pts[1:]):
            res = kin.solve_pose_ik(p, rotation, qs[-1], tol, rot_tol, rest=qs[-1])
            if not res.success:
                raise PathInfeasible(f"pose IK failed at s = {s:.3f}: {res.reason}")
            jump = float(np.abs(res.q - qs[-1]).max())
            if jump > max_joint_step:
                raise PathInfeasible(f"posture jump of {jump:.3f} rad at s = {s:.3f}")
            worst_ik, worst_step = max(worst_ik, res.error), max(worst_step, jump)
            qs.append(res.q)
        q_grid = np.array(qs)
        spline = CubicSpline(s_grid, q_grid, axis=0)
        dmax = np.abs(spline(np.linspace(0, 1, 401), 1)).max(axis=0)
        ddmax = np.abs(spline(np.linspace(0, 1, 401), 2)).max(axis=0)
        T, limited_by = float(requested_time), "requested"
        if limits.velocity is not None:
            t_v = float(np.max(dmax * QUINTIC_PEAK_V / np.asarray(limits.velocity)))
            if t_v > T:
                T, limited_by = t_v, "velocity"
        if limits.acceleration is not None:
            a = np.asarray(limits.acceleration)
            t_a = float(np.max(np.sqrt((ddmax * QUINTIC_PEAK_V**2 + dmax * QUINTIC_PEAK_A) / a)))
            if t_a > T:
                T, limited_by = t_a, "acceleration"
        seg = cls(pts[0].copy(), pts[-1].copy(), np.asarray(rotation, float), s_grid, q_grid, T, limited_by, spline)
        seg.path_points = pts
        seg.validation = seg.validate(kin, worst_ik, worst_step)
        return seg

    def validate(self, kin: ArmKinematics, worst_ik: float = 0.0, worst_step: float = 0.0,
                 samples: int = 101) -> LineValidation:
        """End-effector pose along the timed reference (spline, not only the IK samples)."""
        pts = self.path_points
        if pts is not None:  # plan_path: distance to the polyline
            a, b = pts[:-1], pts[1:]
            ab = b - a
            ab2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
        d = self.end_position - self.start_position
        L = float(np.linalg.norm(d))
        u = d / L if L > 0 else np.zeros(3)
        dev, rot = 0.0, 0.0
        for t in np.linspace(0.0, self.duration, samples if pts is None else max(samples, 2 * len(pts))):
            p, R = kin.fk_pose(self.evaluate(t)[0])
            if pts is None:
                r = p - self.start_position
                dev = max(dev, float(np.linalg.norm(r - u * (r @ u))))
            else:
                w = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
                dev = max(dev, float(np.linalg.norm(a + w[:, None] * ab - p, axis=1).min()))
            rot = max(rot, rotation_angle(R, self.rotation))
        return LineValidation(dev, rot, worst_ik, worst_step, samples)

    def joint_samples(self, n: int = 50) -> list[np.ndarray]:
        """Configurations along the path (for collision checking as a polyline of IK samples)."""
        return [q.copy() for q in self.q_grid]


class ManipulationReference:
    """Arm reference for the supervisor: hold, or play one segment and hold its end."""

    def __init__(self, q_hold: np.ndarray):
        self.q_hold = np.asarray(q_hold, float).copy()
        self.segment = None
        self.t0 = 0.0

    def hold(self, q: np.ndarray) -> None:
        self.q_hold, self.segment = np.asarray(q, float).copy(), None

    def start(self, segment, t: float) -> None:
        self.segment, self.t0 = segment, float(t)

    def finished(self, t: float) -> bool:
        return self.segment is None or t - self.t0 >= self.segment.duration

    def reference(self, t: float) -> Reference:
        z = np.zeros(len(self.q_hold))
        if self.segment is None:
            return Reference(self.q_hold.copy(), z, 0, z.copy())
        local = t - self.t0
        if local >= self.segment.duration:
            return Reference(self.segment.end, z, 0, z.copy())
        q, qd, qdd = self.segment.evaluate(max(local, 0.0))
        return Reference(q, qd, 0, qdd)

    def preview(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        r = self.reference(t)
        return r.q, r.qd
