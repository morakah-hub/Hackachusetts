"""Cycle the arm through a list of joint-space waypoints.

Each segment is a straight joint-space move from the previous waypoint to the
next, timed by a rest-to-rest profile (tasks/trajectory.py), so position,
velocity and acceleration are continuous and start/end at rest. After each
move the reference holds still for `hold_time` seconds (or per-move
`move_times` / `hold_times`, used for the via points of planned paths). The task only produces
references; it never moves the robot itself.

Without `limits`, every move takes exactly `move_time` with the minimum-jerk
quintic (the v0.1-v0.3 behaviour, kept for historical benchmarks; it does not
check speed or acceleration). With `limits`, each move takes at least
`move_time` and is lengthened when needed to respect the velocity,
acceleration, jerk and (optionally) torque limits; `segments` records the
duration of every move and which limit set it.
"""

import bisect
from dataclasses import dataclass

import numpy as np

from tasks.trajectory import KinematicLimits, SegmentPlan, plan_segment


def min_jerk(s: float) -> tuple[float, float]:
    """Return (position, d/ds) of the minimum-jerk profile for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = 30 * s**2 - 60 * s**3 + 30 * s**4
    return pos, vel


def min_jerk_acc(s: float) -> float:
    """d^2/ds^2 of the minimum-jerk profile (0 outside [0, 1])."""
    if s <= 0.0 or s >= 1.0:
        return 0.0
    return 60 * s - 180 * s**2 + 120 * s**3


@dataclass
class Reference:
    q: np.ndarray
    qd: np.ndarray
    waypoint_index: int  # index of the waypoint currently being approached or held
    qdd: np.ndarray | None = None  # reference acceleration (rad/s^2), for torque feedforward


class JointTargetTask:
    def __init__(
        self,
        waypoints: list[list[float]] | np.ndarray,
        move_time: float = 3.0,
        hold_time: float = 1.0,
        joint_limits: np.ndarray | None = None,
        loop: bool = True,
        profile: str = "min_jerk",
        limits: KinematicLimits | None = None,
        torque_fn=None,
        torque_limits: np.ndarray | None = None,
        move_times: list[float] | None = None,
        hold_times: list[float] | None = None,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or len(self.waypoints) < 2:
            raise ValueError("Need at least two waypoints, each a list of joint angles")
        if move_time <= 0 or hold_time < 0:
            raise ValueError("move_time must be > 0 and hold_time >= 0")
        if joint_limits is not None:
            lo, hi = joint_limits[:, 0], joint_limits[:, 1]
            bad = np.argwhere((self.waypoints < lo) | (self.waypoints > hi))
            if bad.size:
                w, j = bad[0]
                raise ValueError(
                    f"Waypoint {w} joint {j + 1} = {self.waypoints[w, j]:.4f} "
                    f"outside limits [{lo[j]:.4f}, {hi[j]:.4f}]"
                )
        if profile != "min_jerk" and limits is None:
            raise ValueError(f"profile '{profile}' needs kinematic limits")
        self.move_time = move_time  # requested duration of each move
        self.hold_time = hold_time
        self.loop = loop
        self.profile = profile
        self.limits = limits
        n = len(self.waypoints)
        count = n if loop else n - 1
        # Per-move requested duration and hold (v0.5: planned paths pass through via
        # points with no hold); default: move_time and hold_time for every move.
        self.move_times = [move_time] * count if move_times is None else [float(x) for x in move_times]
        self.hold_times = [hold_time] * count if hold_times is None else [float(x) for x in hold_times]
        if len(self.move_times) != count or len(self.hold_times) != count:
            raise ValueError(f"move_times and hold_times need one entry per move ({count})")
        if min(self.move_times) <= 0 or min(self.hold_times) < 0:
            raise ValueError("move_times must be > 0 and hold_times >= 0")
        # One plan per move; without limits this is the plain quintic of the requested time.
        self.segments: list[SegmentPlan] = [
            plan_segment(self.waypoints[k], self.waypoints[(k + 1) % n], self.move_times[k], limits, profile,
                         torque_fn, torque_limits)
            for k in range(count)
        ]
        self._starts = np.r_[0.0, np.cumsum([s.move_time + h for s, h in zip(self.segments, self.hold_times)])]

    @property
    def time_scaled(self) -> bool:
        """True if any move was lengthened beyond the requested move_time."""
        return any(s.limited_by != "requested" for s in self.segments)

    @property
    def segment_time(self) -> float:
        """Duration of each move + hold. Only defined when all moves have the same duration."""
        durations = {round(s.move_time + h, 12) for s, h in zip(self.segments, self.hold_times)}
        if len(durations) != 1:
            raise ValueError("moves have different durations; use segment_bounds(k)")
        return self.segments[0].move_time + self.hold_times[0]

    def segment_bounds(self, k: int) -> tuple[float, float, float]:
        """(start, end of move, end of hold) of segment k within one pass."""
        return float(self._starts[k]), float(self._starts[k] + self.segments[k].move_time), float(self._starts[k + 1])

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def duration(self) -> float:
        """Duration of one pass through all segments."""
        return float(self._starts[-1])

    def reference(self, t: float) -> Reference:
        """Joint reference at time t. Starts at waypoint 0 and moves to 1, 2, ..."""
        n = len(self.waypoints)
        if self.loop:
            t = t % self.duration
        elif t >= self.duration:  # past the last segment: hold the final waypoint
            z = np.zeros(self.waypoints.shape[1])
            return Reference(self.waypoints[-1].copy(), z, n - 1, z.copy())
        k = min(bisect.bisect_right(self._starts, t) - 1, len(self.segments) - 1)
        seg = self.segments[k]
        local_t = t - self._starts[k]
        q, qd, qdd = seg.evaluate(local_t)
        if local_t >= seg.move_time:  # holding: exactly at the goal, at rest
            q, qd, qdd = seg.goal.copy(), np.zeros_like(qd), np.zeros_like(qdd)
        return Reference(q, qd, (k + 1) % n, qdd)

    def timing_report(self) -> list[dict]:
        """Per move: requested and final duration, the limit that set it, and peak/limit ratios."""
        out = []
        for k, s in enumerate(self.segments):
            out.append({
                "segment": k, "from": k, "to": (k + 1) % len(self.waypoints),
                "profile": s.profile, "requested_s": s.requested_time, "move_s": s.move_time,
                "limited_by": s.limited_by,
                "limiting_joint": None if s.limiting_joint is None else s.limiting_joint + 1,
                **{f"peak_{k2}_ratio": v for k2, v in s.peak_ratios.items()},
            })
        return out
