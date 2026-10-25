"""Joint-space trajectory generation with explicit feasibility checks (v0.4).

A point-to-point move from q_a to q_b follows the straight joint-space line

    q(t) = q_a + (q_b - q_a) s(t),   s: 0 -> 1, rest to rest,

so every joint starts and stops together and the path is exactly the one
the IK/collision checks of simulation/kinematics.py validate. Only the timing
law s(t) changes. With s monotonic and both ends inside the joint box, q(t)
stays inside the joint limits (the box is convex), so position feasibility
reduces to checking the waypoints.

Per joint, |qd_j| = |dq_j| |s'|, |qdd_j| = |dq_j| |s''|, |jerk_j| = |dq_j| |s'''|,
so joint limits become scalar limits on s:

    v_s = min_j v_max_j / |dq_j|,  a_s = min_j a_max_j / |dq_j|,  j_s = min_j jerk_max_j / |dq_j|.

Timing laws:

* "min_jerk" (quintic, the v0.1-v0.3 profile): s = 10 tau^3 - 15 tau^4 + 6 tau^5,
  tau = t / T. Peaks: s' = 1.875 / T, s'' = 5.7735 / T^2, s''' = 60 / T^3, so the
  shortest feasible T is max(1.875 / v_s, sqrt(5.7735 / a_s), cbrt(60 / j_s)).
* "double_s" (jerk-limited, seven phases of constant jerk): the time-optimal
  rest-to-rest profile under (v_s, a_s, j_s) (Biagiotti & Melchiorri, Trajectory
  Planning for Automatic Machines and Robots, 2008, sec. 3.4). It cruises at the
  speed limit, so it is faster than min_jerk at the same limits, but its jerk is
  piecewise constant (acceleration is continuous, jerk is not).

Time scaling: a move never gets less than the requested duration; if the
requested duration violates a limit, the move is lengthened to the shortest
feasible duration (a uniform time stretch s(t) -> s(t T0 / T), which scales
velocity by 1/f, acceleration by 1/f^2 and jerk by 1/f^3).

Torque feasibility (optional): a torque function tau(q, qd, qdd) (inverse
dynamics) is sampled along the move. If any joint needs more than its torque
budget, the move is stretched further; the dynamic part of the torque scales
roughly with 1/f^2 while gravity does not. If gravity alone exceeds the budget
somewhere on the path, no duration helps and the move is reported infeasible.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

QUINTIC_PEAK_V = 1.875                  # max ds/dtau of the quintic
QUINTIC_PEAK_A = 10.0 / math.sqrt(3.0)  # max |d2s/dtau2| = 5.7735
QUINTIC_PEAK_J = 60.0                   # max |d3s/dtau3|
PROFILES = ("min_jerk", "double_s")


class TrajectoryInfeasible(ValueError):
    """No duration makes the move satisfy the limits (the message says why)."""


@dataclass(frozen=True)
class KinematicLimits:
    """Per-joint limits for reference generation. None = not limited."""

    velocity: np.ndarray | None = None      # rad/s
    acceleration: np.ndarray | None = None  # rad/s^2
    jerk: np.ndarray | None = None          # rad/s^3
    position: np.ndarray | None = None      # (n, 2) rad

    def scaled(self, velocity: float = 1.0, acceleration: float = 1.0, jerk: float = 1.0) -> "KinematicLimits":
        """Copy with each limit multiplied by a factor (e.g. 0.9 for a 10% margin)."""
        def sc(x, f):
            return None if x is None else np.asarray(x, float) * f
        return KinematicLimits(sc(self.velocity, velocity), sc(self.acceleration, acceleration),
                               sc(self.jerk, jerk), self.position)


# ------------------------------------------------------------------ timing laws


def quintic(tau: float) -> tuple[float, float, float, float]:
    """(s, ds, dds, ddds) w.r.t. tau of the minimum-jerk quintic, clamped to [0, 1]."""
    if tau <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    if tau >= 1.0:
        return 1.0, 0.0, 0.0, 0.0
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    ddds = 60 - 360 * tau + 360 * tau**2
    return s, ds, dds, ddds


class DoubleS:
    """Time-optimal rest-to-rest jerk-limited profile for s: 0 -> 1 under (v, a, j) limits on s."""

    def __init__(self, v: float, a: float, j: float):
        if not (v > 0 and a > 0 and j > 0):
            raise ValueError("double_s needs positive velocity, acceleration and jerk limits")
        # Case 1: the speed limit is reached (with or without reaching the acceleration limit).
        if v * j >= a * a:
            tj, ta = a / j, a / j + v / a
        else:
            tj = math.sqrt(v / j)
            ta = 2 * tj
        tv = 1.0 / v - ta
        if tv < 0.0:  # Case 2: too short to cruise.
            tv = 0.0
            tj = a / j
            ta = (a * a / j + math.sqrt(a**4 / j**2 + 4 * a)) / (2 * a)
            if ta < 2 * tj:  # acceleration limit not reached either
                tj = (1.0 / (2 * j)) ** (1.0 / 3.0)
                ta = 2 * tj
        self.tj, self.ta, self.tv = tj, ta, tv
        self.duration = 2 * ta + tv
        # Phase boundaries and constant jerk per phase; integrate exactly.
        durs = [tj, ta - 2 * tj, tj, tv, tj, ta - 2 * tj, tj]
        jerks = [j, 0.0, -j, 0.0, -j, 0.0, j]
        self._t0, self._state, self._jerk = [], [], jerks
        t, s, sd, sdd = 0.0, 0.0, 0.0, 0.0
        for d, jj in zip(durs, jerks):
            self._t0.append(t)
            self._state.append((s, sd, sdd))
            s, sd, sdd = (s + sd * d + sdd * d * d / 2 + jj * d**3 / 6, sd + sdd * d + jj * d * d / 2, sdd + jj * d)
            t += d
        self.end_error = s - 1.0  # exactness check (tests assert ~1e-15)
        self.peak_v = j * tj * (ta - tj)
        self.peak_a = j * tj

    def __call__(self, t: float) -> tuple[float, float, float, float]:
        """(s, s', s'', s''') at time t (s), clamped outside [0, duration]."""
        if t <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        if t >= self.duration:
            return 1.0, 0.0, 0.0, 0.0
        i = max(k for k, t0 in enumerate(self._t0) if t0 <= t)
        s, sd, sdd = self._state[i]
        jj = self._jerk[i]
        d = t - self._t0[i]
        return (s + sd * d + sdd * d * d / 2 + jj * d**3 / 6, sd + sdd * d + jj * d * d / 2, sdd + jj * d, jj)


# ------------------------------------------------------------------ segments


@dataclass
class SegmentPlan:
    """Timing of one point-to-point move and why it got that duration."""

    start: np.ndarray
    goal: np.ndarray
    profile: str
    requested_time: float
    move_time: float
    limited_by: str                 # "requested", "velocity", "acceleration", "jerk" or "torque"
    limiting_joint: int | None      # 0-based joint index, None if "requested"
    peak_ratios: dict = field(default_factory=dict)  # max over joints of peak / limit
    _law: Callable | None = None    # s(t) on [0, base_time]
    _base_time: float = 1.0

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint (q, qd, qdd) at time t after the move started (held at the goal afterwards)."""
        f = self.move_time / self._base_time      # time stretch factor
        s, sd, sdd, _ = self._law(t / f)
        dq = self.goal - self.start
        return self.start + dq * s, dq * (sd / f), dq * (sdd / f**2)

    def jerk_peak(self) -> np.ndarray:
        """Peak |jerk| per joint (rad/s^3) of this move."""
        f = self.move_time / self._base_time
        if self.profile == "min_jerk":
            return np.abs(self.goal - self.start) * QUINTIC_PEAK_J / self.move_time**3
        return np.abs(self.goal - self.start) * self._law.j_s / f**3


def _scalar_limits(dq: np.ndarray, limits: KinematicLimits) -> dict[str, tuple[float, int | None]]:
    """Limits on s implied by each joint limit: name -> (value, limiting joint)."""
    out = {}
    adq = np.abs(dq)
    moving = adq > 1e-12
    for name in ("velocity", "acceleration", "jerk"):
        lim = getattr(limits, name)
        if lim is None or not moving.any():
            out[name] = (math.inf, None)
            continue
        ratio = np.full(len(dq), math.inf)
        ratio[moving] = np.asarray(lim, float)[moving] / adq[moving]
        j = int(np.argmin(ratio))
        out[name] = (float(ratio[j]), j)
    return out


def plan_segment(
    start: np.ndarray,
    goal: np.ndarray,
    requested_time: float,
    limits: KinematicLimits | None = None,
    profile: str = "min_jerk",
    torque_fn: Callable | None = None,
    torque_limits: np.ndarray | None = None,
    samples: int = 101,
) -> SegmentPlan:
    """Shortest move no shorter than `requested_time` that respects the limits.

    torque_fn(q, qd, qdd) -> joint torques; with torque_limits (n,) or (n, 2) it
    adds a torque-feasibility check (lengthening the move or raising
    TrajectoryInfeasible when gravity alone exceeds the limit)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Use one of {PROFILES}")
    if requested_time <= 0:
        raise ValueError("move time must be > 0")
    limits = limits or KinematicLimits()
    if limits.position is not None:
        lo, hi = limits.position[:, 0], limits.position[:, 1]
        for name, q in (("start", start), ("goal", goal)):
            bad = np.nonzero((q < lo) | (q > hi))[0]
            if bad.size:
                j = int(bad[0])
                raise TrajectoryInfeasible(f"{name} joint {j + 1} = {q[j]:.4f} outside [{lo[j]:.4f}, {hi[j]:.4f}]")
    dq = goal - start
    sl = _scalar_limits(dq, limits)

    if profile == "min_jerk":
        need = {
            "velocity": QUINTIC_PEAK_V / sl["velocity"][0],
            "acceleration": math.sqrt(QUINTIC_PEAK_A / sl["acceleration"][0]),
            "jerk": (QUINTIC_PEAK_J / sl["jerk"][0]) ** (1.0 / 3.0),
        }
        law, base = (lambda tau: quintic(tau)), 1.0
    else:
        v, a, j = sl["velocity"][0], sl["acceleration"][0], sl["jerk"][0]
        if not all(math.isfinite(x) for x in (v, a, j)):
            if np.allclose(dq, 0.0):
                v = a = j = 1.0  # a stationary "move": any profile works
            else:
                raise ValueError("double_s needs velocity, acceleration and jerk limits")
        law = DoubleS(v, a, j)
        law.j_s = j
        base = law.duration
        # The double-S is already time-optimal for (v, a, j); report its binding limit.
        binding = "velocity" if law.tv > 0 else ("acceleration" if law.ta > 2 * law.tj + 1e-12 else "jerk")
        need = {binding: base}

    limited_by, joint, move_time = "requested", None, requested_time
    for name, t_min in need.items():
        if t_min > move_time + 1e-12:
            limited_by, joint, move_time = name, sl[name][1], t_min

    seg = SegmentPlan(start, goal, profile, requested_time, move_time, limited_by, joint, _law=law, _base_time=base)

    if torque_fn is not None and torque_limits is not None and not np.allclose(dq, 0.0):
        tl = np.asarray(torque_limits, float)
        tl = tl if tl.ndim == 1 else np.abs(tl).max(axis=1)
        for _ in range(40):
            ts = np.linspace(0.0, seg.move_time, samples)
            tau = np.array([torque_fn(*seg.evaluate(t)) for t in ts])
            grav = np.array([torque_fn(seg.evaluate(t)[0], np.zeros_like(dq), np.zeros_like(dq)) for t in ts])
            ratio = np.abs(tau) / tl
            if ratio.max() <= 1.0:
                break
            if np.any(np.abs(grav) >= tl):
                i, j = np.unravel_index(np.argmax(np.abs(grav) / tl), grav.shape)
                raise TrajectoryInfeasible(
                    f"joint {j + 1} needs {abs(grav[i, j]):.1f} Nm just to hold against gravity "
                    f"(budget {tl[j]:.1f} Nm): no duration is feasible")
            # Dynamic torque scales ~1/f^2: stretch enough for the worst sample, at least 5%.
            dyn = np.abs(tau - grav)
            room = np.maximum(tl - np.abs(grav), 1e-9)
            f = max(1.05, math.sqrt(float((dyn / room).max())))
            seg.move_time *= f
            seg.limited_by, seg.limiting_joint = "torque", int(np.unravel_index(np.argmax(ratio), ratio.shape)[1])
        else:
            raise TrajectoryInfeasible("torque check did not converge after 40 time stretches")
        seg.peak_ratios["torque"] = float(ratio.max())

    # Peak ratios at the final duration (analytic for velocity/acceleration/jerk).
    T, adq = seg.move_time, np.abs(dq)
    if profile == "min_jerk":
        peaks = {"velocity": adq * QUINTIC_PEAK_V / T, "acceleration": adq * QUINTIC_PEAK_A / T**2,
                 "jerk": adq * QUINTIC_PEAK_J / T**3}
    else:
        f = T / base
        peaks = {"velocity": adq * law.peak_v / f, "acceleration": adq * law.peak_a / f**2,
                 "jerk": adq * law.j_s / f**3}
    for name, p in peaks.items():
        lim = getattr(limits, name)
        if lim is not None:
            seg.peak_ratios[name] = float(np.max(p / np.asarray(lim, float)))
    return seg


# ------------------------------------------------------------------ reference checks


@dataclass
class FeasibilityReport:
    """Peak demand of a sampled reference relative to the limits."""

    feasible: bool
    ratios: dict                 # name -> max over time and joints of |value| / limit
    violations: list[str]        # human-readable reasons (empty if feasible)


def check_reference(
    t: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray | None,
    limits: KinematicLimits,
    torque: np.ndarray | None = None,
    torque_limits: np.ndarray | None = None,
    tol: float = 1e-9,
) -> FeasibilityReport:
    """Check sampled reference arrays (time-major) against the limits."""
    ratios, reasons = {}, []
    t = np.asarray(t, float)

    def check(name, values, lim, unit):
        if values is None or lim is None:
            return
        lim = np.asarray(lim, float)
        r = np.abs(values) / lim
        i, j = np.unravel_index(np.argmax(r), r.shape)
        ratios[name] = float(r[i, j])
        if r[i, j] > 1.0 + tol:
            reasons.append(f"joint {j + 1} {name} {abs(values[i, j]):.3f} {unit} > limit {lim[j]:.3f} "
                           f"({r[i, j]:.2f}x) at t = {t[i]:.3f} s")

    if limits.position is not None:
        lo, hi = limits.position[:, 0], limits.position[:, 1]
        out = np.maximum(lo - q, q - hi)
        i, j = np.unravel_index(np.argmax(out), out.shape)
        ratios["position_excess"] = float(max(out[i, j], 0.0))
        if out[i, j] > tol:
            reasons.append(f"joint {j + 1} position {q[i, j]:.4f} rad outside its range at t = {t[i]:.3f} s")
    check("velocity", qd, limits.velocity, "rad/s")
    check("acceleration", qdd, limits.acceleration, "rad/s^2")
    if qdd is not None and limits.jerk is not None and len(t) > 2:
        jerk = np.diff(qdd, axis=0) / np.diff(t)[:, None]
        check("jerk", jerk, limits.jerk, "rad/s^3")
    if torque is not None and torque_limits is not None:
        tl = np.asarray(torque_limits, float)
        check("torque", torque, tl if tl.ndim == 1 else np.abs(tl).max(axis=1), "Nm")
    return FeasibilityReport(not reasons, ratios, reasons)
