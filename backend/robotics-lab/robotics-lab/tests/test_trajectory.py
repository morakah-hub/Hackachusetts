"""Tests for v0.4 trajectory generation: timing laws, time scaling, feasibility checks."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.dynamics import ArmInverseDynamics  # noqa: E402
from robots.panda import ACCELERATION_LIMITS, HOME_QPOS, JERK_LIMITS, VELOCITY_LIMITS, Panda  # noqa: E402
from simulation.benchmark import BENCHMARK_CONFIG  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402
from tasks.joint_target import JointTargetTask  # noqa: E402
from tasks.trajectory import (  # noqa: E402
    QUINTIC_PEAK_A,
    QUINTIC_PEAK_J,
    QUINTIC_PEAK_V,
    DoubleS,
    KinematicLimits,
    TrajectoryInfeasible,
    check_reference,
    plan_segment,
    quintic,
)

LIMITS = KinematicLimits(VELOCITY_LIMITS * 0.9, ACCELERATION_LIMITS * 0.5, JERK_LIMITS * 0.05)
BENCH = yaml.safe_load(open(BENCHMARK_CONFIG, encoding="utf-8"))
WRIST = list(BENCH["scenarios"]["wrist"]["waypoints"].values())
FAST = list(BENCH["scenarios"]["fast"]["waypoints"].values())


def sample(task, dt=0.0005):
    t = np.arange(0.0, task.duration, dt)
    refs = [task.reference(x) for x in t]
    return t, np.array([r.q for r in refs]), np.array([r.qd for r in refs]), np.array([r.qdd for r in refs])


def test_quintic_peak_constants():
    tau = np.linspace(0, 1, 200001)
    vals = np.array([quintic(x) for x in tau])
    assert vals[0, 0] == 0.0 and vals[-1, 0] == 1.0
    assert np.isclose(vals[:, 1].max(), QUINTIC_PEAK_V, rtol=1e-8)
    assert np.isclose(np.abs(vals[:, 2]).max(), QUINTIC_PEAK_A, rtol=1e-6)
    assert np.isclose(np.abs(vals[:, 3]).max(), QUINTIC_PEAK_J, rtol=1e-4)  # supremum, approached at tau -> 0, 1


@pytest.mark.parametrize("v,a,j,regime", [
    (0.5, 10.0, 1000.0, "cruise"),     # reaches the speed limit
    (10.0, 1.0, 5.0, "accel"),         # acceleration limit reached, no cruise
    (10.0, 10.0, 1.0, "jerk"),         # only the jerk limit binds
])
def test_double_s_is_exact_monotonic_and_within_limits(v, a, j, regime):
    p = DoubleS(v, a, j)
    assert abs(p.end_error) < 1e-12
    t = np.linspace(0, p.duration, 40001)
    s = np.array([p(x) for x in t])
    assert np.all(np.diff(s[:, 0]) >= -1e-15)
    assert s[:, 1].max() <= v * (1 + 1e-9) and np.abs(s[:, 2]).max() <= a * (1 + 1e-9)
    assert np.abs(s[:, 3]).max() <= j * (1 + 1e-9)
    assert p(0.0) == (0.0, 0.0, 0.0, 0.0) and p(p.duration + 1)[0] == 1.0
    if regime == "cruise":
        assert p.tv > 0 and np.isclose(s[:, 1].max(), v)
    elif regime == "accel":
        assert p.tv == 0 and np.isclose(np.abs(s[:, 2]).max(), a, rtol=1e-6)
    else:
        assert p.tv == 0 and np.abs(s[:, 2]).max() < a and s[:, 1].max() < v


def test_feasible_request_is_kept_and_infeasible_request_is_lengthened():
    home, left = np.array(FAST[0]), np.array(FAST[1])
    ok = plan_segment(home, left, 3.0, LIMITS)
    assert ok.move_time == 3.0 and ok.limited_by == "requested"
    fast = plan_segment(home, left, 1.0, LIMITS)
    # Joint 7 moves 1.985 rad: the quintic's peak speed 1.875 dq / T must equal 0.9 * 2.61.
    assert fast.limited_by == "velocity" and fast.limiting_joint == 6
    assert np.isclose(fast.move_time, QUINTIC_PEAK_V * abs(left[6] - home[6]) / (0.9 * 2.61))
    assert np.isclose(fast.peak_ratios["velocity"], 1.0)
    assert fast.peak_ratios["acceleration"] <= 1.0 and fast.peak_ratios["jerk"] <= 1.0


def test_short_move_is_acceleration_limited():
    a = HOME_QPOS.copy()
    b = a.copy()
    b[1] += 0.05  # small joint-2 move: acceleration (7.5 * 0.5 rad/s^2) binds before speed
    seg = plan_segment(a, b, 0.01, LIMITS)
    assert seg.limited_by == "acceleration" and seg.limiting_joint == 1
    assert np.isclose(seg.move_time, math.sqrt(QUINTIC_PEAK_A * 0.05 / (7.5 * 0.5)))


@pytest.mark.parametrize("profile", ["min_jerk", "double_s"])
def test_time_scaled_task_respects_limits_and_is_continuous(profile):
    task = JointTargetTask(WRIST, move_time=1.0, hold_time=0.5, profile=profile, limits=LIMITS)
    assert task.time_scaled
    t, q, qd, qdd = sample(task)
    rep = check_reference(t, q, qd, qdd, LIMITS, tol=1e-3)  # jerk by finite differences
    assert rep.feasible, rep.violations
    # Continuous position and velocity across segment boundaries (no jumps between samples).
    assert np.abs(np.diff(q, axis=0)).max() < LIMITS.velocity.max() * 0.0005 * 1.01
    assert np.abs(np.diff(qd, axis=0)).max() < LIMITS.acceleration.max() * 0.0005 * 1.01
    # Each move ends exactly at its waypoint and holds there at rest.
    for k in range(task.num_segments):
        _, move_end, end = task.segment_bounds(k)
        r = task.reference(0.5 * (move_end + end))
        np.testing.assert_array_equal(r.q, task.waypoints[(k + 1) % len(task.waypoints)])
        assert not r.qd.any() and not r.qdd.any()


def test_double_s_is_faster_than_min_jerk_at_the_same_limits():
    mj = JointTargetTask(WRIST, move_time=0.1, hold_time=0.0, profile="min_jerk", limits=LIMITS)
    ds = JointTargetTask(WRIST, move_time=0.1, hold_time=0.0, profile="double_s", limits=LIMITS)
    for a, b in zip(mj.segments, ds.segments):
        assert b.move_time < a.move_time


def test_unlimited_task_reproduces_v03_quintic():
    """Without limits the reference is the v0.3 fixed-duration minimum-jerk profile."""
    task = JointTargetTask(FAST, move_time=1.0, hold_time=1.0)
    start, goal = np.array(FAST[0]), np.array(FAST[1])
    for t in (0.0, 0.123, 0.5, 0.9, 1.0, 1.5):
        s = min(t, 1.0)
        pos = 10 * s**3 - 15 * s**4 + 6 * s**5
        vel = (30 * s**2 - 60 * s**3 + 30 * s**4) if t < 1.0 else 0.0
        r = task.reference(t)
        np.testing.assert_allclose(r.q, start + (goal - start) * pos, atol=1e-15)
        np.testing.assert_allclose(r.qd, (goal - start) * vel, atol=1e-15)
    assert task.segment_time == 2.0 and not task.time_scaled
    # The unchecked reference is infeasible, and the check says exactly why.
    t, q, qd, qdd = sample(task)
    rep = check_reference(t, q, qd, qdd, KinematicLimits(VELOCITY_LIMITS, ACCELERATION_LIMITS))
    assert not rep.feasible and "joint 7 velocity" in rep.violations[0] and "1.43x" in rep.violations[0]


def test_infeasible_waypoint_and_torque_budget_are_reported():
    engine = SimulationEngine("table", Panda(actuation="torque"))
    limits = KinematicLimits(VELOCITY_LIMITS, ACCELERATION_LIMITS, JERK_LIMITS, engine.joint_limits)
    outside = HOME_QPOS.copy()
    outside[3] = 0.5  # joint 4 range is [-3.07, -0.07]
    with pytest.raises(TrajectoryInfeasible, match="joint 4"):
        plan_segment(HOME_QPOS, outside, 2.0, limits)
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    reach_out = np.array([0.0, 1.2, 0.0, -1.0, 0.0, 2.0, 0.0])  # arm stretched forward: large joint-2 gravity torque
    g = dyn.gravity(reach_out)
    budget = np.full(7, 87.0)
    budget[1] = 0.8 * abs(g[1])  # less than gravity needs
    with pytest.raises(TrajectoryInfeasible, match="gravity"):
        plan_segment(HOME_QPOS, reach_out, 2.0, limits, torque_fn=dyn.torque, torque_limits=budget)


def test_torque_budget_lengthens_the_move():
    engine = SimulationEngine("table", Panda(actuation="torque"))
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    a, b = np.array(FAST[0]), np.array(FAST[1])
    free = plan_segment(a, b, 0.6, KinematicLimits())
    ts = np.linspace(0, free.move_time, 101)
    peak = np.abs([dyn.torque(*free.evaluate(t)) for t in ts]).max(axis=0)
    gravity = np.abs([dyn.gravity(free.evaluate(t)[0]) for t in ts]).max(axis=0)
    budget = np.maximum(gravity + 0.5 * (peak - gravity), gravity + 1.0)  # forces a longer move
    seg = plan_segment(a, b, 0.6, KinematicLimits(), torque_fn=dyn.torque, torque_limits=budget)
    assert seg.limited_by == "torque" and seg.move_time > 0.6
    ts = np.linspace(0, seg.move_time, 201)
    tau = np.array([dyn.torque(*seg.evaluate(t)) for t in ts])
    assert np.all(np.abs(tau) <= budget * 1.02)  # sampled at 101 points during planning


def test_runner_builds_time_scaled_tasks_from_config():
    cfg = load_config()
    cfg["task"] = {**cfg["task"], "move_time": 1.0, "waypoints": BENCH["scenarios"]["fast"]["waypoints"]}
    r = SimulationRunner(cfg)
    report = r.task.timing_report()
    assert [s["limited_by"] for s in report][:2] == ["velocity", "velocity"]
    assert report[0]["limiting_joint"] == 7 and report[0]["move_s"] > 1.5
    cfg["task"]["trajectory"] = {**cfg["task"]["trajectory"], "torque_scale": 0.01}  # 0.87 / 0.12 Nm: below gravity
    with pytest.raises(TrajectoryInfeasible):
        SimulationRunner(cfg)


def test_reach_task_hold_end_times_follow_scaled_segments():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "reach.yaml")
    cfg["task"] = {**cfg["task"], "move_time": 0.2}  # too fast for every target: each move gets its own duration
    r = SimulationRunner(cfg)
    task = r.task
    ends = [task.hold_end_time(p) for p in task.reachable]
    assert len(set(np.round(np.diff([0.0] + ends), 9))) > 1  # moves have different durations
    for p, end in zip(task.reachable, ends):
        np.testing.assert_allclose(task.reference(end - 1e-6).q, p.q)
