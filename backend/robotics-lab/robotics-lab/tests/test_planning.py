"""Tests for v0.5 motion planning: collision checking, RRT-Connect, IK candidates, shortcutting."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker, _coarse_to_fine  # noqa: E402
from planning.ik import ik_candidates  # noqa: E402
from planning.motion_planner import NO_VALID_IK, UNREACHABLE, MotionPlanner, PlanningSettings  # noqa: E402
from planning.rrt_connect import PlanStatus, RRTConnect, RRTSettings  # noqa: E402
from planning.shortcut import path_length, shortcut  # noqa: E402
from planning.visualization import draw_planned_paths  # noqa: E402
from robots.panda import HOME_QPOS, Panda  # noqa: E402
from simulation.engine import SimulationEngine, build_spec  # noqa: E402
from simulation.kinematics import ArmKinematics  # noqa: E402

# Two arm configurations on either side of a box placed between them (see obstacle_model):
# joint 1 swings the outstretched arm across the box.
Q_LEFT = np.array([0.6, 0.5, 0.0, -1.9, 0.0, 2.4, 0.785])
Q_RIGHT = np.array([-0.6, 0.5, 0.0, -1.9, 0.0, 2.4, 0.785])


@pytest.fixture(scope="module")
def table_engine():
    return SimulationEngine("table")


@pytest.fixture(scope="module")
def demo_engine():
    return SimulationEngine("planning_demo")


def checker_for(engine, **kw) -> CollisionChecker:
    return CollisionChecker.from_engine(engine, **kw)


def obstacle_model(engine: SimulationEngine, size=(0.04, 0.04, 0.06)) -> mujoco.MjModel:
    """The table scene plus a box deliberately placed where the end effector passes
    halfway between Q_LEFT and Q_RIGHT (the straight joint-space line)."""
    mid_ee = engine.forward_kinematics((Q_LEFT + Q_RIGHT) / 2)
    spec = build_spec("table", Panda())
    body = spec.worldbody.add_body(name="blocker", pos=mid_ee)
    body.add_geom(name="blocker", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size))
    return spec.compile()


@pytest.fixture(scope="module")
def blocked_checker(table_engine):
    e = table_engine
    return CollisionChecker(obstacle_model(e), e.data.qpos, e.arm_qpos_adr, e.arm_dof_adr,
                            e._robot_body_ids(), e.joint_limits)


# --- Configuration checks ---------------------------------------------------------


def test_home_valid_and_config_in_table_rejected(table_engine):
    c = checker_for(table_engine)
    assert c.check(HOME_QPOS).valid
    kin = ArmKinematics(table_engine)
    res = kin.solve_ik(np.array([0.5, 0.1, 0.36]), HOME_QPOS)  # below the table top
    rep = c.check(res.q)
    assert not rep.valid and "table" in rep.reason and rep.contacts


def test_joint_limits_enforced_with_margin(table_engine):
    c = checker_for(table_engine, limit_margin=0.05)
    q = HOME_QPOS.copy()
    q[3] = table_engine.joint_limits[3, 1] - 0.01  # inside the physical range, inside the margin
    rep = c.check(q)
    assert not rep.valid and "joint 4" in rep.reason and "outside" in rep.reason
    q[3] = table_engine.joint_limits[3, 1] - 0.06
    assert c.check(q).valid
    assert not c.check(np.zeros(6)).valid  # wrong dimension is rejected, not crashed


def test_clearance_uses_true_distance_and_rejects_near_misses():
    """The v0.3 IK posture for (0.68, 0.42, 0.52) is 17 mm from an obstacle: free at
    zero clearance, rejected at 20 mm, and contact.dist equals mj_geomDistance."""
    e = SimulationEngine("obstacles")
    q = ArmKinematics(e).plan_target(np.array([0.68, 0.42, 0.52]), HOME_QPOS).q
    strict, loose = checker_for(e, clearance=0.02), checker_for(e, clearance=0.0)
    assert loose.check(q).valid
    rep = strict.check(q)
    assert not rep.valid and "within clearance" in rep.reason
    d = strict.min_distance(q)
    assert 0.0 < d < 0.02
    m, data = strict.model, strict.data
    c = data.contact[0]
    ref = mujoco.mj_geomDistance(m, data, c.geom1, c.geom2, 0.5, np.zeros(6))
    assert abs(c.dist - ref) < 1e-5


def test_checker_respects_intended_filters_and_leaves_plant_unchanged(table_engine):
    """The robot base rests on the table (excluded pair); the margin is added on a copy only."""
    c = checker_for(table_engine, clearance=0.05)
    assert c.check(HOME_QPOS).valid  # link0 sits on the table: excluded, not a violation
    assert np.all(table_engine.model.geom_margin == 0.0)
    assert np.any(c.model.geom_margin > 0.0)


def test_checker_never_touches_live_state(demo_engine):
    e = demo_engine
    before = e.data.qpos.copy(), e.data.qvel.copy(), e.data.ctrl.copy(), e.data.time
    p = MotionPlanner.from_engine(e)
    p.plan_to_position(np.array([0.45, -0.25, 0.50]), np.array([0.246, 0.621, 0.211, -1.791, 0.089, 1.756, -0.785]))
    np.testing.assert_array_equal(before[0], e.data.qpos)
    np.testing.assert_array_equal(before[1], e.data.qvel)
    np.testing.assert_array_equal(before[2], e.data.ctrl)
    assert before[3] == e.data.time


# --- Edge checks --------------------------------------------------------------------


def test_obstacle_between_two_valid_configurations_blocks_the_edge(table_engine, blocked_checker):
    free = checker_for(table_engine)
    assert free.check(Q_LEFT).valid and free.check(Q_RIGHT).valid
    assert free.edge_valid(Q_LEFT, Q_RIGHT)  # without the box the straight move is fine
    c = blocked_checker
    assert c.check(Q_LEFT).valid and c.check(Q_RIGHT).valid  # both endpoints still valid...
    assert not c.edge_valid(Q_LEFT, Q_RIGHT)  # ...but the edge between them is not
    s = c.first_invalid(Q_LEFT, Q_RIGHT)
    assert s is not None and 0.2 < s < 0.8
    ok, reason = c.path_valid([Q_LEFT, Q_RIGHT])
    assert not ok and "edge 0 -> 1" in reason


def test_edge_sampling_bound_is_conservative(table_engine):
    """No point of any robot collision geom moves further between two edge samples than
    cartesian_step (checked on geom centres and bounding-sphere extremes)."""
    c = checker_for(table_engine)
    m, d = c.model, mujoco.MjData(c.model)
    geoms = [g for g in range(m.ngeom) if c._is_robot_geom[g] and m.geom_contype[g]]
    rng = np.random.default_rng(3)

    def points(q):
        d.qpos[:] = c.base_qpos
        d.qpos[c.qpos_adr] = q
        mujoco.mj_kinematics(m, d)
        out = []
        for g in geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            for off in np.vstack([np.zeros(3), np.eye(3), -np.eye(3)]) * m.geom_rbound[g]:
                out.append(d.geom_xpos[g] + R @ off)
        return np.array(out)

    for _ in range(8):
        q0, q1 = rng.uniform(c.q_min, c.q_max), rng.uniform(c.q_min, c.q_max)
        n = c.edge_samples(q0, q1)
        assert c.motion_bound(q0, q1) / n <= c.cartesian_step + 1e-12
        # Path length of every tracked point along the whole edge (finely sampled) stays within the bound.
        pts = np.array([points(q0 + (q1 - q0) * s) for s in np.linspace(0, 1, 250)])
        travelled = np.linalg.norm(np.diff(pts, axis=0), axis=2).sum(axis=0)
        assert travelled.max() <= c.motion_bound(q0, q1) + 1e-9


def test_coarse_to_fine_visits_every_interior_sample_once():
    for n in (1, 2, 3, 7, 8, 13, 64):
        order = list(_coarse_to_fine(n))
        assert sorted(order) == list(range(1, n))
    assert list(_coarse_to_fine(8))[:3] == [4, 2, 6]


# --- RRT-Connect ----------------------------------------------------------------------


def test_rrt_connect_finds_valid_alternative_around_deliberate_obstacle(blocked_checker):
    rrt = RRTConnect(blocked_checker, RRTSettings(max_time=20.0))
    res = rrt.plan(Q_LEFT, Q_RIGHT, seed=0)
    assert res.success, res.reason
    assert res.stats["direct_edge_valid"] is False
    assert len(res.path) > 2  # not the direct edge
    np.testing.assert_array_equal(res.path[0], Q_LEFT)
    np.testing.assert_array_equal(res.path[-1], Q_RIGHT)
    ok, reason = blocked_checker.path_valid(res.path)
    assert ok, reason
    assert res.stats["nodes"] >= len(res.path)
    assert res.stats["collision_checks"] > 0 and res.stats["planning_time_s"] > 0


def test_tree_edges_respect_step_size(blocked_checker):
    """Every path edge is a single EXTEND step (CONNECT also extends one step at a time)."""
    rrt = RRTConnect(blocked_checker, RRTSettings(step_size=0.25, max_time=20.0))
    res = rrt.plan(Q_LEFT, Q_RIGHT, seed=1)
    assert res.success
    steps = [rrt.distance(a, b) for a, b in zip(res.path[:-1], res.path[1:])]
    assert max(steps) <= 0.25 + 1e-9


def test_rrt_is_deterministic_for_a_seed(blocked_checker):
    rrt = RRTConnect(blocked_checker, RRTSettings(max_time=20.0))
    a, b = rrt.plan(Q_LEFT, Q_RIGHT, seed=5), rrt.plan(Q_LEFT, Q_RIGHT, seed=5)
    assert a.success and len(a.path) == len(b.path)
    for x, y in zip(a.path, b.path):
        np.testing.assert_array_equal(x, y)
    assert a.stats["nodes"] == b.stats["nodes"] and a.stats["collision_checks"] == b.stats["collision_checks"]


def test_direct_edge_used_when_free(table_engine):
    rrt = RRTConnect(checker_for(table_engine))
    res = rrt.plan(Q_LEFT, Q_RIGHT, seed=0)
    assert res.success and len(res.path) == 2 and res.stats["direct_edge_valid"] is True
    res = RRTConnect(checker_for(table_engine), RRTSettings(try_direct=False)).plan(Q_LEFT, Q_RIGHT, seed=0)
    assert res.success and res.stats["direct_edge_valid"] is True  # still recorded


@pytest.mark.parametrize("settings, status", [
    (RRTSettings(max_time=1e-4), PlanStatus.TIMEOUT),
    (RRTSettings(max_nodes=6), PlanStatus.NODE_BUDGET),
    (RRTSettings(max_iterations=1, step_size=0.05), PlanStatus.NO_PATH),
])
def test_budgets_end_search_with_explicit_status(blocked_checker, settings, status):
    res = RRTConnect(blocked_checker, settings).plan(Q_LEFT, Q_RIGHT, seed=0)
    assert res.status == status and not res.success and res.path == [] and res.reason


def test_invalid_start_and_goal_rejected(blocked_checker, table_engine):
    rrt = RRTConnect(blocked_checker)
    in_table = ArmKinematics(table_engine).solve_ik(np.array([0.5, 0.1, 0.36]), HOME_QPOS).q
    res = rrt.plan(in_table, Q_RIGHT)
    assert res.status == PlanStatus.INVALID_START and "start" in res.reason
    res = rrt.plan(Q_LEFT, in_table)
    assert res.status == PlanStatus.INVALID_GOAL and "goal" in res.reason
    beyond = Q_LEFT.copy()
    beyond[0] = 3.0  # outside joint 1's range
    res = rrt.plan(beyond, Q_RIGHT)
    assert res.status == PlanStatus.INVALID_START and "outside" in res.reason


# --- Shortcutting -------------------------------------------------------------------


def test_shortcut_shortens_and_keeps_path_valid(blocked_checker):
    rrt = RRTConnect(blocked_checker, RRTSettings(max_time=20.0))
    raw = rrt.plan(Q_LEFT, Q_RIGHT, seed=2).path
    res = shortcut(raw, blocked_checker, iterations=100, seed=0, weights=rrt.weights)
    assert not res.used_original
    assert res.length <= res.original_length + 1e-12 and res.original_length == pytest.approx(path_length(raw))
    assert res.vertices <= res.original_vertices
    np.testing.assert_array_equal(res.path[0], Q_LEFT)
    np.testing.assert_array_equal(res.path[-1], Q_RIGHT)
    ok, reason = blocked_checker.path_valid(res.path)
    assert ok, reason
    assert all(blocked_checker.within_limits(q) for q in res.path)
    again = shortcut(raw, blocked_checker, iterations=100, seed=0, weights=rrt.weights)  # deterministic
    assert all(np.array_equal(a, b) for a, b in zip(res.path, again.path))


def test_shortcut_falls_back_to_original_if_revalidation_fails(blocked_checker, monkeypatch):
    raw = RRTConnect(blocked_checker, RRTSettings(max_time=20.0)).plan(Q_LEFT, Q_RIGHT, seed=2).path
    monkeypatch.setattr(blocked_checker, "path_valid", lambda path: (False, "forced"))
    res = shortcut(raw, blocked_checker, iterations=50, seed=0)
    assert res.used_original and len(res.path) == len(raw)
    assert all(np.array_equal(a, b) for a, b in zip(res.path, raw))


# --- IK candidates and Cartesian planning ---------------------------------------------


DEMO_START = np.array([0.246, 0.621, 0.211, -1.791, 0.089, 1.756, -0.785])


def test_multiple_valid_ik_candidates(demo_engine):
    p = MotionPlanner.from_engine(demo_engine)
    target = np.array([0.45, -0.25, 0.50])
    ik = ik_candidates(p.kin, p.checker, target, DEMO_START, random_seeds=12, seed=0, weights=p.rrt.weights)
    assert ik.converged >= 3 and len(ik.candidates) >= 2
    for q in ik.candidates:
        assert p.checker.check(q).valid
        assert np.linalg.norm(p.kin.fk(q) - target) < 1e-4
    for a in range(len(ik.candidates)):  # distinct postures, nearest to the start first
        for b in range(a + 1, len(ik.candidates)):
            assert np.abs(ik.candidates[a] - ik.candidates[b]).max() >= 0.05
    dist = [np.linalg.norm((q - DEMO_START) * p.rrt.weights) for q in ik.candidates]
    assert dist == sorted(dist)


def test_cartesian_failure_reasons(demo_engine):
    p = MotionPlanner.from_engine(demo_engine)
    far = p.plan_to_position(np.array([1.3, 0.0, 0.6]), DEMO_START, seed=0)
    assert not far.success and far.status == UNREACHABLE and "unreachable" in far.reason
    in_wall = p.plan_to_position(np.array([0.50, 0.0, 0.55]), DEMO_START, seed=0)  # inside the wall
    assert not in_wall.success and in_wall.status == NO_VALID_IK and "no collision-free IK" in in_wall.reason
    in_table = ArmKinematics(demo_engine).solve_ik(np.array([0.5, 0.3, 0.36]), HOME_QPOS).q
    bad_start = p.plan_to_position(np.array([0.45, -0.25, 0.50]), in_table, seed=0)
    assert bad_start.status == PlanStatus.INVALID_START.value
    tight = MotionPlanner.from_engine(demo_engine, PlanningSettings(rrt=RRTSettings(max_time=1e-3)))
    timeout = tight.plan_to_position(np.array([0.45, -0.25, 0.50]), DEMO_START, seed=0)
    assert not timeout.success and timeout.status == PlanStatus.TIMEOUT.value
    budget = MotionPlanner.from_engine(demo_engine, PlanningSettings(rrt=RRTSettings(max_nodes=4)))
    nodes = budget.plan_to_position(np.array([0.45, -0.25, 0.50]), DEMO_START, seed=0)
    assert not nodes.success and nodes.status == PlanStatus.NODE_BUDGET.value
    assert nodes.stats["candidates_tried"] >= 2  # every IK posture was attempted


def test_plan_to_position_reaches_target_around_wall(demo_engine):
    p = MotionPlanner.from_engine(demo_engine)
    target = np.array([0.45, -0.25, 0.50])
    plan = p.plan_to_position(target, DEMO_START, seed=0)
    assert plan.success and plan.stats["direct_edge_valid"] is False
    assert np.linalg.norm(p.kin.fk(plan.q) - target) < 1e-4
    np.testing.assert_allclose(plan.path[0], DEMO_START)
    np.testing.assert_array_equal(plan.path[-1], plan.q)
    assert p.checker.path_valid(plan.path)[0] and p.checker.path_valid(plan.raw_path)[0]
    assert plan.stats["smoothed_length_rad"] <= plan.stats["raw_length_rad"] + 1e-12


def test_planning_settings_from_config_rejects_unknown_keys():
    s = PlanningSettings.from_config({"enabled": True, "seed": 3, "rrt": {"max_nodes": 10},
                                      "shortcut": {"iterations": 7}, "ik": {"random_seeds": 2}})
    assert s.seed == 3 and s.rrt.max_nodes == 10 and s.shortcut_iterations == 7 and s.ik_random_seeds == 2
    with pytest.raises(ValueError, match="Unknown planning settings"):
        PlanningSettings.from_config({"clearence": 0.1})


def test_planned_path_markers_render_into_scene(demo_engine):
    p = MotionPlanner.from_engine(demo_engine)
    plan = p.plan_to_position(np.array([0.45, -0.25, 0.50]), DEMO_START, seed=0)
    scene = mujoco.MjvScene(demo_engine.model, maxgeom=2000)
    n = draw_planned_paths(scene, p.kin.fk, [(plan.raw_path, plan.path)])
    assert n > 0 and scene.ngeom == n
    small = mujoco.MjvScene(demo_engine.model, maxgeom=5)  # full scene: stops quietly
    assert draw_planned_paths(small, p.kin.fk, [(plan.raw_path, plan.path)]) == 5


def test_checker_matches_exhaustive_distance_ground_truth(demo_engine):
    """Random configurations: the checker flags a robot-environment violation exactly when
    the true distance (mj_geomDistance over every collidable robot/environment geom pair,
    minus the excluded base-table pair) is below the clearance. Guards against MuJoCo's
    midphase dropping pairs inside the margin (it missed ~20% of near misses when enabled)."""
    c = checker_for(demo_engine, clearance=0.02)
    m, d = c.model, c.data
    rob = [g for g in range(m.ngeom) if c._is_robot_geom[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
    env = [g for g in range(m.ngeom) if not c._is_robot_geom[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
    excluded = (m.body("link0").id, m.body("table").id)
    pairs = [(a, b) for a in rob for b in env
             if ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
             and (m.geom_bodyid[a], m.geom_bodyid[b]) != excluded]
    rng = np.random.default_rng(0)
    near = 0
    for _ in range(600):
        q = rng.uniform(c.q_min, c.q_max)
        c._collide(q)
        flagged = any(c._is_robot_geom[k.geom1] != c._is_robot_geom[k.geom2] for k in d.contact[: d.ncon])
        true_dist = min(mujoco.mj_geomDistance(m, d, a, b, 0.05, np.zeros(6)) for a, b in pairs)
        assert flagged == (true_dist < c.clearance), f"q={q.tolist()} true distance {true_dist:.4f}"
        near += 0.0 < true_dist < 0.03
    assert near >= 5  # the sample does contain near misses
