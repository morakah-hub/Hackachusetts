"""v0.5.2: exact distances for box-box robot-environment pairs in the planner's collision checker.

MuJoCo 3.16's box-box routine misses margin contacts and misreports contact.dist (v0.5.1 finding,
tools/audit_clearance.py). The checker now asks mj_geomDistance for box-box robot/obstacle pairs.
The Panda configurations below were found by random search and local descent around planned wall
paths and IK poses above the table (docs/research/v0.5.2-worklog.md); each test re-measures the
distances it relies on instead of trusting the comments.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker  # noqa: E402
from planning.motion_planner import MotionPlanner, PlanningSettings  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.robustness import ExactClearance  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

START = np.array([0.246, 0.621, 0.211, -1.791, 0.089, 1.756, -0.785])  # configs/planning_demo.yaml
TARGET = np.array([0.45, -0.25, 0.50])

Q_PAD_WALL_MISS = [-0.06187104694463297, 0.458460114863153, 0.09503598278026311, -1.5330871572764975,
                   -0.28344467690896263, 1.6525045928017015, -0.48979142894645794]  # a pad ~17.8 mm from the wall
Q_PAD_WALL_SAFE = [0.0007, 0.4447, -0.0456, -1.5523, -0.1747, 1.876, -0.3916]    # a pad ~20.5 mm from the wall
Q_PAD_WALL_HIT = [0.2598, 0.4404, -0.2928, -1.6577, -0.1207, 2.1066, -0.8349]    # left pad ~3 mm inside the wall
Q_PAD_TABLE_SAFE = [0.2988, 0.7298, 0.205, -1.7312, 0.1762, 1.9789, -0.8216]     # left pad ~22 mm above the table
Q_PAD_TABLE_HIT = [0.2376, 0.7728, 0.2327, -1.7444, 0.1208, 1.8042, -0.7676]     # left pad ~2.8 mm inside the table top


@pytest.fixture(scope="module")
def engine():
    return SimulationEngine("planning_demo")


@pytest.fixture(scope="module")
def new(engine):
    return CollisionChecker.from_engine(engine)


@pytest.fixture(scope="module")
def old(engine):
    return CollisionChecker.from_engine(engine, exact_box_box=False)


@pytest.fixture(scope="module")
def exact(engine):
    return ExactClearance(engine)


def box_box_distance(ex: ExactClearance, q, env_geom: str) -> float:
    """Smallest mj_geomDistance between a robot box geom and the named environment geom at q."""
    ex.distance(np.asarray(q, float))  # kinematics at q
    m, d = ex.model, ex.data
    g_env = m.geom(env_geom).id
    return min(mujoco.mj_geomDistance(m, d, a, b, 0.1, None) for a, b in ex.pairs
               if b == g_env and m.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX)


def other_distance(ex: ExactClearance, q) -> float:
    """Smallest robot-environment distance over every pair that is not box-box."""
    ex.distance(np.asarray(q, float))
    m, d = ex.model, ex.data
    box = mujoco.mjtGeom.mjGEOM_BOX
    return min(mujoco.mj_geomDistance(m, d, a, b, 0.1, None) for a, b in ex.pairs
               if not (m.geom_type[a] == box and m.geom_type[b] == box))


# --- which pairs are exact ---------------------------------------------------------


def test_exact_pairs_are_the_fingertip_boxes_against_box_obstacles(new, old):
    m = new.model
    robot = {m.body(m.geom_bodyid[g]).name for g in new._ex_rob}
    env = {m.geom(g).name for g in new._ex_env}
    assert robot == {"left_finger", "right_finger"} and len(new._ex_rob) == 10
    assert env == {"table_top", "table_leg1", "table_leg2", "table_leg3", "table_leg4", "wall"}
    assert new._ex_pair.all()  # every fingertip box against every box obstacle
    assert all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in [*new._ex_rob, *new._ex_env])
    assert old._ex_pair.size == 0 and not old._ex_skip.any()


def test_planning_settings_carry_the_flag(engine):
    assert PlanningSettings().exact_box_box
    assert MotionPlanner.from_engine(engine).checker.exact_box_box
    off = PlanningSettings.from_config({"exact_box_box": False})
    assert not MotionPlanner.from_engine(engine, off).checker.exact_box_box
    cfg = load_config("configs/planning_demo.yaml")
    assert cfg["task"]["planning"]["exact_box_box"] is True
    cfg["task"]["targets"] = cfg["task"]["targets"][:1]
    assert SimulationRunner(cfg).planner.checker.exact_box_box


# --- fingertip vs wall -------------------------------------------------------------


def test_fingertip_wall_near_miss_is_rejected(new, old, exact):
    d_pad = box_box_distance(exact, Q_PAD_WALL_MISS, "wall")
    assert 0.015 < d_pad < 0.0185
    assert other_distance(exact, Q_PAD_WALL_MISS) >= 0.0204  # only the fingertip box is too close
    assert old.check(Q_PAD_WALL_MISS).valid  # v0.5: MuJoCo generated no contact for it
    rep = new.check(Q_PAD_WALL_MISS)
    assert not rep.valid and rep.reason.startswith("within clearance") and "finger" in rep.reason and "wall" in rep.reason
    assert new.min_distance(Q_PAD_WALL_MISS) == pytest.approx(d_pad, abs=1e-9)


def test_fingertip_just_outside_clearance_is_accepted(new, old, exact):
    d_pad = box_box_distance(exact, Q_PAD_WALL_SAFE, "wall")
    assert 0.0203 < d_pad < 0.0215
    assert exact.distance(np.asarray(Q_PAD_WALL_SAFE)) >= 0.02
    assert new.check(Q_PAD_WALL_SAFE).valid
    assert not old.check(Q_PAD_WALL_SAFE).valid  # v0.5 false alarm: contact.dist reported < 20 mm


def test_fingertip_wall_collision_is_rejected(new, old, exact):
    assert box_box_distance(exact, Q_PAD_WALL_HIT, "wall") < -0.001
    for c in (new, old):  # penetration was always detected
        rep = c.check(Q_PAD_WALL_HIT)
        assert not rep.valid and rep.reason.startswith("collision")
    assert new.min_distance(Q_PAD_WALL_HIT) < -0.001


# --- fingertip vs table ------------------------------------------------------------


def test_fingertip_above_table_outside_clearance_is_accepted(new, exact):
    d_pad = box_box_distance(exact, Q_PAD_TABLE_SAFE, "table_top")
    assert 0.0205 < d_pad < 0.024
    assert exact.distance(np.asarray(Q_PAD_TABLE_SAFE)) >= 0.02
    assert new.check(Q_PAD_TABLE_SAFE).valid


def test_fingertip_table_collision_is_rejected(new, exact):
    assert box_box_distance(exact, Q_PAD_TABLE_HIT, "table_top") < -0.001
    rep = new.check(Q_PAD_TABLE_HIT)
    assert not rep.valid and rep.reason.startswith("collision") and "table" in rep.reason


PAIR_XML = """<mujoco><option><flag midphase="disable"/></option><worldbody>
<geom name="obstacle" type="box" size="{size}"/>
<body name="pad" pos="0 0 1"><freejoint/><geom name="pad" type="box" size="0.0085 0.004 0.0085"/></body>
</worldbody></mujoco>"""  # the Panda fingertip pad's size


def pair_checker(obstacle_size: str, exact_box_box: bool = True) -> CollisionChecker:
    """Checker for an isolated pad (free body, q = its 7 qpos) next to a fixed box obstacle."""
    m = mujoco.MjModel.from_xml_string(PAIR_XML.format(size=obstacle_size))
    return CollisionChecker(m, np.zeros(m.nq), np.arange(7), np.arange(6), [m.body("pad").id],
                            np.tile([-10.0, 10.0], (7, 1)), clearance=0.02, limit_margin=0.0,
                            exact_box_box=exact_box_box)


@pytest.mark.parametrize("obstacle,size,q", [
    ("table top", "0.55 0.5 0.02", [-0.23622, 0.51011, 0.03059, 0.34661, -0.07087, 0.9184, -0.17717]),
    ("wall", "0.275 0.02 0.13", [-0.11007, 0.032, -0.13742, -0.79212, 0.14083, -0.24559, 0.54074]),
])
def test_isolated_fingertip_near_miss_missed_by_mujoco_caught_by_checker(obstacle, size, q):
    c = pair_checker(size)
    c._collide(np.asarray(q))
    true = mujoco.mj_geomDistance(c.model, c.data, 0, 1, 0.1, None)
    assert 0 < true < 0.003, obstacle
    assert c.data.ncon == 0  # MuJoCo 3.16: no margin contact, although the margin is 20 mm
    assert pair_checker(size, exact_box_box=False).check(q).valid  # so the v0.5 checker accepts it
    rep = c.check(q)
    assert not rep.valid and "within clearance" in rep.reason
    assert c.min_distance(q) == pytest.approx(true, abs=1e-9)


# --- whole robot vs ground truth near obstacles -------------------------------------


def test_checker_matches_exhaustive_distance_near_obstacles(engine, new, old, exact):
    """Configurations jittered around planned wall paths (many fingertip near misses): the
    corrected checker agrees with mj_geomDistance over every robot/environment pair (plus
    penetration-only self-collision) at every configuration; the v0.5 checker does not."""
    selfc = CollisionChecker.from_engine(engine, clearance=0.0, exact_box_box=False)

    def self_collision(q):
        selfc._collide(q)
        d = selfc.data
        return any(selfc._is_robot_geom[c.geom1] and selfc._is_robot_geom[c.geom2] for c in d.contact[: d.ncon])

    planner = MotionPlanner.from_engine(engine)
    base = []
    for seed in (0, 1):
        path = planner.plan_to_position(TARGET, START, seed=seed).path
        for a, b in zip(path[:-1], path[1:]):
            base += [a + (b - a) * t for t in np.linspace(0, 1, 15)]
    rng = np.random.default_rng(0)
    v05_wrong = pad_limited = 0
    for k in range(500):
        q = np.clip(base[k % len(base)] + rng.normal(0, 0.05, 7), new.q_min, new.q_max)
        truth = exact.distance(q) < new.clearance or self_collision(q)
        assert new.in_collision(q) == truth, f"q={q.tolist()}"
        v05_wrong += old.in_collision(q) != truth
        d_box = min(box_box_distance(exact, q, g) for g in ("wall", "table_top"))
        pad_limited += 0.0 < d_box < 0.03 and d_box <= other_distance(exact, q)
    assert pad_limited >= 10  # the sample does exercise fingertip boxes near the obstacles
    assert v05_wrong >= 1


# --- edges ---------------------------------------------------------------------------


def collision_vertices(c: CollisionChecker):
    """(geom id, local vertices) for every collidable robot geom (box corners, mesh vertices)."""
    m, out = c.model, []
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    for g in range(m.ngeom):
        if not c._is_robot_geom[g] or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            out.append((g, signs * m.geom_size[g]))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]
            out.append((g, m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]].copy()))
    return out


def world_vertices(c: CollisionChecker, verts, q) -> np.ndarray:
    d = c.data
    d.qpos[:] = c.base_qpos
    d.qpos[c.qpos_adr] = q
    mujoco.mj_kinematics(c.model, d)
    return np.vstack([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in verts])


def test_edge_samples_bound_how_far_any_robot_point_moves(new):
    """Between consecutive edge samples no vertex of the robot's collision geometry travels
    (path length, 8 sub-steps) more than cartesian_step: the lever-arm bound holds."""
    c = CollisionChecker.from_engine(SimulationEngine("planning_demo"))  # private: we move its data
    verts = collision_vertices(c)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(12):
        q0 = rng.uniform(c.q_min, c.q_max)
        q1 = np.clip(q0 + rng.uniform(-0.8, 0.8, 7), c.q_min, c.q_max)
        n = c.edge_samples(q0, q1)
        for i in range(n):
            pts = [world_vertices(c, verts, q0 + (q1 - q0) * ((i + s / 8) / n)) for s in range(9)]
            travel = sum(np.linalg.norm(b - a, axis=1) for a, b in zip(pts[:-1], pts[1:]))
            worst = max(worst, float(travel.max()))
    assert worst <= c.cartesian_step + 1e-9
    assert worst > 0.2 * c.cartesian_step  # the bound is conservative but not vacuous


def test_planned_edges_keep_clearance_between_samples(engine, new, exact):
    """Paths planned with the corrected checker: the exact distance is >= clearance at every
    configuration the checker evaluated, and >= clearance - cartesian_step / 2 (the derived
    bound) at dense intermediate configurations (every 1 mm of robot motion)."""
    planner = MotionPlanner.from_engine(engine)
    bound = new.clearance - new.cartesian_step / 2
    for seed in (0, 3):
        plan = planner.plan_to_position(TARGET, START, seed=seed)
        assert plan.success
        at_samples = exact.at_checked_samples(plan.path, planner.checker)
        dense = exact.path(plan.path, step=0.001)
        assert at_samples >= new.clearance - 1e-9
        assert bound <= dense <= at_samples + 1e-12
