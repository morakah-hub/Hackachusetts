"""v0.5.1: wall-geometry overrides, replanning under changed known geometry, video writer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.engine import SimulationEngine, environment_path  # noqa: E402
from simulation.robustness import load_experiment, plan_episode, reference_plans, variant_engine, wall_box  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

EXP = load_experiment()


def test_overrides_change_only_the_compiled_model():
    xml_before = environment_path("planning_demo").read_text(encoding="utf-8")
    nominal = SimulationEngine("planning_demo")
    taller = variant_engine(EXP, "taller")
    import mujoco

    for e in (nominal, taller):
        mujoco.mj_forward(e.model, e.data)
    assert wall_box(nominal)["wall_z_max"] == pytest.approx(0.66)
    assert wall_box(taller)["wall_z_max"] == pytest.approx(0.76)
    assert wall_box(taller)["wall_x_min"] == pytest.approx(0.25)
    assert environment_path("planning_demo").read_text(encoding="utf-8") == xml_before


def test_override_errors_are_explicit():
    with pytest.raises(ValueError, match="no geom named"):
        SimulationEngine("planning_demo", overrides={"geoms": {"nope": {"size": [1, 1, 1]}}})
    with pytest.raises(ValueError, match="unsupported keys"):
        SimulationEngine("planning_demo", overrides={"geoms": {"wall": {"rgba": [1, 0, 0, 1]}}})
    with pytest.raises(ValueError, match="expected 3 values"):
        SimulationEngine("planning_demo", overrides={"bodies": {"wall": {"pos": [0.5, 0.0]}}})


@pytest.mark.parametrize("variant", list(EXP["variants"]))
def test_variant_start_is_valid_and_direct_move_blocked(variant):
    from planning.motion_planner import MotionPlanner, PlanningSettings

    planner = MotionPlanner.from_engine(variant_engine(EXP, variant), PlanningSettings.from_config(EXP["planning"]))
    start = np.asarray(EXP["start"], float)
    assert planner.checker.check(start).valid
    # The experiment is about obstacle avoidance: the straight joint move to the target must be blocked.
    q_goal = planner.plan_to_position(np.asarray(EXP["target"], float), start, seed=0).q
    assert q_goal is not None
    assert not planner.checker.edge_valid(start, q_goal)


def test_old_plan_rejected_and_replanned_in_taller_wall():
    old = reference_plans(EXP, [0])
    row = plan_episode(EXP, "taller", 0, variant_engine(EXP, "taller"), old[0])
    assert row["old_plan_valid"] is False
    assert row["old_plan_min_distance_mm"] < 0  # the old path would penetrate the raised wall
    assert row["success"] and row["path_revalidated"]
    assert row["path_min_clearance_mm"] >= 15.0


def test_runner_plans_in_the_overridden_scene():
    cfg = load_config("configs/planning_demo.yaml")
    cfg["task"]["targets"] = cfg["task"]["targets"][:1]
    cfg["environment_overrides"] = EXP["variants"]["taller"]["overrides"]
    runner = SimulationRunner(cfg)
    m = runner.planner.checker.model
    assert m.geom_size[m.geom("wall").id][2] == pytest.approx(0.18)
    plan = runner.task.plans[0]
    assert plan.reachable and len(plan.path) > 2
    ok, _ = runner.planner.checker.path_valid(plan.path)
    assert ok


def test_webp_stream_writes_animation(tmp_path):
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from render_demo_video import WebPStream

    out = tmp_path / "t.webp"
    w = WebPStream(out, (32, 24), fps=10)
    for c in (0, 128, 255):
        w.add(Image.new("RGB", (32, 24), (c, 0, 0)))
    w.close()
    im = Image.open(out)
    assert im.n_frames == 3 and im.size == (32, 24)


def test_exact_clearance_measures_known_cases():
    from simulation.robustness import ExactClearance

    nominal = ExactClearance(SimulationEngine("planning_demo"))
    assert nominal.distance(np.asarray(EXP["start"], float)) > 0.02
    old = reference_plans(EXP, [0])[0]
    assert ExactClearance(variant_engine(EXP, "taller")).path(old) < 0  # old path penetrates the raised wall


BOX_PAIR = """<mujoco><worldbody>
<geom name="wall" type="box" size="0.275 0.02 0.13" margin="0.02"/>
<body name="pad" pos="0 0.1 0"><freejoint/><geom type="box" size="0.01 0.003 0.01"/></body>
</worldbody></mujoco>"""


def _box_pair(pos, quat):
    import mujoco

    m = mujoco.MjModel.from_xml_string(BOX_PAIR)
    d = mujoco.MjData(m)
    q = np.asarray(quat, float)
    d.qpos[:3], d.qpos[3:7] = pos, q / np.linalg.norm(q)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    return m, d, mujoco.mj_geomDistance(m, d, 0, 1, 0.1, None)


def test_box_box_penetration_is_always_detected():
    """MuJoCo's box-box routine never misses actual penetration (random poses, seed 1)."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(2000):
        quat = rng.normal(size=4)
        _, d, true = _box_pair([rng.uniform(-0.3, 0.3), rng.uniform(0.0, 0.04), rng.uniform(-0.15, 0.15)], quat)
        if true < 0:
            hits += 1
            assert d.ncon > 0
    assert hits > 100


NEAR_MISS_POSE = ([-0.0115, 0.0325, -0.1393], [-0.7845, -0.4828, -0.2754, -0.275])  # pad 1.25 mm from the wall


def test_mujoco_box_box_still_misses_this_margin_contact():
    """The MuJoCo 3.16 behavior the v0.5.2 checker works around (v0.5.1 finding): this
    fingertip-pad-sized box is 1.25 mm from the wall yet no contact is generated under a
    20 mm margin. If this starts failing after a MuJoCo upgrade, re-run tools/audit_clearance.py
    before relying on box-box margin contacts; the checker does not need them."""
    _, d, true = _box_pair(*NEAR_MISS_POSE)
    assert 0 < true < 0.005
    assert d.ncon == 0


def test_checker_rejects_box_box_near_miss_inside_the_margin():
    """Was the strict xfail `test_box_box_margin_contact_is_generated_inside_the_margin` in
    v0.5.1 (the v0.5 checker relied on that contact). The v0.5.2 checker answers box-box
    robot/environment pairs with mj_geomDistance, so the same pose is rejected."""
    import mujoco

    from planning.collision import CollisionChecker

    m = mujoco.MjModel.from_xml_string(BOX_PAIR)
    q = np.r_[NEAR_MISS_POSE[0], np.asarray(NEAR_MISS_POSE[1]) / np.linalg.norm(NEAR_MISS_POSE[1])]
    args = (m, np.zeros(m.nq), np.arange(7), np.arange(6), [m.body("pad").id], np.tile([-10.0, 10.0], (7, 1)))
    rep = CollisionChecker(*args, clearance=0.02, limit_margin=0.0).check(q)
    assert not rep.valid and rep.reason.startswith("within clearance (1.2 mm < 20 mm)")
    assert CollisionChecker(*args, clearance=0.02, limit_margin=0.0, exact_box_box=False).check(q).valid  # v0.5
