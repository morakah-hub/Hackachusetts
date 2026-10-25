"""v0.6 known-world pick-and-place: pose IK, Cartesian lines, model boundary, checker, episodes."""

import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.dynamics import ArmInverseDynamics  # noqa: E402
from controllers.prediction import MuJoCoPredictionModel  # noqa: E402
from planning.collision import CollisionChecker  # noqa: E402
from robots.panda import Panda  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.kinematics import ArmKinematics, rotation_angle, rotation_error  # noqa: E402
from simulation.manipulation import run_case  # noqa: E402
from simulation.runner import load_config  # noqa: E402
from tasks.pick_place import CartesianSegment, top_down_rotation  # noqa: E402
from tasks.trajectory import KinematicLimits  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "pick_place.yaml"


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics(SimulationEngine("pick_place", Panda(actuation="torque")))


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG)


# ------------------------------------------------------------------ pose IK


def test_rotational_jacobian_matches_finite_differences(kin):
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = kin.q_rest + rng.normal(0, 0.4, 7)
        J = kin.jacobian6(q)
        _, R0 = kin.fk_pose(q)
        eps = 1e-6
        for j in range(7):
            dq = np.zeros(7)
            dq[j] = eps
            _, R1 = kin.fk_pose(q + dq)
            assert np.allclose(rotation_error(R1, R0) / eps, J[3:, j], atol=1e-6)


def test_rotation_error_is_world_frame_rotation_vector():
    R = top_down_rotation(0.3)
    axis = np.array([0.2, -0.5, 0.8]) / np.linalg.norm([0.2, -0.5, 0.8])
    quat = np.r_[np.cos(0.35), np.sin(0.35) * axis]
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, quat)
    R_t = Rq.reshape(3, 3) @ R  # rotate by 0.7 rad about `axis` in the world frame
    assert np.allclose(rotation_error(R_t, R), 0.7 * axis, atol=1e-12)


def test_pose_ik_converges_on_reachable_poses_and_rejects_unreachable(kin):
    rng = np.random.default_rng(0)
    solved = 0
    for _ in range(20):
        q_true = rng.uniform(kin.q_min, kin.q_max)
        p, R = kin.fk_pose(q_true)
        for seed in (kin.q_rest, q_true + rng.normal(0, 0.3, 7)):
            res = kin.solve_pose_ik(p, R, seed, max_iter=500)
            if res.success:
                p1, R1 = kin.fk_pose(res.q)
                assert np.linalg.norm(p1 - p) < 1e-4 and rotation_angle(R1, R) < 1e-3
                solved += 1
                break
    assert solved >= 18
    p, R = kin.fk_pose(kin.q_rest)
    bad = kin.solve_pose_ik(p + np.array([1.5, 0.0, 0.0]), R, kin.q_rest)
    assert not bad.success and "did not converge" in bad.reason
    # Reachable position, but an orientation the wrist cannot take there (gripper pointing up at table height).
    up = top_down_rotation(0.0) @ np.diag([1.0, -1.0, -1.0])
    bad = kin.solve_pose_ik(np.array([0.5, 0.0, 0.42]), up, kin.q_rest)
    assert not bad.success


def test_position_ik_api_unchanged(kin):
    res = kin.solve_ik(np.array([0.5, 0.1, 0.6]), kin.q_rest)
    assert res.success and res.rotation_error == 0.0


# ------------------------------------------------------------------ Cartesian lines


def test_cartesian_descent_is_straight_with_constant_orientation(kin):
    R = top_down_rotation(0.0)
    limits = KinematicLimits(np.asarray(Panda().velocity_limits) * 0.9, np.asarray(Panda().acceleration_limits) * 0.5)
    start = kin.solve_pose_ik(np.array([0.5, 0.2, 0.545]), R, kin.q_rest)
    seg = CartesianSegment.plan(kin, start.q, np.array([0.5, 0.2, 0.445]), R, 1.5, limits)
    v = seg.validation
    assert v.max_position_deviation < 1e-4 and v.max_rotation_error < np.radians(0.1)
    q0, qd0, _ = seg.evaluate(0.0)
    q1, qd1, _ = seg.evaluate(seg.duration)
    assert np.allclose(q0, start.q) and np.allclose(qd0, 0) and np.allclose(qd1, 0)
    p_end, _ = kin.fk_pose(q1)
    assert np.linalg.norm(p_end - [0.5, 0.2, 0.445]) < 1e-4
    # Reference velocity is the derivative of the reference position.
    t, h = 0.6, 1e-5
    assert np.allclose((seg.evaluate(t + h)[0] - seg.evaluate(t - h)[0]) / (2 * h), seg.evaluate(t)[1], atol=1e-6)


# ------------------------------------------------------------------ controller model boundary


def _engine(mass, contacts):
    e = SimulationEngine("pick_place", Panda(actuation="torque"),
                         {"geoms": {"object_geom": {"mass": mass}}})
    if not contacts:
        e.controller_model = e.contact_free_model({"object": 0.4})
    return e


def _grasp_state(e, depth, shift=0.0, at=(0.5, 0.0, 0.6)):
    """Arm at a lifted top-down pose, fingers pressed `depth` m into the object (contacts active);
    `shift` moves the object along the finger axis (deeper on one side, shallower on the other)."""
    kin = ArmKinematics(e)
    res = kin.solve_pose_ik(np.array(at), top_down_rotation(0.0), kin.q_rest)
    d, m = e.data, e.model
    d.qpos[e.arm_qpos_adr] = res.q
    fq = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_joint1", "finger_joint2")]
    d.qpos[fq] = 0.025 - depth
    mujoco.mj_kinematics(m, d)
    obj = m.jnt_qposadr[m.joint("object_free").id]
    d.qpos[obj:obj + 3] = d.site_xpos[e.ee_site_id] - np.array([0.0, 0.0, 0.015]) + np.array([shift, 0.0, 0.0])
    d.qpos[obj + 3:obj + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    return res.q


def _controller_view(e, q):
    dyn = ArmInverseDynamics(e.controller_model, e.data, e.arm_qpos_adr, e.arm_dof_adr)
    pred = MuJoCoPredictionModel(e.controller_model, e.data, e.arm_qpos_adr, e.arm_dof_adr, e.arm_actuator_ids, 10)
    qd, qdd = np.full(7, 0.1), np.full(7, 0.5)
    lin = pred.linearize(q, qd, dyn.gravity(q))
    return np.r_[dyn.torque(q, qd, qdd), dyn.gravity(q), pred.gravity_torque(q), lin.c]


def test_payload_mass_and_contact_state_do_not_reach_the_controller_model():
    views = []
    for mass in (0.15, 1.15):
        for shift in (0.0, 0.0004):
            e = _engine(mass, contacts=False)
            q = _grasp_state(e, 0.0005, shift)
            assert e.data.ncon > 0  # the plant has finger-object contacts in this state
            assert e.controller_model.body_mass[e.model.body("object").id] == pytest.approx(0.4)
            assert e.controller_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
            views.append(_controller_view(e, q))
    for v in views[1:]:
        assert np.allclose(v, views[0], atol=1e-9)


def test_with_contacts_enabled_the_controller_would_see_the_grip():
    """Control for the boundary test: the default controller model (contacts on) reads the contact state."""
    a, b = _engine(0.4, contacts=True), _engine(0.4, contacts=True)
    qa, qb = _grasp_state(a, 0.0005, 0.0), _grasp_state(b, 0.0005, 0.0004)
    assert not np.allclose(_controller_view(a, qa), _controller_view(b, qb), atol=1e-6)


# ------------------------------------------------------------------ collision checker (v0.6 additions)


def test_checker_attachment_allowed_pairs_and_lever_arms():
    e = SimulationEngine("pick_place_obstacle", Panda(actuation="torque"))
    c = CollisionChecker.from_engine(e)
    base_arms = c.lever_arms.copy()
    _grasp_state(e, 0.001, at=(0.5, 0.25, 0.7))
    c.set_fixed_qpos(e.data.qpos)
    q = e.arm_qpos()
    fingers, obj = c.body_geoms("left_finger", "right_finger"), c.body_geoms("object")
    assert not c.check(q).valid  # fingers in contact with the object
    c.allow(fingers, obj, -np.inf)
    assert c.check(q).valid
    c.attach("object", "hand")
    assert np.all(c.lever_arms >= base_arms) and c.lever_arms[0] > base_arms[0]
    # The attached object follows the hand: lowering the arm into the obstacle is now a collision of the object.
    kin = ArmKinematics(e)
    low = kin.solve_pose_ik(np.array([0.5, 0.0, 0.62]), top_down_rotation(0.0), q).q
    rep = c.check(low)
    assert not rep.valid and any("object" in (a, b) for a, b, _ in rep.contacts)
    c.detach()
    c.clear_allowed()
    assert np.allclose(c.lever_arms, base_arms)
    assert not c.check(q).valid


# ------------------------------------------------------------------ episodes


def test_episode_succeeds_with_ctc_and_observer(cfg):
    r = run_case(cfg, cfg["cases"][0])
    assert r["success"], r["failure_reason"]
    assert r["lift_height_mm"] >= 50 and r["placement_error_mm"] <= 10 and r["forbidden_contact_steps"] == 0
    assert [p["phase"] for p in r["phases"]] == ["approach", "descend", "close", "lift", "hold", "transport",
                                                 "lower", "release", "retreat", "verify"]


def test_episode_around_obstacle_succeeds_with_torque_mpc_without_fallback(cfg):
    c = copy.deepcopy(cfg)
    c["controller"]["type"] = "torque_mpc"
    r = run_case(c, cfg["cases"][3])
    assert r["success"], r["failure_reason"]
    assert r["mpc"]["fallback_steps"] == 0 and r["min_clearance_object_env_mm"] > 15.0


@pytest.mark.parametrize("fault, mass, stage", [
    ({"skip_close": True}, 0.4, "close"),
    ({"robot": {"gripper_stiffness": 100.0, "gripper_damping": 10.0}}, 1.0, "lift"),
    ({"place_offset": [0.03, 0.0]}, 0.4, "verify"),
    ({"release_during_transport": True}, 0.4, "transport"),
])
def test_negative_controls_are_rejected_for_the_intended_reason(cfg, fault, mass, stage):
    r = run_case(cfg, dict(cfg["cases"][0], fault=fault, mass=mass))
    assert not r["success"] and r["failure_stage"] == stage, (r["failure_stage"], r["failure_reason"])
