"""Known-world pick-and-place episodes (v0.6): supervisor, contact monitor and verifier.

One episode = one object pose, one placement target, one object mass, one arm controller:

    approach -> descend -> close -> lift -> hold -> transport -> lower -> release -> retreat -> verify

The supervisor is an event-based state machine (`_program`, a generator: every `yield` is one
physics step). Each phase starts a motion or a gripper command and ends on *measured*
conditions (arm at rest, finger width, object height, object pose relative to the hand) or
on a bounded timeout, which ends the episode with the phase and a reason. The arm reference is
built one motion at a time (tasks/pick_place.py); the arm controller (CTC or torque MPC, the
existing classes) tracks it every physics step exactly as in the other tasks, and the gripper
actuator is commanded separately (engine.set_gripper), never through the arm controller.

Known world. The object's initial pose, size and the obstacles are taken from the simulator
(ground truth) when a phase is planned; there is no perception. The object is a free body
moved only by contact and friction (mj_step): nothing here writes its pose, and no weld or
equality attaches it to the hand. A rigid attachment exists only inside the planner's private
collision model, to predict where the carried object will be (planning/collision.py).

Model boundary. The controllers get `engine.contact_free_model()`: contacts disabled and the
object's mass fixed at a nominal value, so the payload's weight never reaches their inverse
dynamics, gravity torque or prediction through copied contact state (tests/test_pick_place.py).
The payload acts on the arm only in the plant; the disturbance observer, when enabled, sees it
as an unexplained joint torque from measured motion.

Measurement. After every physics step the kinematics are refreshed (mj_kinematics; read-only
for the dynamics) so that poses used for decisions belong to the current state. The contacts
used for the monitor are those MuJoCo applied in that step. Contacts and joint/torque limit
violations are counted at every physics step; clearance is measured every `clearance_every`
steps with mj_geomDistance.
"""

import copy
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from planning.motion_planner import MotionPlanner, PlanningSettings
from robots.panda import Panda
from simulation.engine import SimulationEngine
from simulation.kinematics import ArmKinematics, rotation_angle
from simulation.runner import CONTROLLER_ACTUATION, _merge, build_controller, controller_kind
from controllers.dynamics import ArmInverseDynamics
from tasks.pick_place import (CartesianSegment, JointPathSegment, ManipulationReference, PathInfeasible,
                              carry_arc_points, top_down_rotation)
from tasks.trajectory import KinematicLimits, TrajectoryInfeasible

PHASES = ("approach", "descend", "close", "lift", "hold", "transport", "lower", "release", "retreat", "verify")
# Entered only when configured (pick_place.return_home, studio): a planned move back to a rest posture.
OPTIONAL_PHASES = ("home",)
# Phases in which finger-object contact is intended (grasping and carrying).
GRASP_PHASES = {"close", "lift", "hold", "transport", "lower", "release"}
# Phases in which the object must be off the table (support contact there means it was dropped or dragged).
AIRBORNE_PHASES = {"hold", "transport"}

# Contact categories of a geom (by body).
ENV, ROBOT, FINGER, OBJECT = 0, 1, 2, 3


class EpisodeFailure(Exception):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage, self.reason = stage, reason


@dataclass
class SuccessCriteria:
    """Declared before evaluation (configs/pick_place.yaml `pick_place.success`)."""

    lift_height: float = 0.05          # m above the initial height, measured
    hold_time: float = 1.0             # s held in the air after the lift
    hold_slip: float = 0.005           # m: max object displacement in the hand frame while holding
    drop_slip: float = 0.010           # m: displacement in the hand frame that counts as a drop (transport/lower)
    place_tolerance: float = 0.010     # m: final 3-D distance of the object center from the target
    tilt_tolerance_deg: float = 10.0   # final tilt of the object's z axis from vertical
    settle_speed: float = 0.005        # m/s: the object counts as settled below this speed ...
    settle_time: float = 0.2           # s: ... held for this long
    width_tolerance: float = 0.004     # m: closed finger width vs object width (grasp check)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SuccessCriteria":
        return cls(**(cfg or {}))


def case_overrides(pp: dict, case: dict) -> dict:
    """environment_overrides for one case: object pose, placement marker, object mass/friction.

    Studio additions (optional case keys): "object_pose" {"pos": [x, y, z], "quat": [w, x, y, z]} is the
    object's measured 3-D pose (e.g. resting on another object) instead of (x, y) on the table, and
    "place_z" is the height of the object's center at the placement (e.g. on top of another object)."""
    hz = pp["object"]["half_size"][2]
    table_z = pp["table_height"]
    pose = case.get("object_pose")
    if pose is not None:
        obj = {"pos": [float(v) for v in pose["pos"]], "quat": [float(v) for v in pose.get("quat", (1, 0, 0, 0))]}
    else:
        obj = {"pos": [case["object"][0], case["object"][1], table_z + hz]}
    marker = {"pos": [case["place"][0], case["place"][1], float(case.get("place_z", table_z + hz))]}
    geom = {"mass": float(case.get("mass", pp["object"].get("mass", 0.4)))}
    if "friction" in case:
        geom["friction"] = list(case["friction"])
    return {"bodies": {pp["object"]["body"]: obj, "place_marker": marker}, "geoms": {pp["object"]["geom"]: geom}}


class PickPlaceEpisode:
    def __init__(self, cfg: dict, case: dict, record_every: int = 0):
        self.cfg, self.case = cfg, case
        pp = self.pp = cfg["pick_place"]
        self.criteria = SuccessCriteria.from_config(pp.get("success"))
        self.fault = case.get("fault") or {}
        ctrl_cfg = cfg["controller"]
        kind = controller_kind(ctrl_cfg)
        rob = {**(cfg.get("robot") or {}), **(self.fault.get("robot") or {})}  # weak_grip negative control
        overrides = _merge(cfg.get("environment_overrides") or {}, case_overrides(pp, case))
        self.engine = e = SimulationEngine(case.get("environment", cfg["environment"]), Panda(
            actuation=CONTROLLER_ACTUATION[kind], gripper_stiffness=rob.get("gripper_stiffness"),
            gripper_damping=rob.get("gripper_damping")), overrides)
        physics = cfg.get("physics") or {}  # contact-solver options of the plant (and its copies)
        for key in ("cone", "impratio", "noslip_iterations", "timestep"):
            if key in physics:
                value = physics[key]
                if key == "cone":
                    value = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[value]
                setattr(e.model.opt, key, value)
        if "friction_scale" in physics:  # sensitivity check: sliding friction of every geom (plant only)
            e.model.geom_friction[:, 0] *= float(physics["friction_scale"])
        cm = cfg.get("controller_model") or {}
        if not cm.get("contacts", True):
            e.controller_model = e.contact_free_model(cm.get("nominal_masses"))
        m = e.model
        self.start_q = np.asarray(pp["start"], float)
        self.ref = ManipulationReference(self.start_q)
        self.controller = build_controller(ctrl_cfg, e, self.ref.preview)
        self.controller_name = kind
        e.reset(self.start_q)
        e.set_gripper(pp["gripper"]["open"])
        self.controller.reset()
        e.forward()

        self.kin = ArmKinematics(e, limit_margin=pp["planning"].get("joint_limit_margin", 0.05))
        settings = PlanningSettings.from_config({**pp["planning"], "seed": int(case.get("seed", 0))})
        self.planner = MotionPlanner.from_engine(e, settings, self.kin)
        self.checker = self.planner.checker
        traj = pp["trajectory"]
        r = e.robot
        self.limits = KinematicLimits(
            velocity=np.asarray(r.velocity_limits) * traj["velocity_scale"],
            acceleration=np.asarray(r.acceleration_limits) * traj["acceleration_scale"],
            jerk=np.asarray(r.jerk_limits) * traj["jerk_scale"], position=e.joint_limits)
        self.timing = {"profile": "min_jerk", "limits": self.limits}
        if traj.get("torque_scale") is not None:  # torque budget checked with the controller's (payload-free) model
            dyn = ArmInverseDynamics(e.controller_model, e.data, e.arm_qpos_adr, e.arm_dof_adr)
            self.timing["torque_fn"] = dyn.torque
            self.timing["torque_limits"] = traj["torque_scale"] * np.abs(m.actuator_forcerange[e.arm_actuator_ids]).max(axis=1)

        # Geometry bookkeeping.
        ob = pp["object"]
        self.obj_body = m.body(ob["body"]).id
        self.obj_geom = m.geom(ob["geom"]).id
        self.obj_half = np.asarray(ob["half_size"], float)
        self.hand = m.body("hand").id
        fj = [m.joint(n) for n in ("finger_joint1", "finger_joint2")]
        self.finger_qadr = np.array([m.jnt_qposadr[j.id] for j in fj])
        robot = set(e._robot_body_ids())
        fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self.category = np.full(m.nbody, ENV)
        for b in robot:
            self.category[b] = FINGER if b in fingers else ROBOT
        self.category[self.obj_body] = OBJECT
        # Support surfaces: the table, plus (studio) the bodies the object rests on when it is picked up or
        # placed (stacking). They get the table's rules: support contact is allowed except while the object
        # should be in the air, and the planner lets the carried object approach them while lifting/lowering.
        supports = {m.body(b).id for b in case.get("supports") or ()}
        self.table_geoms = {g for g in range(m.ngeom)
                            if m.body(m.geom_bodyid[g]).name == "table" or m.geom_bodyid[g] in supports}
        self.geom_cat = self.category[m.geom_bodyid]
        live = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        base = m.body(r.base_body).id
        self._robot_geoms = [g for g in range(m.ngeom) if live[g] and m.geom_bodyid[g] in robot and m.geom_bodyid[g] != base]
        self._env_geoms = [g for g in range(m.ngeom) if live[g] and self.geom_cat[g] == ENV and m.geom_bodyid[g] != 0]
        self.gripper_geoms = self.checker.body_geoms("hand", "left_finger", "right_finger")
        self.finger_geoms = self.checker.body_geoms("left_finger", "right_finger")
        self.hand_geoms = self.checker.body_geoms("hand")
        self.object_geoms = self.checker.body_geoms(ob["body"])
        self.table_geom_ids = np.array(sorted(self.table_geoms), dtype=int)

        # Logs and metrics.
        self.phase = "init"
        self.phase_log: list[dict] = []
        self.record_every = record_every
        self.trace: list[dict] = []
        self.forbidden_steps = 0
        self.forbidden_events: list[dict] = []
        self.steps = 0
        self.q_err_max = self.ee_err_max = 0.0
        self.q_err_sq = self.ee_err_sq = 0.0
        self.sat_steps = self.pos_limit_steps = self.vel_limit_steps = 0
        self.min_clear_robot = self.min_clear_object = np.inf
        self.clearance_every = int(pp.get("clearance_every", 5))
        self.metrics: dict = {}
        self.planning: list[dict] = []
        self.paths: list[dict] = []  # joint paths actually used (planner paths, line IK samples), for display
        self._scratch = mujoco.MjData(m)
        self.v_max = np.asarray(r.velocity_limits)
        # Optional (v0.7): seeded measurement noise on the controller's arm observations (as SimulationRunner),
        # and a hook called before every physics step. Supervisor decisions and metrics use the true state.
        noise = cfg.get("sensor_noise") or {}
        self._noise_std = (float(noise.get("position_std", 0.0)), float(noise.get("velocity_std", 0.0)))
        self._rng = np.random.default_rng(int(noise.get("seed", 0)))
        self.before_step = None
        self.last_observation = self.last_command = None
        self.tau_lim = np.abs(m.actuator_forcerange[e.arm_actuator_ids]).max(axis=1)

    # ------------------------------------------------------------------ measurements

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.engine.data
        return d.xpos[self.obj_body].copy(), d.xmat[self.obj_body].reshape(3, 3).copy()

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.engine.data
        return d.xpos[self.hand].copy(), d.xmat[self.hand].reshape(3, 3).copy()

    def object_in_hand(self) -> np.ndarray:
        p, _ = self.object_pose()
        h, R = self.hand_pose()
        return R.T @ (p - h)

    def finger_width(self) -> float:
        return float(self.engine.data.qpos[self.finger_qadr].sum())

    def object_speed(self) -> tuple[float, float]:
        d = self.engine.data
        jnt = self.engine.model.body_jntadr[self.obj_body]
        v = d.qvel[self.engine.model.jnt_dofadr[jnt]:self.engine.model.jnt_dofadr[jnt] + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def ee_fk(self, q: np.ndarray) -> np.ndarray:
        e, s = self.engine, self._scratch
        s.qpos[:] = e.data.qpos
        s.qpos[e.arm_qpos_adr] = q
        mujoco.mj_kinematics(e.model, s)
        return s.site_xpos[e.ee_site_id].copy()

    # ------------------------------------------------------------------ stepping

    def _step(self) -> None:
        e = self.engine
        t = e.time
        ref = self.ref.reference(t)
        if self.before_step is not None:  # e.g. an external force pulse on the plant (never seen by the controller)
            self.before_step(self)
        q, qd = e.arm_qpos(), e.arm_qvel()
        if self._noise_std[0] > 0 or self._noise_std[1] > 0:  # controller observations only; metrics use the truth
            q = q + self._rng.normal(0.0, self._noise_std[0], q.shape)
            qd = qd + self._rng.normal(0.0, self._noise_std[1], qd.shape)
        self.last_observation = (q, qd)
        ctrl = self.controller.compute(q, qd, ref.q, ref.qd, e.dt, t=t, qdd_ref=ref.qdd)
        self.last_command = np.asarray(ctrl, float).copy()
        fallback = bool(getattr(self.controller, "using_fallback", False))
        e.set_arm_command(ctrl)
        e.step()
        mujoco.mj_kinematics(e.model, e.data)  # fresh poses for measurements (no effect on the dynamics)
        self.steps += 1
        ph = self.phase_log[-1]
        ph["steps"] += 1
        ph["fallback_steps"] += int(fallback)
        # Tracking: state after this step vs the reference at its time.
        q1 = e.arm_qpos()
        r1 = self.ref.reference(e.time)
        qe = float(np.abs(q1 - r1.q).max())
        ee = float(np.linalg.norm(self.ee_fk(q1) - self.ee_fk(r1.q)))
        self.q_err_max, self.ee_err_max = max(self.q_err_max, qe), max(self.ee_err_max, ee)
        self.q_err_sq += qe * qe
        self.ee_err_sq += ee * ee
        ph["q_err_max"] = max(ph["q_err_max"], qe)
        ph["ee_err_max"] = max(ph["ee_err_max"], ee)
        force = e.arm_actuator_force()
        self.sat_steps += int(np.any(np.abs(force) >= 0.999 * self.tau_lim))
        jl = e.joint_limits
        self.pos_limit_steps += int(np.any((q1 < jl[:, 0]) | (q1 > jl[:, 1])))
        self.vel_limit_steps += int(np.any(np.abs(e.arm_qvel()) > self.v_max))
        self._monitor_contacts()
        if self.steps % self.clearance_every == 0:
            self._measure_clearance()
        if self.record_every and self.steps % self.record_every == 0:
            p, _ = self.object_pose()
            self.trace.append({"t": e.time, "phase": self.phase, "q": q1.tolist(), "q_ref": r1.q.tolist(),
                               "object": p.tolist(), "width": self.finger_width(), "fallback": fallback})

    def _monitor_contacts(self) -> None:
        d, m = self.engine.data, self.engine.model
        n = d.ncon
        if n == 0:
            return
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        c1, c2 = self.geom_cat[g1], self.geom_cat[g2]
        lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
        forbidden = np.zeros(n, dtype=bool)
        forbidden |= (lo == ENV) & ((hi == ROBOT) | (hi == FINGER))      # robot - environment
        forbidden |= (lo >= ROBOT) & (hi <= FINGER)                      # robot self-contact
        forbidden |= (lo == ROBOT) & (hi == OBJECT)                      # hand/links - object
        if self.phase not in GRASP_PHASES:
            forbidden |= (lo == FINGER) & (hi == OBJECT)
        obj_env = (lo == ENV) & (hi == OBJECT)
        if obj_env.any():
            env_geom = np.where(c1 == ENV, g1, g2)
            on_table = np.isin(env_geom, self.table_geom_ids)
            forbidden |= obj_env & ~on_table                              # object - floor / obstacle
            if self.phase in AIRBORNE_PHASES:
                forbidden |= obj_env & on_table
        if forbidden.any():
            self.forbidden_steps += 1
            self.phase_log[-1]["forbidden_steps"] += 1
            if len(self.forbidden_events) < 20:
                k = int(np.flatnonzero(forbidden)[0])
                self.forbidden_events.append({
                    "time": float(d.time), "phase": self.phase,
                    "pair": [m.body(m.geom_bodyid[g1[k]]).name, m.body(m.geom_bodyid[g2[k]]).name],
                    "geoms": [m.geom(g1[k]).name, m.geom(g2[k]).name], "dist_mm": 1e3 * float(d.contact.dist[k])})

    def _measure_clearance(self, cap: float = 0.10) -> None:
        """Robot-environment distance (every phase) and carried-object-environment distance
        (hold/transport; also obstacle during the other carrying phases), mj_geomDistance on the plant."""
        m, d = self.engine.model, self.engine.data
        best = cap
        for a in self._robot_geoms:
            for b in self._env_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, a, b, min(best, cap), None))
        self.min_clear_robot = min(self.min_clear_robot, best)
        self.phase_log[-1]["min_clear_robot_mm"] = min(self.phase_log[-1]["min_clear_robot_mm"], 1e3 * best)
        if self.phase in ("hold", "transport"):
            best = cap
            for b in self._env_geoms:
                best = min(best, mujoco.mj_geomDistance(m, d, self.obj_geom, b, min(best, cap), None))
            self.min_clear_object = min(self.min_clear_object, best)
            self.phase_log[-1]["min_clear_object_mm"] = min(self.phase_log[-1]["min_clear_object_mm"], 1e3 * best)

    # ------------------------------------------------------------------ phase helpers

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self.phase_log.append({"phase": phase, "t_start": float(self.engine.time), "t_end": None, "steps": 0,
                               "fallback_steps": 0, "forbidden_steps": 0, "q_err_max": 0.0, "ee_err_max": 0.0,
                               "min_clear_robot_mm": np.inf, "min_clear_object_mm": np.inf})

    def _wait(self, condition, timeout: float, reason: str, check=None):
        deadline = self.engine.time + timeout
        while not condition():
            if self.engine.time >= deadline:
                raise EpisodeFailure(self.phase, reason)
            if check is not None:
                check()
            yield

    def _arm_at_rest(self, hold_steps: int) -> callable:
        state = {"n": 0}

        def cond():
            if self.ref.finished(self.engine.time) and np.abs(self.engine.arm_qvel()).max() < 0.02:
                state["n"] += 1
            else:
                state["n"] = 0
            return state["n"] >= hold_steps
        return cond

    def _execute(self, segment, check=None):
        self.ref.start(segment, self.engine.time)
        settle = float(self.pp["timeouts"]["settle"])
        yield from self._wait(self._arm_at_rest(25), segment.duration + settle,
                              f"arm not at rest {settle:.1f} s after the end of the motion", check)

    def _configure_checker(self, mode: str) -> None:
        """Phase-dependent collision rules for planning and line checks (see the module docstring
        of planning/collision.py). Every pair not named here keeps the 20 mm clearance."""
        c = self.checker
        c.detach()
        c.clear_allowed()
        c.set_fixed_qpos(self.engine.data.qpos)  # measured fingers and object pose (known world)
        col = self.pp["collision"]
        # Optional (studio): fingertips may come closer than the clearance to the support surface while they
        # are around the object (short objects). None keeps the v0.6 rule (full clearance everywhere).
        finger_support = col.get("finger_table_min")
        if mode == "free":           # object is an obstacle like any other
            return
        if mode == "near_object":     # descend / retreat with open fingers around the object
            c.allow(self.gripper_geoms, self.object_geoms, col["gripper_object_min"])
            if finger_support is not None:
                c.allow(self.finger_geoms, self.table_geom_ids, finger_support)
            return
        if mode in ("lift", "carry", "lower"):
            c.allow(self.finger_geoms, self.object_geoms, -np.inf)          # intended grasp contact
            c.allow(self.hand_geoms, self.object_geoms, col["gripper_object_min"])
            c.attach(self.pp["object"]["body"], "hand")
            if mode in ("lift", "lower"):                                    # resting on / leaving the table
                c.allow(self.object_geoms, self.table_geom_ids, col["object_table_min"])
                if finger_support is not None:
                    c.allow(self.finger_geoms, self.table_geom_ids, finger_support)
            return
        raise ValueError(mode)

    def _line(self, end_position: np.ndarray, rotation: np.ndarray, requested: float, mode: str) -> CartesianSegment:
        try:
            seg = CartesianSegment.plan(self.kin, self.ref.reference(self.engine.time).q, end_position, rotation,
                                        requested, self.limits)
        except PathInfeasible as exc:
            raise EpisodeFailure(self.phase, f"Cartesian line infeasible: {exc}") from None
        v = seg.validation
        self._configure_checker(mode)
        ok, why = self.checker.path_valid(seg.joint_samples())
        self.paths.append({"phase": self.phase, "kind": "line", "path": [q.tolist() for q in seg.q_grid]})
        self.planning.append({"phase": self.phase, "kind": "line", "duration_s": seg.duration,
                              "limited_by": seg.limited_by, "max_dev_mm": 1e3 * v.max_position_deviation,
                              "max_rot_deg": float(np.degrees(v.max_rotation_error)), "collision_free": ok,
                              "reason": why})
        if not ok:
            raise EpisodeFailure(self.phase, f"Cartesian line not collision-free: {why}")
        tol = self.pp["line_tolerance"]
        if v.max_position_deviation > tol["position"] or np.degrees(v.max_rotation_error) > tol["rotation_deg"]:
            raise EpisodeFailure(self.phase, f"Cartesian line deviates {1e3 * v.max_position_deviation:.2f} mm / "
                                             f"{np.degrees(v.max_rotation_error):.2f} deg from the straight line")
        return seg

    def _carry_arc(self, end_position: np.ndarray, rotation: np.ndarray) -> CartesianSegment | None:
        """Transport along an arc around the robot base (radius, azimuth and height interpolated linearly) at
        the grasp orientation, with the object attached for the collision check. Returns None when the arc
        cannot be used (IK, posture jump, tolerance or collision; the reason is logged in `planning`)."""
        e = self.engine
        q0 = self.ref.reference(e.time).q
        p0, _ = self.kin.fk_pose(q0)
        center = e.data.xpos[e.model.body(e.robot.base_body).id][:2]
        pts, length, dth = carry_arc_points(p0, end_position, center)
        info = {"phase": self.phase, "kind": "arc", "length_m": length, "sweep_deg": float(np.degrees(dth))}
        self.planning.append(info)
        if length < 0.02:
            info.update(collision_free=None, reason="too short for an arc")
            return None
        speed = float(self.pp["timing"].get("carry_speed", 0.3))
        requested = max(float(self.pp["timing"]["line_time"]), length / speed)
        try:
            seg = CartesianSegment.plan_path(self.kin, q0, pts, rotation, requested, self.limits)
        except PathInfeasible as exc:
            info.update(collision_free=None, reason=f"arc infeasible: {exc}")
            return None
        v = seg.validation
        self._configure_checker("carry")
        ok, why = self.checker.path_valid(seg.joint_samples())
        tol = self.pp["line_tolerance"]
        within = v.max_position_deviation <= tol["position"] and np.degrees(v.max_rotation_error) <= tol["rotation_deg"]
        info.update(duration_s=seg.duration, limited_by=seg.limited_by, max_dev_mm=1e3 * v.max_position_deviation,
                    max_rot_deg=float(np.degrees(v.max_rotation_error)), collision_free=ok,
                    reason=why if not ok else ("" if within else "deviates from the arc"))
        if not ok or not within:
            return None
        self.paths.append({"phase": self.phase, "kind": "line", "path": [q.tolist() for q in seg.q_grid]})
        return seg

    def _line_feasible(self, q_from: np.ndarray, end_position: np.ndarray, rotation: np.ndarray) -> bool:
        """Kinematic check that a straight constant-orientation line from q_from can be followed within the
        line tolerances (studio: used to choose approach and pre-place postures)."""
        try:
            seg = CartesianSegment.plan(self.kin, q_from, end_position, rotation, float(self.pp["timing"]["line_time"]),
                                        self.limits)
        except PathInfeasible:
            return False
        v, tol = seg.validation, self.pp["line_tolerance"]
        return v.max_position_deviation <= tol["position"] and np.degrees(v.max_rotation_error) <= tol["rotation_deg"]

    def _plan_move(self, ee_target: np.ndarray, rotation: np.ndarray, mode: str, accept=None) -> JointPathSegment:
        self._configure_checker(mode)
        q_from = self.ref.reference(self.engine.time).q
        t0 = time.perf_counter()
        plan = self.planner.plan_to_position(ee_target, q_from, rotation=rotation, accept=accept)
        return self._path_segment(plan, t0)

    def _plan_joint_move(self, q_goal: np.ndarray, mode: str) -> JointPathSegment:
        """Collision-free joint-space move to a configuration (studio: return to the rest posture)."""
        self._configure_checker(mode)
        q_from = self.ref.reference(self.engine.time).q
        t0 = time.perf_counter()
        plan = self.planner.plan_joint(q_from, np.asarray(q_goal, float))
        return self._path_segment(plan, t0)

    def _path_segment(self, plan, t0: float) -> JointPathSegment:
        info = {"phase": self.phase, "kind": "rrt", "success": plan.success, "status": plan.status,
                "reason": plan.reason, "wall_s": time.perf_counter() - t0,
                "vertices": len(plan.path), "length_rad": plan.stats.get("smoothed_length_rad")}
        self.planning.append(info)
        if not plan.success:
            raise EpisodeFailure(self.phase, f"planning failed ({plan.status}): {plan.reason}")
        self.paths.append({"phase": self.phase, "kind": "rrt", "path": [q.tolist() for q in plan.path],
                           "raw_path": [q.tolist() for q in plan.raw_path]})
        try:
            seg = JointPathSegment(plan.path, float(self.pp["timing"]["edge_time"]), self.engine.joint_limits,
                                   **self.timing)
        except TrajectoryInfeasible as exc:
            raise EpisodeFailure(self.phase, f"trajectory infeasible: {exc}") from None
        info["duration_s"] = seg.duration
        return seg

    # ------------------------------------------------------------------ the task

    def _program(self):
        pp, cr = self.pp, self.criteria
        g = pp["grasp"]
        R = top_down_rotation(float(g.get("yaw", 0.0)))
        line_t = float(pp["timing"]["line_time"])
        obj0, _ = self.object_pose()
        belief = obj0 + np.r_[np.asarray(self.fault.get("grasp_offset", [0, 0]), float), 0.0]
        grasp = belief + np.array([0.0, 0.0, g["height_above_center"]])
        pre = grasp + np.array([0.0, 0.0, g["approach_height"]])
        self.metrics["object_initial"] = obj0.tolist()
        # Optional (studio): only accept approach / pre-place postures from which the following straight line
        # (descent, lowering) can be followed; otherwise a posture near a joint limit can end the episode there.
        straight = bool(pp.get("straight_line_postures"))

        self._enter("approach")
        descend_ok = (lambda q: self._line_feasible(q, grasp, R)) if straight else None
        yield from self._execute(self._plan_move(pre, R, "free", accept=descend_ok))

        self._enter("descend")
        yield from self._execute(self._line(grasp, R, line_t, "near_object"))

        self._enter("close")
        # Object width between the fingers. v0.6 boxes are square in x-y; the studio passes the side that lies
        # across the finger axis for the chosen grasp yaw.
        width = float(pp["object"].get("grasp_width", 2 * float(self.obj_half[1])))
        if not self.fault.get("skip_close"):
            self.engine.set_gripper(pp["gripper"]["close"])
        state = {"last": self.finger_width(), "n": 0, "t0": self.engine.time}

        def fingers_stopped():
            w = self.finger_width()
            state["n"] = state["n"] + 1 if abs(w - state["last"]) < 2e-6 else 0
            state["last"] = w
            return self.engine.time - state["t0"] > 0.1 and state["n"] >= 25
        yield from self._wait(fingers_stopped, float(pp["timeouts"]["close"]), "fingers did not come to rest")
        w = self.finger_width()
        self.metrics["grasp_width_mm"] = 1e3 * w
        if abs(w - width) > cr.width_tolerance:
            raise EpisodeFailure("close", f"fingers at {1e3 * w:.1f} mm, object is {1e3 * width:.0f} mm wide: "
                                          f"not holding the object")

        self._enter("lift")
        lift_to = self.ref.reference(self.engine.time).q
        p_lift, _ = self.kin.fk_pose(lift_to)
        yield from self._execute(self._line(p_lift + [0, 0, g["lift_height"]], R, line_t, "lift"))
        obj, _ = self.object_pose()
        lifted = float(obj[2] - obj0[2])
        self.metrics["lift_height_mm"] = 1e3 * lifted
        if lifted < cr.lift_height:
            raise EpisodeFailure("lift", f"object rose {1e3 * lifted:.1f} mm (< {1e3 * cr.lift_height:.0f} mm): "
                                         f"not lifted")

        self._enter("hold")
        rel0 = self.object_in_hand()
        slip = {"max": 0.0}

        def held():
            s = float(np.linalg.norm(self.object_in_hand() - rel0))
            slip["max"] = max(slip["max"], s)
            if s > cr.hold_slip:
                raise EpisodeFailure("hold", f"object slipped {1e3 * s:.1f} mm in the hand (> {1e3 * cr.hold_slip:.0f} mm)")
            if self.object_pose()[0][2] - obj0[2] < cr.lift_height:
                raise EpisodeFailure("hold", "object fell below the lift height")
        t_end = self.engine.time + cr.hold_time
        yield from self._wait(lambda: self.engine.time >= t_end, cr.hold_time + 1.0, "hold did not finish", held)
        self.metrics["hold_slip_mm"] = 1e3 * slip["max"]

        self._enter("transport")
        place = np.array([*self.case["place"], float(self.case.get("place_z", pp["table_height"] + self.obj_half[2]))])
        place_belief = place + np.r_[np.asarray(self.fault.get("place_offset", [0, 0]), float), 0.0]
        # Where the end effector must be to put the object's center on the target: the measured
        # end-effector-to-object offset of the actual grasp (known world, measured state).
        ee_now = self.ee_fk(self.engine.arm_qpos())
        offset = ee_now - self.object_pose()[0]
        place_ee = place_belief + offset + np.array([0.0, 0.0, pp["place_clearance"]])
        preplace_ee = place_ee + np.array([0.0, 0.0, g["lift_height"]])
        carry = {"max": 0.0}

        def carried():
            s = float(np.linalg.norm(self.object_in_hand() - rel0))
            carry["max"] = max(carry["max"], s)
            key = f"{self.phase}_slip_mm"
            self.metrics[key] = max(self.metrics.get(key, 0.0), 1e3 * s)
            if s > cr.drop_slip:
                raise EpisodeFailure(self.phase, f"object dropped / slipped {1e3 * s:.1f} mm in the hand "
                                                 f"(> {1e3 * cr.drop_slip:.0f} mm)")
        # Studio option: carry along a constant-orientation arc around the base when it is feasible and
        # collision-free (a joint-space path can tilt the hand on long swings); otherwise plan as in v0.6.
        lower_ok = (lambda q: self._line_feasible(q, place_ee, R)) if straight else None
        seg = self._carry_arc(preplace_ee, R) if pp.get("transport") == "arc" else None
        if seg is not None and lower_ok is not None and not lower_ok(seg.end):
            self.planning[-1]["reason"] = "the straight lowering cannot be followed from the end of the arc"
            seg = None
        if seg is None:
            seg = self._plan_move(preplace_ee, R, "carry", accept=lower_ok)
        if self.fault.get("release_during_transport"):  # negative control: open the gripper half-way
            self.ref.start(seg, self.engine.time)
            t_open = self.engine.time + 0.5 * seg.duration
            yield from self._wait(lambda: self.engine.time >= t_open, seg.duration, "never", carried)
            self.engine.set_gripper(pp["gripper"]["open"])
            settle = float(pp["timeouts"]["settle"])
            yield from self._wait(self._arm_at_rest(25), 0.5 * seg.duration + settle, "arm not at rest", carried)
        else:
            yield from self._execute(seg, carried)

        self._enter("lower")
        # Fresh grasp offset: the object may have moved in the hand during transport.
        offset = self.ee_fk(self.engine.arm_qpos()) - self.object_pose()[0]
        place_ee = place_belief + offset + np.array([0.0, 0.0, pp["place_clearance"]])
        start_ee, _ = self.kin.fk_pose(self.ref.reference(self.engine.time).q)
        place_ee[:2] = start_ee[:2]  # the line stays vertical; the transport put the object above the target
        yield from self._execute(self._line(place_ee, R, line_t, "lower"), carried)
        self.metrics["carry_slip_mm"] = 1e3 * carry["max"]

        self._enter("release")
        self.engine.set_gripper(pp["gripper"]["open"])
        state = {"last": self.finger_width(), "n": 0, "t0": self.engine.time}
        yield from self._wait(fingers_stopped, float(pp["timeouts"]["release"]), "fingers did not come to rest")
        if self.finger_width() < width + 0.02:
            raise EpisodeFailure("release", f"fingers opened only to {1e3 * self.finger_width():.1f} mm")

        self._enter("retreat")
        p_now, _ = self.kin.fk_pose(self.ref.reference(self.engine.time).q)
        yield from self._execute(self._line(p_now + [0, 0, g["retreat_height"]], R, line_t, "near_object"))

        self._enter("verify")
        quiet = {"since": None}

        def settled():
            v, w_ = self.object_speed()
            if v < cr.settle_speed and w_ < 0.05:
                quiet["since"] = self.engine.time if quiet["since"] is None else quiet["since"]
                return self.engine.time - quiet["since"] >= cr.settle_time
            quiet["since"] = None
            return False
        yield from self._wait(settled, float(pp["timeouts"]["verify"]), "object did not settle")
        p, Rm = self.object_pose()
        err = p - place
        tilt = float(np.degrees(np.arccos(np.clip(Rm[2, 2], -1.0, 1.0))))
        self.metrics.update(placement_error_mm=1e3 * float(np.linalg.norm(err)),
                            placement_error_xy_mm=1e3 * float(np.linalg.norm(err[:2])),
                            placement_error_z_mm=1e3 * float(err[2]), final_tilt_deg=tilt,
                            object_final=p.tolist())
        if np.linalg.norm(err) > cr.place_tolerance:
            raise EpisodeFailure("verify", f"object {1e3 * np.linalg.norm(err):.1f} mm from the target "
                                           f"(> {1e3 * cr.place_tolerance:.0f} mm)")
        if tilt > cr.tilt_tolerance_deg:
            raise EpisodeFailure("verify", f"object tilted {tilt:.1f} deg (> {cr.tilt_tolerance_deg:.0f} deg)")

        # Optional (studio): move back to a rest configuration after a verified placement, collision-free
        # around everything on the table (the placed object included).
        home = pp.get("return_home")
        if home is not None:
            self._enter("home")
            yield from self._execute(self._plan_joint_move(np.asarray(home, float), "free"))

    # ------------------------------------------------------------------ run

    def run(self, on_step=None) -> dict:
        """Run the episode; `on_step(episode)` is called after every physics step (e.g. to render)."""
        self.start()
        while self.advance():
            if on_step is not None:
                on_step(self)
        return self.finish()

    # Incremental interface (v0.7 workbench): start(), then advance() one physics step at a time until it
    # returns False, then finish(). run() is exactly this loop.

    def start(self) -> None:
        self._wall0 = time.perf_counter()
        self._program_iter = self._program()
        self._failure: EpisodeFailure | None = None
        self.done = False

    def advance(self) -> bool:
        """Let the supervisor act (this may plan), then take one physics step. False once the episode ended."""
        if self.done:
            return False
        try:
            next(self._program_iter)
        except StopIteration:
            self.done = True
            return False
        except EpisodeFailure as exc:
            self._failure, self.done = exc, True
            return False
        self._step()
        return True

    def abort(self, reason: str) -> None:
        """End the episode early (e.g. stopped by the user); it is reported as a failure at stage 'stopped'."""
        if not self.done:
            self._failure, self.done = EpisodeFailure("stopped", reason), True
            self._program_iter.close()

    def finish(self) -> dict:
        failure = self._failure
        if self.phase_log:
            self.phase_log[-1]["t_end"] = float(self.engine.time)
            for a, b in zip(self.phase_log[:-1], self.phase_log[1:]):
                a["t_end"] = b["t_start"]
        if failure is None and self.forbidden_steps:
            failure = EpisodeFailure("contacts", f"{self.forbidden_steps} physics steps with forbidden contacts "
                                                 f"(first: {self.forbidden_events[0]})")
        return self.result(failure, time.perf_counter() - self._wall0)

    def result(self, failure: EpisodeFailure | None, wall: float) -> dict:
        n = max(self.steps, 1)
        out = {
            "case": self.case.get("name"), "mass_kg": float(self.engine.model.body_mass[self.obj_body]),
            "controller": self.cfg.get("label", self.controller_name), "controller_type": self.controller_name,
            "seed": int(self.case.get("seed", 0)), "fault": self.fault or None,
            "success": failure is None, "failure_stage": None if failure is None else failure.stage,
            "failure_reason": None if failure is None else failure.reason,
            "sim_time_s": float(self.engine.time), "wall_time_s": wall, "steps": self.steps,
            "forbidden_contact_steps": self.forbidden_steps, "forbidden_events": self.forbidden_events,
            "joint_err_max_rad": self.q_err_max, "joint_err_rms_rad": float(np.sqrt(self.q_err_sq / n)),
            "ee_err_max_mm": 1e3 * self.ee_err_max, "ee_err_rms_mm": 1e3 * float(np.sqrt(self.ee_err_sq / n)),
            "torque_saturation_steps": self.sat_steps, "position_limit_steps": self.pos_limit_steps,
            "velocity_limit_steps": self.vel_limit_steps,
            "min_clearance_robot_env_mm": None if not np.isfinite(self.min_clear_robot) else 1e3 * self.min_clear_robot,
            "min_clearance_object_env_mm": None if not np.isfinite(self.min_clear_object) else 1e3 * self.min_clear_object,
            "phases": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in p.items()}
                       for p in self.phase_log],
            "planning": self.planning, **self.metrics,
        }
        stats = getattr(self.controller, "stats", None)
        if stats is not None and stats.updates:
            ups = stats.updates
            statuses: dict[str, int] = {}
            for u in ups:
                statuses[u.status] = statuses.get(u.status, 0) + 1
            solve = np.array([u.solve_time for u in ups])
            upd = np.array([u.update_time for u in ups])
            prim = [u.prim_res for u in ups if np.isfinite(u.prim_res)]
            dual = [u.dual_res for u in ups if np.isfinite(u.dual_res)]
            out["mpc"] = {
                "updates": len(ups), "fallback_updates": int(sum(u.fallback for u in ups)), "statuses": statuses,
                "fallback_steps": int(sum(p["fallback_steps"] for p in self.phase_log)),
                "solve_ms_mean": 1e3 * float(solve.mean()), "solve_ms_p95": 1e3 * float(np.percentile(solve, 95)),
                "update_ms_mean": 1e3 * float(upd.mean()), "update_ms_max": 1e3 * float(upd.max()),
                "deadline_overruns": int(np.sum(upd > self.controller.control_period)),
                "prim_res_max": float(max(prim)) if prim else None, "dual_res_max": float(max(dual)) if dual else None,
                "control_rate_hz": 1.0 / self.controller.control_period,
            }
        out["controller_rate_hz"] = (1.0 / self.controller.control_period if hasattr(self.controller, "control_period")
                                     else 1.0 / self.engine.dt)
        return out


def run_case(cfg: dict, case: dict, **kwargs) -> dict:
    return PickPlaceEpisode(copy.deepcopy(cfg), case, **kwargs).run()
