"""Closed-loop simulation runner: task reference -> controller -> actuators -> physics.

Each physics step (model.opt.timestep):
  1. read measured arm state from MjData
  2. task produces the joint reference for the current time
  3. controller turns (reference, measurement) into actuator commands
     (the MPC re-solves only every control period and holds its command between)
  4. mj_step integrates the dynamics

The controller is chosen by config["controller"]["type"]: joint_pid or joint_mpc.

With config["task"]["planning"]["enabled"] (v0.5) the task is built from
collision-free joint paths found by the motion planner (planning/). Planning
happens once, here, when the task is built; the controllers only see the
resulting joint reference.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from controllers.base import Controller
from controllers.computed_torque import ComputedTorqueController
from controllers.disturbance import JointDisturbanceObserver
from controllers.dynamics import ArmInverseDynamics
from controllers.mpc import JointMPCController, MPCWeights
from controllers.pid import JointPIDController
from controllers.prediction import MuJoCoPredictionModel
from controllers.torque_mpc import TorqueMPCController, TorqueMPCWeights
from planning.motion_planner import MotionPlanner, PlanningSettings
from robots.panda import Panda
from simulation.engine import PROJECT_ROOT, SimulationEngine
from simulation.kinematics import ArmKinematics
from tasks.joint_target import JointTargetTask
from tasks.planned_joint import PlannedJointTask
from tasks.reach import CartesianReachTask
from tasks.trajectory import KinematicLimits

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

CONTROLLER_TYPES = {
    "joint_pid": "joint_pid", "pid": "joint_pid",
    "joint_mpc": "joint_mpc", "mpc": "joint_mpc",
    "torque_ctc": "torque_ctc", "ctc": "torque_ctc",
    "torque_mpc": "torque_mpc",
}
# Each controller requires one actuator model; the engine is built to match.
CONTROLLER_ACTUATION = {"joint_pid": "position", "joint_mpc": "position", "torque_ctc": "torque", "torque_mpc": "torque"}


def controller_kind(ctrl_cfg: dict) -> str:
    kind = CONTROLLER_TYPES.get(ctrl_cfg.get("type", "joint_pid"))
    if kind is None:
        raise ValueError(f"Unknown controller type '{ctrl_cfg.get('type')}'. Use one of {sorted(CONTROLLER_TYPES)}")
    return kind


def control_substeps(rate_hz: float, engine: SimulationEngine) -> int:
    substeps = (1.0 / rate_hz) / engine.dt
    if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
        raise ValueError(f"control_rate_hz={rate_hz} must divide the physics rate {1 / engine.dt:.0f} Hz")
    return int(round(substeps))


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config. `extends: <file>` (relative to the config) merges it over that base;
    the `task` section is replaced as a whole, not merged."""
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("extends", None)
    if base_name:
        base = load_config(path.parent / base_name)
        task = cfg.pop("task", None)
        cfg = _merge(base, cfg)
        if task is not None:
            cfg["task"] = task
    return cfg


def build_pid(ctrl_cfg: dict, engine: SimulationEngine) -> JointPIDController:
    return JointPIDController(
        dof=engine.robot.dof,
        kp=ctrl_cfg.get("kp", 0.5),
        ki=ctrl_cfg.get("ki", 5.0),
        kd=ctrl_cfg.get("kd", 0.0),
        velocity_feedforward=ctrl_cfg.get("velocity_feedforward", 0.1),
        integral_limit=ctrl_cfg.get("integral_limit", 0.05),
        ctrl_limits=engine.ctrl_limits,
    )


def build_mpc(ctrl_cfg: dict, engine: SimulationEngine, reference_preview=None) -> JointMPCController:
    """MPC from config["controller"]["mpc"]; the PID gains in config["controller"] configure its fallback."""
    mpc_cfg = ctrl_cfg.get("mpc", {})
    model = MuJoCoPredictionModel(
        engine.controller_model,
        engine.data,
        engine.arm_qpos_adr,
        engine.arm_dof_adr,
        engine.arm_actuator_ids,
        substeps=control_substeps(float(mpc_cfg.get("control_rate_hz", 50.0)), engine),
        method=mpc_cfg.get("linearization", "power"),
    )
    vel = mpc_cfg.get("velocity_limits")
    return JointMPCController(
        model,
        fallback=build_pid(ctrl_cfg, engine),
        horizon=int(mpc_cfg.get("horizon", 15)),
        weights=MPCWeights(**mpc_cfg.get("weights", {})),
        velocity_limits=np.asarray(vel if vel is not None else engine.robot.velocity_limits, dtype=float),
        joint_limits=engine.joint_limits,
        joint_limit_margin=float(mpc_cfg.get("joint_limit_margin", 0.02)),
        torque_limit_fraction=float(mpc_cfg.get("torque_limit_fraction", 0.95)),
        soft_constraints=bool(mpc_cfg.get("soft_constraints", True)),
        command_slew_scale=mpc_cfg.get("command_slew_scale", 1.0),
        disturbance_gain=float(mpc_cfg.get("disturbance_gain", 0.0)),
        velocity_margin=float(mpc_cfg.get("velocity_margin", 0.0)),
        reference_preview=reference_preview if mpc_cfg.get("reference_preview", True) else None,
        solver_settings=mpc_cfg.get("solver"),
    )


def build_observer(obs_cfg: dict | None, engine: SimulationEngine) -> JointDisturbanceObserver | None:
    """Disturbance observer from a config section {enabled, time_constant, limit_fraction}; None if disabled.
    It uses the controller's model (engine.controller_model), never the plant's."""
    if not obs_cfg or not obs_cfg.get("enabled", False):
        return None
    limits = np.abs(engine.controller_model.actuator_forcerange[engine.arm_actuator_ids]).max(axis=1)
    return JointDisturbanceObserver(
        engine.controller_model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr,
        time_constant=float(obs_cfg.get("time_constant", 0.02)),
        limit=float(obs_cfg.get("limit_fraction", 0.5)) * limits,
    )


def build_ctc(ctrl_cfg: dict, engine: SimulationEngine, fallback: bool = False) -> ComputedTorqueController:
    """Computed-torque control from config["controller"]["ctc"] (torque actuation). As the torque
    MPC's fallback (fallback=True) it is the plain v0.3 CTC: no integral, no observer."""
    cfg = ctrl_cfg.get("ctc", {})
    return ComputedTorqueController(
        ArmInverseDynamics(engine.controller_model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr),
        torque_limits=engine.ctrl_limits,
        natural_frequency=cfg.get("natural_frequency", 25.0),
        damping_ratio=cfg.get("damping_ratio", 1.0),
        integral_gain=0.0 if fallback else float(cfg.get("integral_gain", 0.0)),
        integral_limit=float(cfg.get("integral_limit", 0.1)),
        observer=None if fallback else build_observer(cfg.get("disturbance_observer"), engine),
    )


def build_torque_mpc(ctrl_cfg: dict, engine: SimulationEngine, reference_preview=None) -> TorqueMPCController:
    """Torque MPC from config["controller"]["torque_mpc"]; config["controller"]["ctc"] configures its fallback."""
    cfg = ctrl_cfg.get("torque_mpc", {})
    model = MuJoCoPredictionModel(
        engine.controller_model,
        engine.data,
        engine.arm_qpos_adr,
        engine.arm_dof_adr,
        engine.arm_actuator_ids,
        substeps=control_substeps(float(cfg.get("control_rate_hz", 50.0)), engine),
        method=cfg.get("linearization", "power"),
    )
    vel = cfg.get("velocity_limits")
    return TorqueMPCController(
        model,
        fallback=build_ctc(ctrl_cfg, engine, fallback=True),
        horizon=int(cfg.get("horizon", 25)),
        weights=TorqueMPCWeights(**cfg.get("weights", {})),
        velocity_limits=np.asarray(vel if vel is not None else engine.robot.velocity_limits, dtype=float),
        joint_limits=engine.joint_limits,
        joint_limit_margin=float(cfg.get("joint_limit_margin", 0.02)),
        soft_constraints=bool(cfg.get("soft_constraints", True)),
        reference_preview=reference_preview if cfg.get("reference_preview", True) else None,
        solver_settings=cfg.get("solver"),
        velocity_margin=float(cfg.get("velocity_margin", 0.0)),
        observer=build_observer(cfg.get("disturbance_observer"), engine),
        disturbance_feedthrough=bool((cfg.get("disturbance_observer") or {}).get("feedthrough", False)),
    )


def build_controller(ctrl_cfg: dict, engine: SimulationEngine, reference_preview=None) -> Controller:
    kind = controller_kind(ctrl_cfg)
    if engine.robot.actuation != CONTROLLER_ACTUATION[kind]:
        raise ValueError(f"Controller '{kind}' needs {CONTROLLER_ACTUATION[kind]} actuators, "
                         f"the engine has {engine.robot.actuation}")
    if kind == "joint_pid":
        return build_pid(ctrl_cfg, engine)
    if kind == "joint_mpc":
        return build_mpc(ctrl_cfg, engine, reference_preview)
    if kind == "torque_ctc":
        return build_ctc(ctrl_cfg, engine)
    return build_torque_mpc(ctrl_cfg, engine, reference_preview)


@dataclass
class RunLog:
    time: list[float] = field(default_factory=list)
    q: list[np.ndarray] = field(default_factory=list)
    q_ref: list[np.ndarray] = field(default_factory=list)
    ee: list[np.ndarray] = field(default_factory=list)
    qd: list[np.ndarray] = field(default_factory=list)
    qd_ref: list[np.ndarray] = field(default_factory=list)
    qdd_ref: list[np.ndarray] = field(default_factory=list)
    ctrl: list[np.ndarray] = field(default_factory=list)   # servo setpoints sent (rad)
    force: list[np.ndarray] = field(default_factory=list)  # actuator torque applied (Nm)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in vars(self).items()}


class SimulationRunner:
    def __init__(self, config: dict):
        self.config = config
        actuation = CONTROLLER_ACTUATION[controller_kind(config["controller"])]
        self.engine = SimulationEngine(config.get("environment", "table"), Panda(actuation=actuation),
                                       config.get("environment_overrides"))
        mismatch = config.get("controller_model", {})
        if mismatch:  # model-mismatch experiment: controllers get a wrong model, the plant stays exact
            self.engine.controller_model = self.engine.perturbed_model(mass_scale=mismatch.get("mass_scale", 1.0))

        self.planner: MotionPlanner | None = None
        self.task = self._build_task(config["task"])
        # Optional measurement noise (v0.4): controllers see q + n_q, qd + n_qd (Gaussian, fixed seed).
        # Metrics always use the true state.
        noise = config.get("sensor_noise") or {}
        self._noise_std = (float(noise.get("position_std", 0.0)), float(noise.get("velocity_std", 0.0)))
        self._noise_seed = int(noise.get("seed", 0))

        self.controller = build_controller(config["controller"], self.engine, self._reference_preview)

        # Cartesian goal of each waypoint, shown by the target marker.
        self.waypoint_ee = np.array(
            [self.engine.forward_kinematics(q) for q in self.task.waypoints]
        )
        for i, target in enumerate(getattr(self.task, "waypoint_targets", [])):
            if target is not None:
                self.waypoint_ee[i] = target
        goal_index = getattr(self.task, "goal_index", None)
        if goal_index is not None:  # planned via points show the waypoint they lead to
            self.waypoint_ee = self.waypoint_ee[goal_index]
        self.reset()

    def reset(self) -> None:
        self.engine.reset(self.task.waypoints[0])
        self.engine.set_gripper(self.config.get("gripper", 255))
        self.controller.reset()
        self._rng = np.random.default_rng(self._noise_seed)
        self.engine.set_target_marker(self.waypoint_ee[0])
        self.engine.forward()

    def trajectory_options(self, traj_cfg: dict | None) -> dict:
        """Timing options for the tasks from config["task"]["trajectory"] (v0.4).

        None/absent: the v0.3 fixed-duration minimum-jerk moves, unchecked. Otherwise
        moves are lengthened to respect scaled datasheet limits (and, with
        torque_scale, a torque budget checked with the controller's model)."""
        if not traj_cfg:
            return {}
        r = self.engine.robot
        limits = KinematicLimits(
            velocity=np.asarray(r.velocity_limits) * traj_cfg.get("velocity_scale", 1.0),
            acceleration=np.asarray(r.acceleration_limits) * traj_cfg.get("acceleration_scale", 1.0),
            jerk=np.asarray(r.jerk_limits) * traj_cfg.get("jerk_scale", 1.0),
            position=self.engine.joint_limits,
        )
        opts = {"profile": traj_cfg.get("profile", "min_jerk"), "limits": limits}
        if traj_cfg.get("torque_scale") is not None:
            e = self.engine
            dyn = ArmInverseDynamics(e.controller_model, e.data, e.arm_qpos_adr, e.arm_dof_adr)
            opts["torque_fn"] = dyn.torque
            opts["torque_limits"] = traj_cfg["torque_scale"] * np.abs(
                e.model.actuator_forcerange[e.arm_actuator_ids]).max(axis=1)
        return opts

    def _make_planner(self, plan_cfg: dict, kinematics: ArmKinematics | None = None) -> MotionPlanner:
        self.planner = MotionPlanner.from_engine(self.engine, PlanningSettings.from_config(plan_cfg), kinematics)
        return self.planner

    def _build_task(self, task_cfg: dict):
        kind = task_cfg.get("type", "joint_target")
        move_time, hold_time = task_cfg.get("move_time", 3.0), task_cfg.get("hold_time", 1.0)
        timing = self.trajectory_options(task_cfg.get("trajectory"))
        plan_cfg = dict(task_cfg.get("planning") or {})
        planning = bool(plan_cfg.get("enabled", False))
        if kind == "joint_target":
            waypoints = task_cfg["waypoints"]
            self.waypoint_names = list(waypoints)
            if planning:
                planner = self._make_planner(plan_cfg)
                return PlannedJointTask(
                    list(waypoints.values()),
                    plan=planner.plan_joint,
                    move_time=move_time,
                    hold_time=hold_time,
                    joint_limits=self.engine.joint_limits,
                    loop=task_cfg.get("loop", True),
                    path_move_time=plan_cfg.get("segment_time"),
                    names=self.waypoint_names,
                    **timing,
                )
            return JointTargetTask(
                list(waypoints.values()),
                move_time=move_time,
                hold_time=hold_time,
                joint_limits=self.engine.joint_limits,
                loop=task_cfg.get("loop", True),
                **timing,
            )
        if kind == "cartesian_reach":
            ik_cfg = task_cfg.get("ik", {})
            self.kinematics = ArmKinematics(
                self.engine,
                limit_margin=ik_cfg.get("joint_limit_margin", 0.05),
                damping=ik_cfg.get("damping", 0.05),
            )
            tol = ik_cfg.get("tolerance", 1e-4)
            targets = [(t["name"], t["position"]) for t in task_cfg["targets"]]
            start = np.asarray(task_cfg.get("start", self.engine.robot.home_qpos), dtype=float)
            if planning:
                planner = self._make_planner({**plan_cfg, "ik": {"tolerance": tol, **(plan_cfg.get("ik") or {})}},
                                             self.kinematics)
                plan = planner.plan_to_position
            else:
                # IK + straight-path collision check (v0.3). check_path: false skips the path
                # check: an explicit, unsafe comparison mode that may drive into obstacles.
                check_path = bool(ik_cfg.get("check_path", True))
                plan = lambda pos, q_from: self.kinematics.plan_target(pos, q_from, tol, check_path=check_path)
            task = CartesianReachTask(
                targets,
                start,
                plan=plan,
                move_time=move_time,
                hold_time=hold_time,
                joint_limits=self.engine.joint_limits,
                path_move_time=plan_cfg.get("segment_time") if planning else None,
                **timing,
            )
            self.waypoint_names = ["start"] + [p.name for p in task.reachable]
            return task
        raise ValueError(f"Unknown task type '{kind}'. Use joint_target or cartesian_reach")

    def _reference_preview(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        ref = self.task.reference(t)
        return ref.q, ref.qd

    def step(self) -> None:
        """Advance one control + physics step."""
        e = self.engine
        ref = self.task.reference(e.time)
        q, qd = e.arm_qpos(), e.arm_qvel()
        if self._noise_std[0] > 0 or self._noise_std[1] > 0:
            q = q + self._rng.normal(0.0, self._noise_std[0], q.shape)
            qd = qd + self._rng.normal(0.0, self._noise_std[1], qd.shape)
        ctrl = self.controller.compute(q, qd, ref.q, ref.qd, e.dt, t=e.time, qdd_ref=ref.qdd)
        e.set_arm_command(ctrl)
        e.set_target_marker(self.waypoint_ee[ref.waypoint_index])
        e.step()

    def run_headless(self, duration: float, log_every: int = 10, before_step=None) -> RunLog:
        """Run without a viewer. `before_step(runner)` is called before every step
        (e.g. to apply a test disturbance)."""
        log = RunLog()
        e = self.engine
        n = int(round(duration / e.dt))
        for i in range(n):
            if before_step is not None:
                before_step(self)
            self.step()
            if i % log_every == 0:
                ref = self.task.reference(e.time)
                log.time.append(e.time)
                log.q.append(e.arm_qpos())
                log.q_ref.append(ref.q)
                log.ee.append(e.ee_position())
                log.qd.append(e.arm_qvel())
                log.qd_ref.append(ref.qd)
                log.qdd_ref.append(ref.qdd if ref.qdd is not None else np.zeros_like(ref.qd))
                log.ctrl.append(e.data.ctrl[e.arm_actuator_ids].copy())
                log.force.append(e.arm_actuator_force())
        return log

    def run_viewer(self, duration: float | None = None) -> None:
        """Run in MuJoCo's built-in interactive viewer, synced to wall-clock time."""
        import mujoco.viewer

        realtime = self.config.get("viewer", {}).get("realtime", True)
        m, d = self.engine.model, self.engine.data
        threads_before = set(threading.enumerate())
        try:
            with mujoco.viewer.launch_passive(m, d) as viewer:
                if self.config.get("viewer", {}).get("show_planned_path", True):
                    with viewer.lock():
                        self.draw_planned_paths(viewer.user_scn)
                self._viewer_loop(viewer, duration, realtime)
        finally:
            # MuJoCo's passive viewer keeps tearing down its OpenGL context on a
            # background thread after close(). Exiting the interpreter before it
            # finishes crashes with an access violation on Windows, so wait for it.
            for thread in set(threading.enumerate()) - threads_before:
                thread.join(timeout=5.0)

    def draw_planned_paths(self, scene) -> int:
        """Add the planned end-effector paths (green: executed, orange: raw RRT) to a scene."""
        from planning.visualization import draw_planned_paths

        planned = getattr(self.task, "planned_paths", [])
        if not planned:
            return 0
        kin = ArmKinematics(self.engine)
        return draw_planned_paths(scene, kin.fk, planned)

    def _viewer_loop(self, viewer, duration: float | None, realtime: bool) -> None:
        d = self.engine.data
        steps_per_frame = max(1, int(round((1 / 60) / self.engine.dt)))  # ~60 FPS
        start_wall, start_sim = time.perf_counter(), d.time
        while viewer.is_running():
            if duration is not None and d.time - start_sim >= duration:
                break
            frame_start = time.perf_counter()
            for _ in range(steps_per_frame):
                self.step()
            viewer.sync()
            if realtime:
                lag = (d.time - start_sim) - (time.perf_counter() - start_wall)
                if lag > 0:
                    time.sleep(lag)
            elif (elapsed := time.perf_counter() - frame_start) < 1 / 60:
                time.sleep(1 / 60 - elapsed)
