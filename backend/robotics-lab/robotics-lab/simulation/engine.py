"""MuJoCo model assembly and physics stepping.

The engine composes an environment MJCF (environments/*.xml) with a robot spec
by attaching the robot at the environment's "robot_mount" site. All motion is
produced by mj_step from actuator commands; joint positions are only written
once, in reset(), to set the initial state.
"""

from pathlib import Path

import mujoco
import numpy as np

from robots.panda import Panda

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENTS_DIR = PROJECT_ROOT / "environments"
MOUNT_SITE = "robot_mount"
TARGET_BODY = "target"
# Free boxes added through overrides["objects"] (studio): same contact settings as the v0.6 object.
FREE_BOX_FRICTION = (1.0, 0.005, 0.0001)
FREE_BOX_CONDIM = 4


def environment_path(name: str) -> Path:
    path = ENVIRONMENTS_DIR / f"{name}.xml"
    if not path.is_file():
        available = sorted(p.stem for p in ENVIRONMENTS_DIR.glob("*.xml"))
        raise FileNotFoundError(f"Unknown environment '{name}'. Available: {available}")
    return path


def add_free_box(env: mujoco.MjSpec, name: str, pos, half_size, mass: float, quat=None, friction=None,
                 rgba=None) -> None:
    """Add a free rigid box to the environment spec: body `name`, free joint `<name>_free`, box geom
    `<name>_geom` (mass in kg, inertia from the shape). It moves only through contact and friction,
    exactly like the v0.6 object in environments/pick_place.xml."""
    if not name or env.body(name) is not None:
        raise ValueError(f"environment override: cannot add object '{name}' (empty or duplicate name)")
    half = np.asarray(half_size, float)
    if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(half <= 0):
        raise ValueError(f"environment override: object '{name}' needs 3 positive half sizes")
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"environment override: object '{name}' needs a positive mass")
    body = env.worldbody.add_body(name=name, pos=np.asarray(pos, float).tolist(),
                                  quat=np.asarray([1.0, 0.0, 0.0, 0.0] if quat is None else quat, float).tolist())
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(), mass=float(mass),
                  friction=list(FREE_BOX_FRICTION if friction is None else friction), condim=FREE_BOX_CONDIM,
                  rgba=list((0.8, 0.8, 0.8, 1.0) if rgba is None else rgba))


def apply_overrides(env: mujoco.MjSpec, overrides: dict | None) -> None:
    """Change environment geometry in the spec (the XML file is untouched).

    `overrides` = {"bodies": {name: {"pos": [x, y, z]}}, "geoms": {name: {"size": [...], "pos": [...]}}}.
    Geoms also accept "mass" (kg, inertia follows from the shape) and "friction" ([slide, spin, roll])
    (v0.6: the manipulated object's mass and friction per episode).
    Used for episodes whose obstacle geometry differs from the file (v0.5.1); the
    planner's collision checker copies the compiled model, so it sees the same geometry.

    Studio additions: bodies also accept "quat" ([w, x, y, z]), and `overrides["objects"]` is a list of
    free boxes added before the edits ({"name", "pos", "half_size", "mass", optional "quat", "friction",
    "rgba"}; see add_free_box), so that "bodies"/"geoms" entries can address them.
    """
    for obj in (overrides or {}).get("objects") or []:
        add_free_box(env, **obj)
    for kind, lookup, fields in (("bodies", env.body, ("pos", "quat")),
                                 ("geoms", env.geom, ("pos", "size", "mass", "friction"))):
        for name, values in ((overrides or {}).get(kind) or {}).items():
            element = lookup(name)
            if element is None:
                raise ValueError(f"environment override: no {kind[:-1] if kind == 'geoms' else 'body'} named '{name}'")
            unknown = set(values) - set(fields)
            if unknown:
                raise ValueError(f"environment override for '{name}': unsupported keys {sorted(unknown)}")
            for field, value in values.items():
                current = np.asarray(getattr(element, field), float)
                value = np.asarray(value, float)
                if value.shape != current.shape:
                    raise ValueError(f"environment override {name}.{field}: expected {current.shape[0]} values")
                setattr(element, field, value)


def build_spec(environment: str, robot: Panda, overrides: dict | None = None) -> mujoco.MjSpec:
    """Return an MjSpec containing the environment with the robot attached."""
    env = mujoco.MjSpec.from_file(str(environment_path(environment)))
    apply_overrides(env, overrides)
    mount = env.site(MOUNT_SITE)
    if mount is None:
        raise ValueError(f"Environment '{environment}' has no '{MOUNT_SITE}' site")

    frame = env.worldbody.add_frame(pos=mount.pos, quat=mount.quat)
    robot_spec = robot.load_spec()
    frame.attach_body(robot_spec.body(robot.base_body), "", "")

    # The robot base rests on its mount surface; ignore that contact.
    table = env.body("table")
    if table is not None:
        env.add_exclude(bodyname1=robot.base_body, bodyname2="table")
    return env


class SimulationEngine:
    """Owns MjModel/MjData and exposes robot-level state and actuation."""

    def __init__(self, environment: str = "table", robot: Panda | None = None, overrides: dict | None = None):
        self.robot = robot or Panda()
        self.environment = environment
        self.overrides = overrides
        self.spec = build_spec(environment, self.robot, overrides)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.arm_qpos_adr = np.array(
            [m.jnt_qposadr[m.joint(j).id] for j in self.robot.arm_joints]
        )
        self.arm_dof_adr = np.array(
            [m.jnt_dofadr[m.joint(j).id] for j in self.robot.arm_joints]
        )
        self.arm_actuator_ids = np.array(
            [m.actuator(a).id for a in self.robot.arm_actuators]
        )
        self.gripper_actuator_id = m.actuator(self.robot.gripper_actuator).id
        self.ee_site_id = m.site(self.robot.ee_site).id
        self.joint_limits = np.array(
            [m.jnt_range[m.joint(j).id] for j in self.robot.arm_joints]
        )
        self.ctrl_limits = m.actuator_ctrlrange[self.arm_actuator_ids].copy()
        # Model handed to model-based controllers. Same as the plant unless a
        # model-mismatch experiment replaces it (see perturbed_model).
        self.controller_model = self.model

        target_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
        self.target_mocap_id = m.body_mocapid[target_id] if target_id >= 0 else -1

        self.reset()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    @property
    def time(self) -> float:
        return self.data.time

    def reset(self, arm_qpos: np.ndarray | None = None) -> None:
        """Reset to the Menagerie home keyframe (optionally overriding arm joints).

        Position servos start with their setpoint at the joint angles; torque
        motors start at zero torque (the controller acts before the first step).
        """
        key = self.model.key("home").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        if arm_qpos is not None:
            self.data.qpos[self.arm_qpos_adr] = arm_qpos
            if self.robot.actuation == "position":
                self.data.ctrl[self.arm_actuator_ids] = arm_qpos
        if self.robot.actuation == "torque":
            self.data.ctrl[self.arm_actuator_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_dof_adr].copy()

    def ee_position(self) -> np.ndarray:
        """End-effector site position from the last kinematics pass. After step() this
        is the position at the start of that step (one physics step behind qpos);
        use forward_kinematics(arm_qpos()) when exact alignment matters."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def set_arm_command(self, ctrl: np.ndarray) -> None:
        """Send arm commands, clipped to ctrlrange: joint-angle setpoints (rad) for
        position servos, joint torques (Nm) for torque motors."""
        lo, hi = self.ctrl_limits[:, 0], self.ctrl_limits[:, 1]
        self.data.ctrl[self.arm_actuator_ids] = np.clip(ctrl, lo, hi)

    def arm_actuator_force(self) -> np.ndarray:
        """Torque produced by each arm actuator in the last step (after force-range clipping)."""
        return self.data.actuator_force[self.arm_actuator_ids].copy()

    def apply_external_force(self, body: str, force: np.ndarray) -> None:
        """Push a body with a world-frame force (N) at its center of mass, e.g. as a test disturbance."""
        self.data.xfrc_applied[self.model.body(body).id, :3] = force

    def clear_external_forces(self) -> None:
        self.data.xfrc_applied[:] = 0.0

    def set_gripper(self, value: float) -> None:
        self.data.ctrl[self.gripper_actuator_id] = value

    def set_target_marker(self, position: np.ndarray) -> None:
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = position

    def forward(self) -> None:
        """Recompute derived quantities (poses, contacts) without advancing time."""
        mujoco.mj_forward(self.model, self.data)

    def step(self, nstep: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep)

    def perturbed_model(self, mass_scale: float = 1.0) -> mujoco.MjModel:
        """Copy of the plant model with every robot body's mass and inertia scaled,
        for model-mismatch experiments (the plant itself is unchanged)."""
        import copy

        m = copy.deepcopy(self.model)
        ids = self._robot_body_ids()
        m.body_mass[ids] *= mass_scale
        m.body_inertia[ids] *= mass_scale
        return m

    def contact_free_model(self, nominal_masses: dict[str, float] | None = None) -> mujoco.MjModel:
        """Controller model for manipulation (v0.6): a copy of the current controller model with
        contacts disabled and the named free bodies at fixed nominal masses.

        Model-based controllers copy the live state (all qpos/qvel, including a grasped object's
        pose) into their private data. With contacts enabled there, mj_inverse / mj_forward would
        compute the finger-object and object-table contact forces from that state, and through the
        penetration they depend on the object's true mass: the controller would "see" the payload.
        Without contacts the object is a separate free body that cannot exert force on the arm, so
        the arm's inverse dynamics, gravity torque and prediction are those of the bare arm, the
        same for every payload. The payload is then an unmodelled disturbance (estimated, if at all,
        only by the disturbance observer from measured joint motion)."""
        import copy

        m = copy.deepcopy(self.controller_model)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        for body, mass in (nominal_masses or {}).items():
            b = m.body(body).id
            scale = float(mass) / m.body_mass[b]
            m.body_mass[b] = float(mass)
            m.body_inertia[b] *= scale
        return m

    def forward_kinematics(self, arm_qpos: np.ndarray) -> np.ndarray:
        """End-effector position for a joint configuration, without touching the live state."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.arm_qpos_adr] = arm_qpos
        mujoco.mj_kinematics(self.model, scratch)
        return scratch.site_xpos[self.ee_site_id].copy()

    def robot_contacts(self) -> list[tuple[str, str]]:
        """Active contacts involving at least one robot geom, as (body1, body2) name pairs."""
        m, d = self.model, self.data
        robot_bodies = set(self._robot_body_ids())
        pairs = []
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in robot_bodies or b2 in robot_bodies:
                pairs.append((m.body(b1).name, m.body(b2).name))
        return pairs

    def _robot_body_ids(self) -> list[int]:
        m = self.model
        base = m.body(self.robot.base_body).id
        return [b for b in range(m.nbody) if b == base or self._is_descendant(b, base)]

    def _is_descendant(self, body: int, ancestor: int) -> bool:
        while body != 0:
            body = self.model.body_parentid[body]
            if body == ancestor:
                return True
        return False
