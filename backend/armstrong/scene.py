"""Workcell scene: floor, a round table sized to the arm's reach, the Panda at its centre, any number of
rigid box objects and one calibrated RGB-D camera on a mast in front of the table.

Conventions: meters, radians, right-handed world with +Z up. Quaternions are stored [w, x, y, z]
(MuJoCo order) everywhere in the backend; the browser converts explicitly.

The model is compiled from the scene's objects. Adding, removing or resizing objects recompiles it (while
the robot is idle); moving objects only writes their free-joint qpos. During a run nothing writes object
state: objects move only through contact and friction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import robot as rb

TABLE_TOP_Z = 0.75
TABLE_THICKNESS = 0.04
MOUNT_POS = (0.0, 0.0, TABLE_TOP_Z)
# Usable ring for objects (radius from the robot base axis, azimuth from +x). Reach validated with top-down
# grasps; the azimuth limit keeps every object inside the camera's field of view.
R_IN, R_OUT = 0.32, 0.70
THETA_MAX_DEG = 60.0
TABLE_RADIUS = R_OUT + 0.08
OBJ_FRICTION = (1.0, 0.005, 0.0001)


@dataclass
class ObjectSpec:
    """A rigid box in the simulation (size = full extents along its own x, y, z)."""

    id: str
    size: tuple[float, float, float]
    mass: float
    color: str                                   # #rrggbb
    pos: tuple[float, float, float]              # centre, world frame
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class CameraSpec:
    id: str = "rgbd"
    position: tuple[float, float, float] = (1.32, 0.0, TABLE_TOP_Z + 0.78)
    target: tuple[float, float, float] = (0.48, 0.0, TABLE_TOP_Z)
    vertical_fov_deg: float = 45.0
    width: int = 480
    height: int = 360
    rate_hz: float = 4.0


@dataclass
class SceneConfig:
    objects: list[ObjectSpec] = field(default_factory=list)
    camera: CameraSpec = field(default_factory=CameraSpec)

    def structure_key(self) -> tuple:
        """Everything that needs a recompile when it changes (poses do not)."""
        return tuple((o.id, tuple(np.round(o.size, 5)), round(o.mass, 5), o.color) for o in self.objects)


def hex_rgb(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    try:
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (0.7, 0.7, 0.7)


def look_at_quat(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera quaternion [w,x,y,z]: camera looks along its -Z, +Y is image-up."""
    p, t = np.asarray(position, float), np.asarray(target, float)
    f = t - p
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    R = np.column_stack([x, y, -f])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _object_xml(o: ObjectSpec) -> str:
    r, g, b = hex_rgb(o.color)
    hx, hy, hz = (0.5 * float(v) for v in o.size)
    f = OBJ_FRICTION
    return (f'<body name="{o.id}" pos="{o.pos[0]} {o.pos[1]} {o.pos[2]}" '
            f'quat="{o.quat[0]} {o.quat[1]} {o.quat[2]} {o.quat[3]}">'
            f'<freejoint name="{o.id}_free"/>'
            f'<geom name="{o.id}_geom" type="box" size="{hx} {hy} {hz}" mass="{o.mass}" '
            f'friction="{f[0]} {f[1]} {f[2]}" condim="4" rgba="{r:.4f} {g:.4f} {b:.4f} 1"/></body>')


def _env_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    c = cfg.camera
    tz = TABLE_TOP_Z - TABLE_THICKNESS / 2
    ped_h = (TABLE_TOP_Z - TABLE_THICKNESS) / 2
    objects = "\n    ".join(_object_xml(o) for o in cfg.objects)
    # Camera rig: a mast and housing behind the camera, visual only (group 3 is not rendered by the RGB-D camera).
    cx, cy, cz = c.position
    xml = f"""
<mujoco model="armstrong_workcell">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <statistic center="0.3 0 0.8" extent="1.2"/>
  <visual>
    <global offwidth="960" offheight="720"/>
    <map znear="0.01" zfar="20"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.16 0.16 0.17" rgb2="0.19 0.19 0.20"
      width="256" height="256"/>
    <material name="floor" texture="floor_tex" texrepeat="8 8" reflectance="0.05"/>
    <material name="tabletop" rgba="0.62 0.60 0.56 1" specular="0.2"/>
    <material name="pedestal" rgba="0.23 0.24 0.26 1"/>
    <material name="rig" rgba="0.18 0.19 0.21 1"/>
    <material name="lens" rgba="0.05 0.35 0.38 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="1.2 -0.8 2.8" dir="-0.45 0.3 -1" directional="true" diffuse="0.55 0.55 0.55"
      castshadow="true"/>
    <light name="fill" pos="-0.5 1.2 2.4" dir="0.3 -0.5 -1" directional="true" diffuse="0.2 0.2 0.22"
      castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.05" material="floor"/>
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="cylinder" size="{TABLE_RADIUS} {TABLE_THICKNESS / 2}" pos="0 0 {tz}"
        material="tabletop"/>
      <geom name="table_pedestal" type="cylinder" size="0.14 {ped_h:.4f}" pos="0 0 {ped_h:.4f}" material="pedestal"/>
      <geom name="table_foot" type="cylinder" size="0.36 0.012" pos="0 0 0.012" material="pedestal"/>
    </body>
    <body name="camera_rig" pos="{cx} {cy} 0">
      <geom name="rig_mast" type="cylinder" size="0.018 {cz / 2:.4f}" pos="0.07 0 {cz / 2:.4f}" material="rig"
        contype="0" conaffinity="0" group="3"/>
      <geom name="rig_foot" type="cylinder" size="0.16 0.01" pos="0.07 0 0.01" material="rig"
        contype="0" conaffinity="0" group="3"/>
      <geom name="rig_housing" type="box" size="0.045 0.07 0.03" pos="0.055 0 {cz:.4f}" material="rig"
        contype="0" conaffinity="0" group="3"/>
      <geom name="rig_lens" type="cylinder" size="0.018 0.008" pos="0.005 0 {cz:.4f}" quat="0.7071 0 0.7071 0"
        material="lens" contype="0" conaffinity="0" group="3"/>
    </body>
    {objects}
    <camera name="{c.id}" pos="{cx} {cy} {cz}" fovy="{c.vertical_fov_deg}"/>
  </worldbody>
</mujoco>
"""
    spec = mujoco.MjSpec.from_string(xml)
    spec.camera(c.id).quat = look_at_quat(c.position, c.target)
    return spec


@dataclass
class ObjectIds:
    body: int
    geom: int
    qadr: int
    dadr: int


@dataclass
class SceneIds:
    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    arm_act: np.ndarray
    finger_qadr: np.ndarray
    grip_act: int
    ee_site: int
    hand: int
    camera: int
    table_body: int
    floor_geom: int
    robot_bodies: np.ndarray
    finger_bodies: np.ndarray
    objects: dict[str, ObjectIds]


def build_model(cfg: SceneConfig) -> tuple[mujoco.MjModel, SceneIds]:
    env = _env_spec(cfg)
    frame = env.worldbody.add_frame(pos=list(MOUNT_POS))
    panda = rb.load_panda_spec()
    frame.attach_body(panda.body(rb.BASE_BODY), "", "")
    env.add_exclude(bodyname1=rb.BASE_BODY, bodyname2="table")
    with warnings.catch_warnings():   # empty attach prefix: names stay as in the Menagerie model
        warnings.simplefilter("ignore")
        model = env.compile()
    return model, scene_ids(model, [o.id for o in cfg.objects])


def scene_ids(m: mujoco.MjModel, object_ids: list[str]) -> SceneIds:
    def jq(name):
        return m.jnt_qposadr[m.joint(name).id]

    def jd(name):
        return m.jnt_dofadr[m.joint(name).id]

    base = m.body(rb.BASE_BODY).id
    robot = []
    for b in range(m.nbody):
        a = b
        while a not in (0, base):
            a = m.body_parentid[a]
        if a == base:
            robot.append(b)
    objs = {}
    for oid in object_ids:
        b = m.body(oid).id
        objs[oid] = ObjectIds(body=b, geom=m.geom(f"{oid}_geom").id, qadr=int(m.jnt_qposadr[m.body_jntadr[b]]),
                              dadr=int(m.jnt_dofadr[m.body_jntadr[b]]))
    return SceneIds(
        arm_qadr=np.array([jq(j) for j in rb.ARM_JOINTS]),
        arm_dadr=np.array([jd(j) for j in rb.ARM_JOINTS]),
        arm_act=np.array([m.actuator(a).id for a in rb.ARM_ACTUATORS]),
        finger_qadr=np.array([jq(j) for j in rb.FINGER_JOINTS]),
        grip_act=m.actuator(rb.GRIPPER_ACTUATOR).id,
        ee_site=m.site(rb.EE_SITE).id,
        hand=m.body(rb.HAND_BODY).id,
        camera=m.camera(0).id,
        table_body=m.body("table").id,
        floor_geom=m.geom("floor").id,
        robot_bodies=np.array(robot),
        finger_bodies=np.array([m.body(n).id for n in rb.FINGER_BODIES]),
        objects=objs,
    )


def set_initial_state(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, cfg: SceneConfig,
                      arm_q: np.ndarray | None = None) -> None:
    """Initialization / stopped-scene edit: arm at home (or arm_q), gripper open, objects placed."""
    mujoco.mj_resetData(m, d)
    d.qpos[ids.arm_qadr] = rb.HOME_Q if arm_q is None else arm_q
    d.qpos[ids.finger_qadr] = 0.04
    d.ctrl[ids.grip_act] = rb.GRIPPER_OPEN
    place_objects(m, d, ids, cfg.objects)
    mujoco.mj_forward(m, d)


def place_objects(m: mujoco.MjModel, d: mujoco.MjData, ids: SceneIds, objects: list[ObjectSpec]) -> None:
    for o in objects:
        oi = ids.objects.get(o.id)
        if oi is None:
            continue
        d.qpos[oi.qadr:oi.qadr + 3] = o.pos
        q = np.asarray(o.quat, float)
        d.qpos[oi.qadr + 3:oi.qadr + 7] = q / (np.linalg.norm(q) or 1.0)
        d.qvel[oi.dadr:oi.dadr + 6] = 0.0
