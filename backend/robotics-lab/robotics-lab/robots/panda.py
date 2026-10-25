"""Franka Emika Panda description backed by the official MuJoCo Menagerie model.

The MJCF lives unmodified in robots/models/franka_emika_panda (see SOURCE.txt).
Two arm actuation variants are built from it in code (MjSpec):

* "position" (Menagerie default): actuators 1-7 are position servos. ctrl is a
  joint-angle setpoint and MuJoCo computes tau = kp * (ctrl - q) - kd * qdot,
  clipped to the actuator force range.
* "torque": actuators 1-7 become direct joint-torque motors. ctrl is the joint
  torque in Nm (gain 1, no bias), limited to the same physical force range
  (87 Nm joints 1-4, 12 Nm joints 5-7). Geometry, inertia, armature, joint
  damping and joint limits are unchanged.

Actuator 8 drives the gripper in both variants (0 = closed, 255 = open). It is a
position servo on the "split" tendon L = (f1 + f2) / 2 (finger joints f1, f2, each
0..0.04 m): force = kp (0.04 ctrl / 255 - L) - kv dL/dt, clipped to +-100 N, and each
finger receives half of it. The Menagerie gains (kp = 100 N/m, kv = 10 N s/m) squeeze a
50 mm box with only 2.5 N (1.25 N per finger), enough to hold ~0.25 kg with friction 1.
`gripper_stiffness` (v0.6, opt-in) raises kp (kv = `gripper_damping`) so the closed
command squeezes like the real Franka Hand (continuous grasp force 70 N); ctrl range,
force range, tendon and fingers are unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models" / "franka_emika_panda"
MODEL_XML = MODEL_DIR / "panda.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
GRIPPER_OPEN = 255.0
BASE_BODY = "link0"
EE_BODY = "hand"
EE_SITE = "ee_site"
# Point midway between the fingertips, measured from the hand body origin.
EE_SITE_OFFSET = (0.0, 0.0, 0.1034)

# Menagerie "home" keyframe for the arm joints.
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

ACTUATIONS = ("position", "torque")

# Maximum joint speeds (rad/s) from the Franka Emika Panda datasheet. The
# Menagerie MJCF does not model them; the MPC uses them as constraints and the
# benchmark reports violations.
VELOCITY_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
# Joint acceleration (rad/s^2) and jerk (rad/s^3) limits from the Franka Control
# Interface documentation ("Limits for Panda"). Used by the v0.4 trajectory
# generator (tasks/trajectory.py), usually scaled down by configs/*.yaml.
ACCELERATION_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
JERK_LIMITS = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])


@dataclass(frozen=True)
class Panda:
    """Name-level description of the Panda used by the engine and controllers."""

    arm_joints: tuple[str, ...] = ARM_JOINTS
    arm_actuators: tuple[str, ...] = ARM_ACTUATORS
    gripper_actuator: str = GRIPPER_ACTUATOR
    base_body: str = BASE_BODY
    ee_site: str = EE_SITE
    home_qpos: tuple[float, ...] = tuple(HOME_QPOS)
    velocity_limits: tuple[float, ...] = tuple(VELOCITY_LIMITS)
    acceleration_limits: tuple[float, ...] = tuple(ACCELERATION_LIMITS)
    jerk_limits: tuple[float, ...] = tuple(JERK_LIMITS)
    actuation: str = "position"
    gripper_stiffness: float | None = None  # N/m on the split tendon; None = Menagerie (100 N/m)
    gripper_damping: float | None = None    # N s/m; None = Menagerie (10 N s/m)

    def __post_init__(self):
        if self.actuation not in ACTUATIONS:
            raise ValueError(f"actuation must be one of {ACTUATIONS}, got '{self.actuation}'")

    @property
    def dof(self) -> int:
        return len(self.arm_joints)

    def load_spec(self) -> mujoco.MjSpec:
        """Load the Menagerie MJCF as an editable spec and add an end-effector site."""
        if not MODEL_XML.is_file():
            raise FileNotFoundError(f"Panda model not found at {MODEL_XML}")
        spec = mujoco.MjSpec.from_file(str(MODEL_XML))
        spec.body(EE_BODY).add_site(
            name=self.ee_site,
            pos=EE_SITE_OFFSET,
            size=[0.005, 0, 0],
            rgba=[0, 1, 0, 0.5],
            group=4,
        )
        if self.actuation == "torque":
            self._make_torque_actuators(spec)
        if self.gripper_stiffness is not None or self.gripper_damping is not None:
            self._set_gripper_gains(spec)
        return spec

    def _set_gripper_gains(self, spec: mujoco.MjSpec) -> None:
        """Stiffer gripper servo (same ctrl and force range): ctrl 0..255 maps to L = 0..0.04 m."""
        act = spec.actuator(self.gripper_actuator)
        kp = 100.0 if self.gripper_stiffness is None else float(self.gripper_stiffness)
        kv = 10.0 if self.gripper_damping is None else float(self.gripper_damping)
        gain = np.array(act.gainprm, dtype=float)
        bias = np.array(act.biasprm, dtype=float)
        gain[0] = kp * 0.04 / 255.0
        bias[:3] = [0.0, -kp, -kv]
        act.gainprm, act.biasprm = gain, bias

    def _make_torque_actuators(self, spec: mujoco.MjSpec) -> None:
        """Turn the arm position servos into torque motors with the same force limits."""
        key = spec.key("home")
        key_ctrl = np.array(key.ctrl)
        for i, name in enumerate(self.arm_actuators):
            act = spec.actuator(name)
            limit = np.array(act.forcerange, dtype=float)
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm = np.r_[1.0, np.zeros(len(act.gainprm) - 1)]
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.biasprm = np.zeros(len(act.biasprm))
            act.ctrlrange = limit
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = limit
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            key_ctrl[i] = 0.0  # the keyframe's joint-angle setpoints would otherwise become torques
        key.ctrl = key_ctrl
