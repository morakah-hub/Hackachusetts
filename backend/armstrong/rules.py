"""Robot profile and the rules for what this arm and gripper can pick up (from Workcell Studio).

The object model is a box proxy: size = (w, d, h) along the object's own x, y, z axes, upright on its base.
Numbers: Franka Emika Panda datasheet (3 kg payload, 855 mm reach, 80 mm hand stroke). The fingers must
open to the object width + 20 mm to let go, so the side across the fingers is at most 58 mm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import scene as sc

ROBOT = {"id": "panda", "name": "Franka Emika Panda", "payload_kg": 3.0, "datasheet_reach_m": 0.855,
         "gripper_stroke_m": 0.08}
LIMITS = {"min_grasp_width_m": 0.015, "max_grasp_width_m": 0.058, "min_height_m": 0.030, "max_height_m": 0.200,
          "max_footprint_m": 0.300, "min_mass_kg": 0.01, "max_tilt_deg": 10.0, "gap_m": 0.015}
GRASP_DEPTH = {"offset_m": 0.015, "min_m": 0.015, "max_m": 0.028}
PLACE_TOL_MM = 15.0


def grasp_depth(height: float) -> float:
    """How far below the object top the end-effector site (between the fingertip pads) closes."""
    gd = GRASP_DEPTH
    return float(np.clip(0.5 * height - gd["offset_m"], gd["min_m"], gd["max_m"]))


def wrap_angle(a: float) -> float:
    """Angle in [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def yaw_of_quat(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tilt_of_quat(quat) -> float:
    """Angle (deg) between the object's z axis and world z."""
    w, x, y, z = (float(v) for v in quat)
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def quat_from_yaw(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


@dataclass
class Problem:
    """A reason something cannot be done, written for the person using the app."""

    code: str
    message: str
    hint: str = ""
    object_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class World:
    table_height: float = sc.TABLE_TOP_Z
    table_radius: float = sc.TABLE_RADIUS
    r_in: float = sc.R_IN
    r_out: float = sc.R_OUT
    theta_max_deg: float = sc.THETA_MAX_DEG

    def to_dict(self) -> dict:
        return asdict(self)

    def zone_problem(self, x: float, y: float, name: str) -> Problem | None:
        r, th = float(np.hypot(x, y)), float(np.degrees(np.arctan2(y, x)))
        if r < self.r_in:
            return Problem("too_close", f"{name} would be too close to the robot base.",
                           f"Keep objects at least {1e3 * self.r_in:.0f} mm from the centre of the base.")
        if r > self.r_out:
            return Problem("out_of_reach", f"{name} would be out of the arm's reach.",
                           f"The arm reaches {1e3 * self.r_out:.0f} mm from the centre of its base.")
        if abs(th) > self.theta_max_deg + 1e-6:
            return Problem("outside_zone", f"{name} would be outside the work area.",
                           "Keep objects in the shaded area in front of the robot, where the camera sees them.")
        return None


@dataclass
class GraspOption:
    angle: float        # world angle of the finger closing axis (one of two equivalent representations)
    width: float        # object side across the fingers (m)
    rel: float          # closing axis relative to the object's x axis (0 or pi/2)


class RobotRules:
    def __init__(self):
        self.robot = ROBOT
        self.lim = LIMITS
        self.payload = float(ROBOT["payload_kg"])

    def lift_problems(self, name: str, size, mass: float, object_id: str | None = None) -> list[Problem]:
        """Why the arm cannot pick this object up at all (independent of where it is)."""
        w, d, h = (float(v) for v in size)
        lim = self.lim
        out = []
        if mass > self.payload:
            out.append(Problem("too_heavy", f"{name} weighs {mass:.2f} kg, over the {self.payload:.1f} kg payload "
                                            f"of the {self.robot['name']}.",
                               "Use a lighter object or check the mass you entered.", object_id))
        lo, hi = lim["min_grasp_width_m"], lim["max_grasp_width_m"]
        if not any(lo <= side <= hi for side in (w, d)):
            narrow = min(w, d)
            if narrow > hi:
                out.append(Problem("too_wide", f"{name} is {1e3 * narrow:.0f} mm across at its narrowest side; the "
                                               f"gripper can take at most {1e3 * hi:.0f} mm.",
                                   "The fingers open to 80 mm and need room to let go.", object_id))
            elif max(w, d) < lo:
                out.append(Problem("too_thin", f"{name} is thinner than {1e3 * lo:.0f} mm on every side.", "",
                                   object_id))
            else:
                out.append(Problem("no_grip_side", f"Neither side of {name} fits the gripper.", "", object_id))
        if h < lim["min_height_m"]:
            out.append(Problem("too_short", f"{name} is {1e3 * h:.0f} mm tall; the fingertips need at least "
                                            f"{1e3 * lim['min_height_m']:.0f} mm to close around it.", "", object_id))
        if h > lim["max_height_m"]:
            out.append(Problem("too_tall", f"{name} is {1e3 * h:.0f} mm tall; objects up to "
                                           f"{1e3 * lim['max_height_m']:.0f} mm are supported.", "", object_id))
        return out

    def shape_problems(self, name: str, size, mass: float) -> list[Problem]:
        """Why an object definition is invalid (cannot even be placed)."""
        try:
            w, d, h = (float(v) for v in size)
        except (TypeError, ValueError):
            return [Problem("bad_size", f"{name} needs a width, depth and height.")]
        out = []
        if not all(np.isfinite(v) and v > 0 for v in (w, d, h)):
            out.append(Problem("bad_size", f"{name} needs a positive width, depth and height."))
        elif max(w, d) > self.lim["max_footprint_m"] or h > 0.4:
            out.append(Problem("too_large", f"{name} is too large for the table (at most "
                                            f"{1e3 * self.lim['max_footprint_m']:.0f} mm long, 400 mm tall)."))
        if not np.isfinite(mass) or mass < self.lim["min_mass_kg"]:
            out.append(Problem("bad_mass", f"{name} needs a mass of at least {self.lim['min_mass_kg']:.2f} kg."))
        return out

    def grasp_options(self, size, yaw: float) -> list[GraspOption]:
        """Finger closing-axis directions that fit the object, narrowest side first, each in both
        representations (the hand turned around by pi)."""
        w, d, _ = (float(v) for v in size)
        cands = [(0.0, w), (np.pi / 2, d)]
        cands = [c for c in cands if self.lim["min_grasp_width_m"] <= c[1] <= self.lim["max_grasp_width_m"]]
        cands.sort(key=lambda c: c[1])
        out = []
        for rel, width in cands:
            g = wrap_angle(yaw + rel)
            for rep in (g, wrap_angle(g + np.pi)):
                out.append(GraspOption(rep, width, rel))
        return out

    def profile(self) -> dict:
        r = self.robot
        return {**r, "dof": 7, "limits": dict(self.lim)}
