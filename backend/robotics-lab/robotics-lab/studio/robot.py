"""Robot profile and the rules for what this arm and gripper can pick up.

All limits come from configs/studio.yaml (see the comments there for where each number comes from). The
object model is a box proxy: size = (w, d, h) along the object's own x, y, z axes, upright on its base.
"""

from dataclasses import dataclass

import numpy as np

from studio.settings import StudioSettings


def grasp_depth(height: float, gd: dict) -> float:
    """How far below the object top the end-effector site (between the fingertip pads) closes."""
    return float(np.clip(0.5 * height - gd["offset_m"], gd["min_m"], gd["max_m"]))


def wrap_angle(a: float) -> float:
    """Angle in [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def yaw_of_quat(quat) -> float:
    """Rotation about world z of a (w, x, y, z) quaternion's x axis."""
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
        return {"code": self.code, "message": self.message, "hint": self.hint, "object_id": self.object_id}


@dataclass
class GraspOption:
    yaw: float          # gripper finger axis angle from world x (one of the two equivalent representations)
    width: float        # object side across the fingers (m)


class RobotRules:
    """Object limits for this robot, and the grasp options for an object."""

    def __init__(self, settings: StudioSettings):
        self.settings = settings
        s = settings.studio
        self.robot = s["robot"]
        self.lim = s["objects"]
        self.gd = s["grasp_depth"]
        self.payload = float(self.robot["payload_kg"])

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
                                   f"The fingers open to {1e3 * self.robot['gripper_stroke_m']:.0f} mm and need "
                                   f"room to let go.", object_id))
            elif max(w, d) < lo:
                out.append(Problem("too_thin", f"{name} is thinner than {1e3 * lo:.0f} mm on every side.", "",
                                   object_id))
            else:
                out.append(Problem("no_grip_side", f"Neither side of {name} fits the gripper: one is thinner than "
                                                   f"{1e3 * lo:.0f} mm, the other wider than {1e3 * hi:.0f} mm.", "",
                                   object_id))
        if h < lim["min_height_m"]:
            out.append(Problem("too_short", f"{name} is {1e3 * h:.0f} mm tall; the fingertips need at least "
                                            f"{1e3 * lim['min_height_m']:.0f} mm to close around it.", "", object_id))
        if h > lim["max_height_m"]:
            out.append(Problem("too_tall", f"{name} is {1e3 * h:.0f} mm tall; objects up to "
                                           f"{1e3 * lim['max_height_m']:.0f} mm are supported.", "", object_id))
        return out

    def shape_problems(self, name: str, size, mass: float) -> list[Problem]:
        """Why an object definition is invalid (cannot even be placed)."""
        w, d, h = (float(v) for v in size)
        out = []
        if not all(np.isfinite(v) and v > 0 for v in (w, d, h)):
            out.append(Problem("bad_size", f"{name} needs a positive width, depth and height."))
        elif max(w, d) > self.lim["max_footprint_m"]:
            out.append(Problem("too_large", f"{name} is {1e3 * max(w, d):.0f} mm long; the table accepts objects up "
                                            f"to {1e3 * self.lim['max_footprint_m']:.0f} mm.", ""))
        if not np.isfinite(mass) or mass < self.lim["min_mass_kg"]:
            out.append(Problem("bad_mass", f"{name} needs a mass of at least {self.lim['min_mass_kg']:.2f} kg."))
        return out

    def grasp_options(self, size, yaw: float) -> list[GraspOption]:
        """Finger axis directions that fit the object, narrowest side first. Each physical grasp has two
        representations (yaw and yaw + pi, the hand turned around); both are returned."""
        w, d, _ = (float(v) for v in size)
        cands = [(yaw, w), (yaw + np.pi / 2, d)]
        cands = [c for c in cands if self.lim["min_grasp_width_m"] <= c[1] <= self.lim["max_grasp_width_m"]]
        cands.sort(key=lambda c: c[1])
        out = []
        for g, width in cands:
            g = wrap_angle(g)
            for rep in (g, wrap_angle(g + np.pi)):
                out.append(GraspOption(rep, width))
        return out

    def depth(self, height: float) -> float:
        return grasp_depth(height, self.gd)

    def profile(self) -> dict:
        r = self.robot
        return {"id": r["id"], "name": r["name"], "payload_kg": self.payload,
                "datasheet_reach_m": r["datasheet_reach_m"], "gripper_stroke_m": r["gripper_stroke_m"],
                "dof": 7, "limits": dict(self.lim)}
