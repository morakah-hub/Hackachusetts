"""Ground-truth outcome evaluator for one pick-and-place leg.

Reads simulator truth (object poses, contacts) to score a leg and to display the world. It is never an
input to perception or to the task supervisor's decisions.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import SceneIds

PLACE_TOL_XY = 0.015
PLACE_TOL_Z = 0.006
TILT_TOL_DEG = 10.0
SETTLE_SPEED = 0.01
NUDGE_TOL = 0.01


class Evaluator:
    def __init__(self, model: mujoco.MjModel, ids: SceneIds):
        self.m, self.ids = model, ids
        self.robot = np.zeros(model.nbody, bool)
        self.robot[ids.robot_bodies] = True
        self.fingers = np.zeros(model.nbody, bool)
        self.fingers[ids.finger_bodies] = True
        self.names = {o.body: oid for oid, o in ids.objects.items()}
        self.target: int | None = None
        self.reset(None, None)

    def reset(self, data: mujoco.MjData | None, object_id: str | None) -> None:
        self.target = self.ids.objects[object_id].body if object_id else None
        self.z0 = None if data is None or self.target is None else float(data.xpos[self.target][2])
        self.max_lift = 0.0
        self.forbidden = 0
        self.forbidden_first: str | None = None
        self.carry_ref = None
        self.max_slip = 0.0
        self.others0 = {} if data is None else {b: data.xpos[b].copy() for b in self.names if b != self.target}

    def _name(self, b: int) -> str:
        return "the floor" if b == 0 else self.names.get(b, self.m.body(b).name)

    def step(self, d: mujoco.MjData, phase: str) -> None:
        if self.target is None:
            return
        p = d.xpos[self.target]
        if self.z0 is None:
            self.z0 = float(p[2])
        self.max_lift = max(self.max_lift, float(p[2] - self.z0))
        if phase in ("lift", "transport", "lower"):
            h = self.ids.hand
            rel = d.xmat[h].reshape(3, 3).T @ (p - d.xpos[h])
            if self.carry_ref is None:
                self.carry_ref = rel
            self.max_slip = max(self.max_slip, float(np.linalg.norm(rel - self.carry_ref)))
        m = self.m
        for c in d.contact[: d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            bad = None
            for a, b in ((b1, b2), (b2, b1)):
                if self.robot[a] and not self.robot[b]:
                    if b == self.target and self.fingers[a]:
                        continue
                    bad = f"{m.body(a).name} touched {self._name(b)}"
                if a in self.names and b == 0:
                    bad = f"{self._name(a)} touched the floor"
            if bad:
                self.forbidden += 1
                if self.forbidden_first is None:
                    self.forbidden_first = f"{bad} during {phase} (t = {d.time:.2f} s)"
                break

    def touching_fingers(self, d: mujoco.MjData) -> bool:
        m = self.m
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.target in (b1, b2) and (self.fingers[b1] or self.fingers[b2]):
                return True
        return False

    def final(self, d: mujoco.MjData, target_xyz) -> dict:
        b = self.target
        p = d.xpos[b].copy()
        R = d.xmat[b].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1))))
        err = p - np.asarray(target_xyz, float)
        exy = float(np.linalg.norm(err[:2]))
        oi = next(o for o in self.ids.objects.values() if o.body == b)
        speed = float(np.linalg.norm(d.qvel[oi.dadr:oi.dadr + 3]))
        reasons = []
        if exy > PLACE_TOL_XY:
            reasons.append(f"object center is {1e3 * exy:.1f} mm from the target (tolerance {1e3 * PLACE_TOL_XY:.0f} mm)")
        if abs(err[2]) > PLACE_TOL_Z:
            reasons.append(f"height error {1e3 * err[2]:.1f} mm (not resting where planned)")
        if tilt > TILT_TOL_DEG:
            reasons.append(f"object tilted {tilt:.1f} deg")
        if speed > SETTLE_SPEED:
            reasons.append(f"object still moving ({1e3 * speed:.1f} mm/s)")
        if self.touching_fingers(d):
            reasons.append("object still touching the gripper")
        nudged = {}
        for ob, p0 in self.others0.items():
            dist = float(np.linalg.norm(d.xpos[ob][:2] - p0[:2]))
            if dist > NUDGE_TOL:
                nudged[self._name(ob)] = dist
        return {"placed": not reasons, "reasons": reasons, "placement_error_m": float(np.linalg.norm(err)),
                "placement_error_xy_m": exy, "placement_error_z_m": float(err[2]), "final_tilt_deg": tilt,
                "final_position_m": p.tolist(), "max_lift_m": self.max_lift,
                "object_slip_m": self.max_slip if self.carry_ref is not None else None,
                "forbidden_contacts": self.forbidden, "first_forbidden_contact": self.forbidden_first,
                "nudged_m": nudged,
                "tolerances": {"xy_m": PLACE_TOL_XY, "z_m": PLACE_TOL_Z, "tilt_deg": TILT_TOL_DEG}}
