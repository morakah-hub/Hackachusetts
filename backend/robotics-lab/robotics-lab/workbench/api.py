"""Shared execution interface of the workbench (v0.7): requests, capabilities, presets, results.

Everything that starts a run (the browser UI, the command console, the Python API, tests) builds a
`RunRequest`, which `validate()` checks against the capability registry and `resolve()` turns into
the *same* configuration dictionaries the CLIs use (`configs/*.yaml` + overrides). Execution then goes
through the existing backends: `SimulationRunner` (reaching, main.py) and `PickPlaceEpisode`
(pick_place.py). Nothing here plans, controls, steps physics or verifies success.

Units: metres, kilograms, seconds, newtons; positions in the world frame (origin on the floor below the
robot base, z up; the table top is at z = 0.40 m and the robot base sits on it at the origin in x, y).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from simulation.runner import PROJECT_ROOT, load_config

CONFIGS = PROJECT_ROOT / "configs"


class RequestError(ValueError):
    """An invalid or unsupported request; the message says why and what is allowed."""


# ------------------------------------------------------------------ presets (named, reproducible)

CONTROLLERS = {
    "torque_mpc": "Torque MPC (OSQP, 50 Hz QP, 25-step / 0.5 s horizon; CTC fallback) - configs/default.yaml",
    "torque_ctc": "Computed-torque control (500 Hz, wn = 25 rad/s, zeta = 1) - configs/default.yaml",
    "joint_pid": "Joint PID outer loop on the Menagerie position servos (500 Hz) - configs/default.yaml",
    "joint_mpc": "Servo-setpoint MPC (v0.2, 50 Hz) on the position servos - configs/default.yaml",
}
TORQUE_CONTROLLERS = {"torque_mpc", "torque_ctc"}

CONTACT_PRESETS = {
    "v0.6-working": {"physics": {"cone": "elliptic", "impratio": 10},
                     "label": "v0.6 working contact solver: elliptic friction cone, impratio 10 (configs/pick_place.yaml)"},
    "mujoco-default": {"physics": {"cone": "pyramidal", "impratio": 1},
                       "label": "MuJoCo default pyramidal cone, impratio 1 - the v0.6 sensitivity condition "
                                "under which the heavy block slipped out (docs/research/v0.6-results.md)"},
}
GRIP_PRESETS = {
    "v0.6-strong": {"robot": {"gripper_stiffness": 1500.0, "gripper_damping": 20.0},
                    "label": "v0.6 gripper: 1500 N/m tendon servo, ~37 N squeeze on a 50 mm box"},
    "menagerie-weak": {"robot": {"gripper_stiffness": 100.0, "gripper_damping": 10.0},
                       "label": "Menagerie gains: 100 N/m, ~2.5 N squeeze (the v0.6 negative control)"},
}

# ------------------------------------------------------------------ capability registry

TABLE_XY = {"x": (0.30, 0.68), "y": (-0.32, 0.32)}   # supported object/target region on the table (m)
REACH_BOX = {"x": (0.20, 0.80), "y": (-0.50, 0.50), "z": (0.45, 1.10)}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                     # "reach" | "pick_place"
    title: str
    config: str                   # base config file in configs/
    environment: str
    controllers: tuple[str, ...]
    options: tuple[str, ...]      # request fields this scenario supports
    obstacle: str | None = None   # environment geom that `obstacle` edits
    description: str = ""


SCENARIOS = {
    "wall_reach": Scenario(
        "wall_reach", "reach", "Reach over a wall (v0.5 planning demo)", "planning_demo.yaml", "planning_demo",
        ("torque_mpc", "torque_ctc", "joint_pid", "joint_mpc"),
        ("controller", "observer", "targets", "obstacle", "plan_seed", "force_pulse", "noise"), "wall",
        "Collision-aware planning (RRT-Connect, 20 mm clearance) to Cartesian targets on both sides of a wall."),
    "table_reach": Scenario(
        "table_reach", "reach", "Reach targets above an empty table", "planning_demo.yaml", "table",
        ("torque_mpc", "torque_ctc", "joint_pid", "joint_mpc"),
        ("controller", "observer", "targets", "plan_seed", "force_pulse", "noise"), None,
        "Planned reaching without obstacles (the planner returns the direct edge when it is free)."),
    "pick_place": Scenario(
        "pick_place", "pick_place", "Pick and place on an open table (v0.6)", "pick_place.yaml", "pick_place",
        ("torque_ctc", "torque_mpc"),
        ("controller", "observer", "payload_kg", "object_xy", "place_xy", "plan_seed", "contact_preset",
         "grip_preset", "force_pulse", "noise"), None,
        "Known-world physical pick-and-place: top-down grasp, lift, carry, place within 10 mm."),
    "pick_place_obstacle": Scenario(
        "pick_place_obstacle", "pick_place", "Pick and place around an obstacle (v0.6)", "pick_place.yaml",
        "pick_place_obstacle", ("torque_ctc", "torque_mpc"),
        ("controller", "observer", "payload_kg", "object_xy", "place_xy", "plan_seed", "contact_preset",
         "grip_preset", "obstacle", "force_pulse", "noise"), "obstacle",
        "As pick_place, with a static box between the pick and place regions."),
}

UNSUPPORTED = {
    ("pick_place", "joint_pid"): "pick-and-place is validated only with the torque-level controllers "
                                 "(torque_ctc, torque_mpc); the position-servo controllers were never tested with "
                                 "grasping",
}

DEFAULT_TARGETS = {"wall_reach": [[0.45, -0.25, 0.50], [0.45, 0.25, 0.50]],
                   "table_reach": [[0.50, -0.20, 0.60], [0.45, 0.25, 0.70]]}
DEFAULT_OBSTACLES = {"wall_reach": {"pos": [0.525, 0.0, 0.53], "size": [0.275, 0.02, 0.13]},
                     "pick_place_obstacle": {"pos": [0.5, 0.0, 0.49], "size": [0.08, 0.02, 0.09]}}
OBJECT_HALF = (0.025, 0.025, 0.03)
BODIES_FOR_FORCE = ("hand", "link4", "link5", "link6", "link7", "object")


# ------------------------------------------------------------------ request


@dataclass
class RunRequest:
    scenario: str = "pick_place"
    controller: str = "torque_mpc"
    observer: bool = True
    payload_kg: float = 0.40
    object_xy: list[float] = field(default_factory=lambda: [0.50, 0.20])
    place_xy: list[float] = field(default_factory=lambda: [0.50, -0.20])
    targets: list[list[float]] | None = None          # reach scenarios; None = scenario defaults
    obstacle: dict | None = None                      # {"pos": [x, y, z], "size": [hx, hy, hz]} (box centre, half sizes)
    plan_seed: int = 0
    contact_preset: str = "v0.6-working"
    grip_preset: str = "v0.6-strong"
    force_pulse: dict | None = None                   # {"body", "direction": [x,y,z], "magnitude_n", "start_s", "duration_s"}
    noise: dict | None = None                         # {"position_std", "velocity_std", "seed"}
    mpc_snapshots: bool = True                        # record MPC updates (bounded) for the inspector
    label: str = ""

    # -------------------------------------------------------------- helpers

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRequest":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise RequestError(f"unknown request fields: {sorted(unknown)}")
        return cls(**copy.deepcopy(data))

    def replace(self, **changes) -> "RunRequest":
        d = self.to_dict()
        d.update(changes)
        return RunRequest.from_dict(d)

    @property
    def spec(self) -> Scenario:
        if self.scenario not in SCENARIOS:
            raise RequestError(f"unknown scenario '{self.scenario}'; available: {sorted(SCENARIOS)}")
        return SCENARIOS[self.scenario]

    # -------------------------------------------------------------- validation

    def validate(self) -> "RunRequest":
        """Raise RequestError with a reason if the request is unsupported; return self otherwise."""
        sc = self.spec
        if self.controller not in CONTROLLERS:
            raise RequestError(f"unknown controller '{self.controller}'; available: {sorted(CONTROLLERS)}")
        if self.controller not in sc.controllers:
            why = UNSUPPORTED.get((sc.kind, "joint_pid"), "not supported for this scenario")
            raise RequestError(f"{self.controller} is not available in '{sc.name}': {why}")
        if self.observer and self.controller not in TORQUE_CONTROLLERS:
            raise RequestError("the disturbance observer exists only for the torque controllers (torque_ctc, torque_mpc)")
        if self.mpc_snapshots and self.controller != "torque_mpc":
            self.mpc_snapshots = False  # nothing to record; not an error
        if self.contact_preset not in CONTACT_PRESETS:
            raise RequestError(f"unknown contact preset '{self.contact_preset}'; available: {sorted(CONTACT_PRESETS)}")
        if self.grip_preset not in GRIP_PRESETS:
            raise RequestError(f"unknown grip preset '{self.grip_preset}'; available: {sorted(GRIP_PRESETS)}")
        if sc.kind == "pick_place":
            if not 0.05 <= float(self.payload_kg) <= 2.0:
                raise RequestError(f"payload {self.payload_kg} kg outside the supported 0.05-2.0 kg "
                                   f"(gripper ~37 N squeeze, friction 1)")
            for name in ("object_xy", "place_xy"):
                xy = _vec(getattr(self, name), 2, name)
                _in_box(xy, TABLE_XY, name)
            o, p = np.asarray(self.object_xy, float), np.asarray(self.place_xy, float)
            if np.linalg.norm(o - p) < 0.08:
                raise RequestError("object and place positions must be at least 80 mm apart")
            obs = self.resolved_obstacle()
            if obs is not None:
                for name, xy in (("object", o), ("place target", p)):
                    if _overlaps(xy, OBJECT_HALF, obs, margin=0.03):
                        raise RequestError(f"{name} at {xy.round(3).tolist()} overlaps the obstacle (with 30 mm "
                                           f"margin); move it or the obstacle")
        else:
            for k, tgt in enumerate(self.resolved_targets()):
                _in_box(_vec(tgt, 3, f"target {k + 1}"), REACH_BOX, f"target {k + 1}")
        if self.obstacle is not None:
            if sc.obstacle is None:
                raise RequestError(f"scenario '{sc.name}' has no editable obstacle")
            pos = _vec(self.obstacle.get("pos"), 3, "obstacle pos")
            size = _vec(self.obstacle.get("size"), 3, "obstacle size (half extents)")
            if np.any(size < 0.005) or np.any(size > 0.40):
                raise RequestError("obstacle half extents must be 5-400 mm")
            if pos[2] - size[2] < 0.399:
                raise RequestError("the obstacle must rest on or above the table top (z - half height >= 0.40 m)")
            _in_box(pos[:2], {"x": (0.15, 0.85), "y": (-0.45, 0.45)}, "obstacle centre")
        if self.force_pulse is not None:
            fp = self.force_pulse
            if fp.get("body") not in BODIES_FOR_FORCE or (fp.get("body") == "object" and sc.kind != "pick_place"):
                raise RequestError(f"force pulse body must be one of {list(BODIES_FOR_FORCE)} (object: pick-and-place only)")
            d = _vec(fp.get("direction"), 3, "force direction")
            if np.linalg.norm(d) < 1e-9:
                raise RequestError("force direction must be non-zero")
            if not 0 < float(fp.get("magnitude_n", 0)) <= 100:
                raise RequestError("force magnitude must be in (0, 100] N")
            if float(fp.get("start_s", -1)) < 0 or not 0 < float(fp.get("duration_s", 0)) <= 5:
                raise RequestError("force start must be >= 0 s and duration in (0, 5] s")
        if self.noise is not None:
            for k in ("position_std", "velocity_std"):
                v = float(self.noise.get(k, 0.0))
                if not 0 <= v <= 0.05:
                    raise RequestError(f"noise {k} must be in [0, 0.05] (rad or rad/s)")
        return self

    def resolved_targets(self) -> list[list[float]]:
        return [list(map(float, t)) for t in (self.targets or DEFAULT_TARGETS.get(self.scenario, []))]

    def resolved_obstacle(self) -> dict | None:
        sc = self.spec
        if sc.obstacle is None:
            return None
        return self.obstacle or DEFAULT_OBSTACLES.get(sc.name)

    # -------------------------------------------------------------- resolution to backend configs

    def resolve(self) -> dict:
        """The backend configuration this request means: {"kind", "config", "case"?, "notes"}."""
        self.validate()
        sc = self.spec
        cfg = load_config(CONFIGS / sc.config)
        cfg["environment"] = sc.environment
        cfg["controller"]["type"] = self.controller
        for sec in ("ctc", "torque_mpc"):
            cfg["controller"].setdefault(sec, {}).setdefault("disturbance_observer", {})["enabled"] = bool(self.observer)
        if self.noise and (self.noise.get("position_std") or self.noise.get("velocity_std")):
            cfg["sensor_noise"] = {"position_std": float(self.noise.get("position_std", 0.0)),
                                   "velocity_std": float(self.noise.get("velocity_std", 0.0)),
                                   "seed": int(self.noise.get("seed", 0))}
        obs = self.obstacle
        out = {"kind": sc.kind, "scenario": sc.name, "config": cfg}
        if sc.kind == "reach":
            task = cfg["task"]
            task["targets"] = [{"name": f"target{k + 1}", "position": t} for k, t in enumerate(self.resolved_targets())]
            task["planning"]["seed"] = int(self.plan_seed)
            if obs is not None:
                cfg["environment_overrides"] = {"geoms": {sc.obstacle: {"pos": [0, 0, 0], "size": obs["size"]}},
                                                "bodies": {sc.obstacle: {"pos": obs["pos"]}}}
            return out
        cfg["physics"] = dict(CONTACT_PRESETS[self.contact_preset]["physics"])
        cfg["robot"] = dict(GRIP_PRESETS[self.grip_preset]["robot"])
        if obs is not None and self.obstacle is not None:
            cfg["environment_overrides"] = {"geoms": {sc.obstacle: {"pos": [0, 0, 0], "size": obs["size"]}},
                                            "bodies": {sc.obstacle: {"pos": obs["pos"]}}}
        cfg["label"] = f"{'mpc' if self.controller == 'torque_mpc' else 'ctc'}_{'obs' if self.observer else 'noobs'}"
        out["case"] = {"name": self.label or "workbench", "environment": sc.environment,
                       "object": [float(v) for v in self.object_xy], "place": [float(v) for v in self.place_xy],
                       "mass": float(self.payload_kg), "seed": int(self.plan_seed)}
        return out

    def summary(self) -> list[str]:
        """Human-readable resolved settings (shown next to every command)."""
        sc = self.spec
        lines = [f"scenario {sc.name}: {sc.title}", f"controller {self.controller}: {CONTROLLERS[self.controller]}"]
        if self.controller in TORQUE_CONTROLLERS:
            lines.append(f"disturbance observer {'ON (T = 10 ms)' if self.observer else 'OFF'}")
        if sc.kind == "pick_place":
            lines += [f"payload {float(self.payload_kg):.3f} kg (unknown to the controller)",
                      f"object at (x, y) = ({self.object_xy[0]:.3f}, {self.object_xy[1]:.3f}) m, "
                      f"place at ({self.place_xy[0]:.3f}, {self.place_xy[1]:.3f}) m",
                      f"contact preset {self.contact_preset}: {CONTACT_PRESETS[self.contact_preset]['label']}",
                      f"grip preset {self.grip_preset}: {GRIP_PRESETS[self.grip_preset]['label']}",
                      "grasp policy: prescribed top-down grasp, finger axis along world x (not editable)"]
        else:
            lines.append("targets " + ", ".join(f"({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})" for t in self.resolved_targets())
                         + " m")
        obs = self.resolved_obstacle()
        if obs is not None:
            lines.append(f"obstacle centre {np.round(obs['pos'], 3).tolist()} m, half sizes "
                         f"{np.round(obs['size'], 3).tolist()} m{'' if self.obstacle else ' (scene default)'}")
        lines.append(f"planning seed {self.plan_seed}")
        if self.force_pulse:
            fp = self.force_pulse
            lines.append(f"force pulse {fp['magnitude_n']} N on {fp['body']} along {fp['direction']} (world), "
                         f"t = {fp['start_s']} s for {fp['duration_s']} s")
        if self.noise and (self.noise.get("position_std") or self.noise.get("velocity_std")):
            lines.append(f"arm measurement noise: q {self.noise.get('position_std', 0)} rad, qd "
                         f"{self.noise.get('velocity_std', 0)} rad/s (seed {self.noise.get('seed', 0)})")
        return lines


def _vec(value, n: int, name: str) -> np.ndarray:
    try:
        v = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise RequestError(f"{name} must be {n} numbers") from None
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise RequestError(f"{name} must be {n} finite numbers, got {value}")
    return v


def _in_box(v, box: dict, name: str) -> None:
    for i, axis in enumerate(box):
        lo, hi = box[axis]
        if not lo <= v[i] <= hi:
            raise RequestError(f"{name} {axis} = {v[i]:.3f} m is outside the supported range [{lo}, {hi}] m")


def _overlaps(xy, half, obstacle: dict, margin: float) -> bool:
    c, s = np.asarray(obstacle["pos"], float), np.asarray(obstacle["size"], float)
    return bool(np.all(np.abs(np.asarray(xy) - c[:2]) < s[:2] + np.asarray(half[:2]) + margin))


# ------------------------------------------------------------------ status / result


@dataclass
class RunStatus:
    state: str = "idle"           # idle | building | ready | running | paused | finished | failed | replay
    run_id: str | None = None
    sim_time: float = 0.0
    phase: str = ""
    message: str = ""
    active_controller: str = ""
    steps: int = 0


@dataclass
class RunResult:
    run_id: str
    request: dict
    success: bool | None
    outcome: str                  # "success" | "failure" | "stopped" | "completed" (reach)
    failure_stage: str | None
    failure_reason: str | None
    metrics: dict
    sim_time_s: float
    wall_time_s: float
    directory: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def capabilities() -> dict:
    """Registry as plain data for the UI."""
    return {
        "scenarios": {k: {**asdict(v), "default_targets": DEFAULT_TARGETS.get(k), "default_obstacle":
                          DEFAULT_OBSTACLES.get(k)} for k, v in SCENARIOS.items()},
        "controllers": CONTROLLERS, "contact_presets": {k: v["label"] for k, v in CONTACT_PRESETS.items()},
        "grip_presets": {k: v["label"] for k, v in GRIP_PRESETS.items()},
        "bounds": {"table_xy": TABLE_XY, "reach_box": REACH_BOX, "payload_kg": [0.05, 2.0]},
        "force_bodies": list(BODIES_FOR_FORCE), "object_half_size": OBJECT_HALF,
    }
