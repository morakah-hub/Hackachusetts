"""Run adapters, recording and persistence for the workbench (v0.7).

`LiveRun` wraps one backend run behind a step-wise interface:

    run = LiveRun(request)      # builds the engine / planner / controller (reaching plans here)
    while run.step():           # one physics step (pick-and-place: the supervisor may plan first)
        ...
    result = run.finish()       # RunResult; run.stop(reason) ends it early

* reaching: `SimulationRunner` (the class behind main.py), stepped with `runner.step()`;
* pick-and-place: `PickPlaceEpisode` (behind pick_place.py), stepped with `episode.advance()`.

Optional plant-side extras applied identically to both: a force pulse on a body (`xfrc_applied`, which the
controllers' private state copies exclude) and seeded measurement noise (configured in the backend config,
so the backends apply it themselves).

`Recorder` samples the measured state every `sample_every` physics steps and snapshots every torque-MPC
update (bounded). It only reads: controller attributes are copied, never modified. `Recording` is the
read side used by the inspectors, the explanations and replay; it is identical for live and saved runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from simulation.engine import PROJECT_ROOT
from simulation.manipulation import PHASES, PickPlaceEpisode
from simulation.runner import SimulationRunner
from workbench.api import RunRequest, RunResult

RUNS_DIR = PROJECT_ROOT / "results" / "workbench" / "runs"
REACH_PHASES = ("move", "hold")
SAMPLE_FIELDS = ("t", "q", "qd", "q_obs", "qd_obs", "q_ref", "qd_ref", "ctrl", "tau", "ee", "ee_ref", "d_hat",
                 "mpc_cmd", "feedthrough", "phase", "fallback", "contact", "forbidden", "force", "obj_pos",
                 "obj_quat", "grip_width", "grip_ctrl", "clearance", "qpos")


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"
    import osqp
    import scipy
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty_code": bool(git("status", "--porcelain", "--untracked-files=no", "--", "*.py", "configs",
                                       "environments", "workbench")),
            "python": sys.version.split()[0], "mujoco": mujoco.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "osqp": osqp.__version__, "platform": platform.platform()}


class Recorder:
    def __init__(self, run: "LiveRun", sample_every: int = 5, max_samples: int = 60000, max_snapshots: int = 2000):
        self.run = run
        self.sample_every = sample_every
        self.samples: dict[str, list] = {k: [] for k in SAMPLE_FIELDS}
        self.max_samples = max_samples
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self._contact_any = self._forbidden_any = False
        self._last_updates = 0
        self._last_phase = None
        e = run.engine
        self._scratch = mujoco.MjData(e.model)

    def _fk(self, q):
        e, s = self.run.engine, self._scratch
        s.qpos[:] = e.data.qpos
        s.qpos[e.arm_qpos_adr] = q
        mujoco.mj_kinematics(e.model, s)
        return s.site_xpos[e.ee_site_id].copy()

    def event(self, kind: str, text: str, **data) -> None:
        with self.lock:
            self.events.append({"t": float(self.run.engine.time), "kind": kind, "text": text, **data})

    def after_step(self) -> None:
        run, e = self.run, self.run.engine
        self._contact_any |= bool(e.data.ncon and e.robot_contacts())
        self._forbidden_any |= run.forbidden_this_step()
        phase = run.phase
        if phase != self._last_phase:
            self.event("phase", f"phase {phase}", phase=phase)
            self._last_phase = phase
        ctrl = run.controller
        stats = getattr(ctrl, "stats", None)
        if stats is not None and len(stats.updates) != self._last_updates:
            self._last_updates = len(stats.updates)
            if run.request.mpc_snapshots:
                self._snapshot(ctrl, stats.updates[-1])
            if stats.updates[-1].fallback:
                self.event("fallback", f"MPC update at t = {stats.updates[-1].time:.3f} s failed "
                                       f"({stats.updates[-1].status}); CTC fallback until the next success")
        if run.steps % self.sample_every == 0:
            self._sample()

    def _sample(self) -> None:
        run, e = self.run, self.run.engine
        ctrl = run.controller
        q, qd = e.arm_qpos(), e.arm_qvel()
        ref = run.reference()
        obs = run.observation()
        obj = run.object_pose()
        d_hat = getattr(ctrl, "disturbance_estimate", None)
        has_obs = getattr(ctrl, "observer", None) is not None
        mpc_cmd = getattr(ctrl, "_command", None)
        feed = None
        if getattr(ctrl, "disturbance_feedthrough", False) and has_obs and mpc_cmd is not None:
            feed = -(ctrl.observer.estimate - ctrl._d_update)
        row = {
            "t": e.time, "q": q, "qd": qd, "q_obs": obs[0], "qd_obs": obs[1], "q_ref": ref[0], "qd_ref": ref[1],
            "ctrl": e.data.ctrl[e.arm_actuator_ids].copy(), "tau": e.arm_actuator_force(),
            "ee": self._fk(q), "ee_ref": self._fk(ref[0]),
            "d_hat": d_hat if has_obs else np.full(7, np.nan),
            "mpc_cmd": np.full(7, np.nan) if mpc_cmd is None or getattr(ctrl, "using_fallback", False) else mpc_cmd.copy(),
            "feedthrough": np.full(7, np.nan) if feed is None else feed,
            "phase": run.phase, "fallback": bool(getattr(ctrl, "using_fallback", False)),
            "contact": self._contact_any, "forbidden": self._forbidden_any, "force": run.current_force.copy(),
            "obj_pos": obj[0], "obj_quat": obj[1], "grip_width": run.grip_width(),
            "grip_ctrl": float(e.data.ctrl[e.gripper_actuator_id]), "clearance": run.clearance(),
            "qpos": e.data.qpos.copy(),
        }
        self._contact_any = self._forbidden_any = False
        with self.lock:
            if len(self.samples["t"]) < self.max_samples:
                for k, v in row.items():
                    self.samples[k].append(v)

    def _snapshot(self, ctrl, upd) -> None:
        p = ctrl.last_prediction
        fresh = p is not None and abs(p["time"] - upd.time) < 1e-9 and not upd.fallback
        snap = {"time": float(upd.time), "status": upd.status, "iterations": upd.iterations,
                "solve_ms": 1e3 * upd.solve_time, "update_ms": 1e3 * upd.update_time, "fallback": bool(upd.fallback),
                "prim_res": upd.prim_res, "dual_res": upd.dual_res, "max_slack": upd.max_slack,
                "has_prediction": bool(fresh)}
        if fresh:
            N, nx, nu = ctrl.N, ctrl.nx, ctrl.nu
            z = ctrl._z_prev
            snap.update(x0=p["x0"].tolist(), states=p["states"].tolist(), commands=p["commands"].tolist(),
                        references=p["references"].tolist(), gravity=p["gravity"].tolist(),
                        disturbance=p["disturbance"].tolist(),
                        tau_prev=getattr(ctrl, "_tau_prev_at_update", p["gravity"]).tolist(),
                        slack=(z[N * (nx + nu):].reshape(N, nx).tolist() if ctrl.soft and z is not None else None))
        with self.lock:
            self.snapshots.append(snap)


class LiveRun:
    """One run of the existing backend, advanced one physics step at a time."""

    def __init__(self, request: RunRequest, run_id: str | None = None, sample_every: int = 5):
        self.request = request.validate()
        self.resolved = request.resolve()
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.kind = self.resolved["kind"]
        self.wall0 = time.perf_counter()
        self.current_force = np.zeros(3)
        self._force_body = None
        self.steps = 0
        self.done = False
        self.stopped_reason = None
        self.reach_contact_steps = 0
        self._hold_errors: dict[str, float] = {}
        if self.kind == "reach":
            self.runner = SimulationRunner(self.resolved["config"])   # plans every target here
            self.engine = self.runner.engine
            self.controller = self.runner.controller
            self.duration = self.runner.task.duration
            self.episode = None
            self._hold_ends = {p.name: self.runner.task.hold_end_time(p) for p in self.runner.task.reachable}
        else:
            self.episode = PickPlaceEpisode(self.resolved["config"], self.resolved["case"])
            self.engine = self.episode.engine
            self.controller = self.episode.controller
            self.episode.before_step = lambda ep: self._apply_force()
            self.episode.start()
            self.runner = None
        fp = request.force_pulse
        if fp:
            self._force_body = self.engine.model.body(fp["body"]).id
            d = np.asarray(fp["direction"], float)
            self._force_vec = float(fp["magnitude_n"]) * d / np.linalg.norm(d)
        self.control_period = float(getattr(self.controller, "control_period", self.engine.dt))
        self.recorder = Recorder(self, sample_every)
        self.recorder.event("start", f"run {self.run_id} built: {self.request.spec.title}")
        if self.kind == "reach":
            for p in self.runner.task.plans:
                self.recorder.event("plan", f"target {p.name} {np.round(p.position, 3).tolist()}: "
                                            f"{'planned' if p.reachable else 'REJECTED: ' + p.reason}")
        self.recorder._sample()

    # ------------------------------------------------------------------ measurements for the recorder

    @property
    def phase(self) -> str:
        if self.episode is not None:
            return self.episode.phase
        ref = self.runner.task.reference(self.engine.time)
        return "hold" if not np.any(ref.qd) else "move"

    def reference(self):
        if self.episode is not None:
            r = self.episode.ref.reference(self.engine.time)
        else:
            r = self.runner.task.reference(self.engine.time)
        return r.q, r.qd

    def observation(self):
        if self.episode is not None and self.episode.last_observation is not None:
            return self.episode.last_observation
        return self.engine.arm_qpos(), self.engine.arm_qvel()  # runner: noise is drawn inside step(), not kept

    def object_pose(self):
        m, d = self.engine.model, self.engine.data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        if b < 0:
            return np.full(3, np.nan), np.full(4, np.nan)
        return d.xpos[b].copy(), d.xquat[b].copy()

    def grip_width(self) -> float:
        m, d = self.engine.model, self.engine.data
        adr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_joint1", "finger_joint2")]
        return float(d.qpos[adr].sum())

    def clearance(self) -> float:
        """Latest plant clearance measured by the backend (pick-and-place, every 10 ms), else NaN."""
        if self.episode is None:
            return float("nan")
        log = self.episode.phase_log
        v = log[-1].get("min_clear_robot_mm") if log else None
        return float(v) if v is not None and np.isfinite(v) else float("nan")

    def forbidden_this_step(self) -> bool:
        if self.episode is not None:
            n = self.episode.forbidden_steps
            new = n != getattr(self, "_last_forbidden", 0)
            self._last_forbidden = n
            return new
        return False

    # ------------------------------------------------------------------ stepping

    def _apply_force(self) -> None:
        fp = self.request.force_pulse
        if not fp:
            return
        t = self.engine.time
        on = float(fp["start_s"]) <= t < float(fp["start_s"]) + float(fp["duration_s"])
        f = self._force_vec if on else np.zeros(3)
        if on and not self.current_force.any():
            self.recorder.event("force", f"force pulse ON: {fp['magnitude_n']} N on {fp['body']}")
        if not on and self.current_force.any():
            self.recorder.event("force", "force pulse OFF (cleared)")
        self.engine.data.xfrc_applied[self._force_body, :3] = f
        self.current_force = f.copy()

    def step(self) -> bool:
        """One physics step. False when the run has ended (no step taken)."""
        if self.done:
            return False
        if self.episode is not None:
            alive = self.episode.advance()
            if not alive:
                self.done = True
                return False
        else:
            if self.engine.time >= self.duration - 1e-9:
                self.done = True
                return False
            self._apply_force()
            self.runner.step()
            mujoco.mj_kinematics(self.engine.model, self.engine.data)  # fresh poses (no effect on the dynamics)
            if self.engine.data.ncon and self.engine.robot_contacts():
                self.reach_contact_steps += 1
            for name, t_end in self._hold_ends.items():
                if name not in self._hold_errors and self.engine.time >= t_end - 1e-9:
                    plan = next(p for p in self.runner.task.reachable if p.name == name)
                    self._hold_errors[name] = 1e3 * float(np.linalg.norm(
                        self.engine.forward_kinematics(self.engine.arm_qpos()) - plan.position))
        self.steps += 1
        self.recorder.after_step()
        return True

    def stop(self, reason: str = "stopped by the user") -> None:
        if self.done:
            return
        self.stopped_reason = reason
        if self.episode is not None:
            self.episode.abort(reason)
        self.done = True
        self.recorder.event("stop", reason)

    def finish(self) -> RunResult:
        self.done = True
        self.engine.data.xfrc_applied[:] = 0.0
        self.current_force = np.zeros(3)
        wall = time.perf_counter() - self.wall0
        req = self.request.to_dict()
        if self.episode is not None:
            r = self.episode.finish()
            outcome = "success" if r["success"] else ("stopped" if r["failure_stage"] == "stopped" else "failure")
            metrics = {k: v for k, v in r.items() if k not in ("phases", "planning", "forbidden_events")}
            metrics["phases"], metrics["planning"], metrics["forbidden_events"] = (r["phases"], r["planning"],
                                                                                   r["forbidden_events"])
            res = RunResult(self.run_id, req, r["success"], outcome, r["failure_stage"], r["failure_reason"],
                            metrics, float(self.engine.time), wall)
        else:
            task = self.runner.task
            targets = [{"name": p.name, "position": p.position.tolist(), "planned": p.reachable,
                        "reason": p.reason or None, "status": p.status,
                        "error_at_end_of_hold_mm": self._hold_errors.get(p.name)} for p in task.plans]
            log = self.recorder.samples
            q, qr = np.asarray(log["q"]), np.asarray(log["q_ref"])
            metrics = {"targets": targets, "robot_contact_steps": self.reach_contact_steps,
                       "max_joint_tracking_error_rad": float(np.abs(q - qr).max()) if len(q) else None,
                       "criterion": "workbench reach criterion: every target planned, reached within 1 mm at the "
                                    "end of its hold, and 0 physics steps with robot contacts"}
            stats = getattr(self.controller, "stats", None)
            if stats is not None and stats.updates:
                metrics["mpc"] = stats.summary(self.control_period)
            if self.stopped_reason:
                ok, outcome, stage, why = False, "stopped", "stopped", self.stopped_reason
            else:
                ok = (all(t["planned"] for t in targets) and self.reach_contact_steps == 0 and
                      all((t["error_at_end_of_hold_mm"] or np.inf) <= 1.0 for t in targets))
                bad = [t for t in targets if not t["planned"]]
                stage = None if ok else ("planning" if bad else "execution")
                why = None if ok else ("; ".join(f"{t['name']}: {t['reason']}" for t in bad) or
                                       "a target was not reached within 1 mm or the robot made contact")
                outcome = "success" if ok else "failure"
            res = RunResult(self.run_id, req, ok, outcome, stage, why, metrics, float(self.engine.time), wall)
        self.recorder.event("end", f"run ended: {res.outcome}" + (f" ({res.failure_stage}: {res.failure_reason})"
                                                                   if res.failure_reason else ""))
        self.result = res
        return res

    # ------------------------------------------------------------------ paths for display

    def planned_paths(self) -> list[dict]:
        if self.episode is not None:
            return list(self.episode.paths)
        out = []
        for p in self.runner.task.plans:
            if p.path is not None:
                out.append({"phase": f"to {p.name}", "kind": "rrt", "path": [q.tolist() for q in p.path],
                            "raw_path": [np.asarray(q).tolist() for q in (p.raw_path or p.path)]})
        return out


# ------------------------------------------------------------------ recordings (live or saved)


class Recording:
    """Read-only view of a run's recorded data (arrays by field), its MPC snapshots and its events."""

    def __init__(self, samples: dict, snapshots: list, events: list, meta: dict, paths: list, result: dict | None):
        self.meta, self.events, self.snapshots, self.paths, self.result = meta, events, snapshots, paths, result
        self.data = {}
        for k, v in samples.items():
            if k == "phase":
                self.data[k] = np.asarray(v, dtype=object)
            else:
                self.data[k] = np.asarray(v, dtype=float)

    @classmethod
    def from_live(cls, run: LiveRun) -> "Recording":
        rec = run.recorder
        with rec.lock:
            samples = {k: list(v) for k, v in rec.samples.items()}
            snaps, events = list(rec.snapshots), list(rec.events)
        return cls(samples, snaps, events, run_meta(run), run.planned_paths(),
                   json.loads(run.result.to_json()) if getattr(run, "result", None) else None)

    @property
    def t(self) -> np.ndarray:
        return self.data["t"]

    def index(self, t: float) -> int:
        """Index of the last sample at or before t."""
        ts = self.t
        if not len(ts):
            return 0
        return int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))

    def snapshot_at(self, t: float) -> dict | None:
        """The latest MPC update at or before t (its timestamp is part of the dict)."""
        best = None
        for s in self.snapshots:
            if s["time"] <= t + 1e-9:
                best = s
            else:
                break
        return best

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: (np.asarray(v, dtype=str) if k == "phase" else v) for k, v in self.data.items()}
        np.savez_compressed(directory / "telemetry.npz", **arrays)
        (directory / "mpc_snapshots.json").write_text(json.dumps(self.snapshots))
        (directory / "events.json").write_text(json.dumps(self.events, indent=1))
        (directory / "paths.json").write_text(json.dumps(self.paths))
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=1))
        if self.result is not None:
            (directory / "result.json").write_text(json.dumps(self.result, indent=1))
        return directory

    @classmethod
    def load(cls, directory: Path) -> "Recording":
        directory = Path(directory)
        with np.load(directory / "telemetry.npz", allow_pickle=False) as z:
            samples = {k: (z[k].astype(object) if k == "phase" else z[k]) for k in z.files}
        result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").is_file() else None
        return cls(samples, json.loads((directory / "mpc_snapshots.json").read_text()),
                   json.loads((directory / "events.json").read_text()), json.loads((directory / "meta.json").read_text()),
                   json.loads((directory / "paths.json").read_text()), result)


def run_meta(run: LiveRun) -> dict:
    e = run.engine
    return {"run_id": run.run_id, "request": run.request.to_dict(), "resolved_summary": run.request.summary(),
            "kind": run.kind, "physics_dt": e.dt, "control_period": run.control_period,
            "sample_every_steps": run.recorder.sample_every, "sample_dt": run.recorder.sample_every * e.dt,
            "mpc_horizon": getattr(run.controller, "N", None),
            "controller_class": type(run.controller).__name__,
            "joint_limits": e.joint_limits.tolist(), "velocity_limits": list(e.robot.velocity_limits),
            "torque_limits": np.abs(e.model.actuator_forcerange[e.arm_actuator_ids]).max(axis=1).tolist(),
            "mpc_weights": _weights(run.controller), "config_sha256": hashlib.sha256(
                json.dumps(run.resolved, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "resolved_config": run.resolved, "provenance": provenance(),
            "phases": list(PHASES) if run.kind == "pick_place" else list(REACH_PHASES)}


def _weights(ctrl) -> dict | None:
    w = getattr(ctrl, "w", None)
    if w is None or not hasattr(ctrl, "tau_scale"):
        return None
    return {"position": np.broadcast_to(w.position, (7,)).tolist(), "velocity": np.broadcast_to(w.velocity, (7,)).tolist(),
            "terminal": w.terminal, "effort": np.broadcast_to(w.effort, (7,)).tolist(),
            "torque_rate": np.broadcast_to(w.torque_rate, (7,)).tolist(), "slack_quadratic": w.slack_quadratic,
            "slack_linear": w.slack_linear, "tau_scale": ctrl.tau_scale.tolist(),
            "q_bounds": ctrl.q_bounds.tolist(), "v_bounds": ctrl.v_bounds.tolist(),
            "torque_limits": ctrl.torque_limits.tolist(), "soft": ctrl.soft, "horizon": ctrl.N,
            "velocity_margin": ctrl.velocity_margin}


def save_run(run: LiveRun, root: Path = RUNS_DIR) -> Path:
    rec = Recording.from_live(run)
    d = rec.save(root / run.run_id)
    run.result.directory = str(d)
    (d / "result.json").write_text(run.result.to_json())
    return d


def list_runs(root: Path = RUNS_DIR) -> list[dict]:
    out = []
    for d in sorted(root.glob("*"), reverse=True) if root.is_dir() else []:
        try:
            meta = json.loads((d / "meta.json").read_text())
            res = json.loads((d / "result.json").read_text()) if (d / "result.json").is_file() else {}
        except (OSError, ValueError):
            continue
        out.append({"run_id": d.name, "scenario": meta["request"]["scenario"], "controller": meta["request"]["controller"],
                    "observer": meta["request"]["observer"], "label": meta["request"].get("label", ""),
                    "outcome": res.get("outcome"), "sim_time_s": res.get("sim_time_s")})
    return out


def run_to_end(request: RunRequest, save: bool = True, root: Path = RUNS_DIR) -> tuple[RunResult, Recording]:
    """Python API: build, run headless to the end, finish, optionally save. Returns (result, recording)."""
    run = LiveRun(request)
    while run.step():
        pass
    res = run.finish()
    if save:
        save_run(run, root)
    return res, Recording.from_live(run)
