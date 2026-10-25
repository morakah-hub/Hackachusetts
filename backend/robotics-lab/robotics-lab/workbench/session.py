"""Workbench session: one worker thread that owns the simulation and the renderer (v0.7).

Lifecycle and thread ownership
------------------------------
* `Session` starts exactly one worker thread ("workbench-sim"). Every live MuJoCo object of the
  active run (model, data, controllers, planner) and every `mujoco.Renderer` (OpenGL context) is
  created, used and closed on that thread. Callers (HTTP handler threads, the command console, tests)
  never touch them: they enqueue actions and read copies (JPEG bytes, status, recordings) under a lock.
* One active physics run at a time. Physics advances in wall-clock slices paced by `speed` (1 = real
  time, 0 = as fast as possible); the physics timestep is never changed. Rendering happens at most
  `fps` times per second of wall time, independent of the physics rate.
* Configuration (`configure`) changes the *pending* request only. It is applied by `reset` / `run`,
  never to an active trajectory. When no run is active the pending scene is rebuilt as a preview
  (coalesced: once after a burst of edits), so the viewport shows what will run.
* `close()` stops the run, closes the renderers on the worker thread and joins the thread.
* Replay renders recorded `qpos` into a separate MjData of a separately built model; it never touches the
  live plant.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import traceback
from dataclasses import asdict

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from planning.visualization import add_points, add_polyline
from robots.panda import Panda
from simulation.engine import SimulationEngine
from simulation.manipulation import case_overrides
from simulation.runner import CONTROLLER_ACTUATION, _merge
from workbench.api import RequestError, RunRequest, RunStatus
from workbench.runs import RUNS_DIR, LiveRun, Recording, save_run

DEFAULT_CAMERA = {"lookat": [0.42, 0.0, 0.55], "distance": 1.75, "azimuth": 145.0, "elevation": -24.0}
# (width, height, shadows and reflections). Shadows/reflections cost ~45 ms per offscreen frame on the
# development laptop (vs ~10 ms without), so only "high" uses them. Display quality never changes physics.
QUALITY = {"low": (640, 400, False), "medium": (960, 600, False), "high": (1280, 800, True)}


def scene_engine(request: RunRequest) -> SimulationEngine:
    """A fresh engine with the request's scene geometry (for replay rendering): same environment,
    overrides and robot variant as the run, built from the declarative request."""
    res = request.resolve()
    cfg = res["config"]
    rob = cfg.get("robot") or {}
    robot = Panda(actuation=CONTROLLER_ACTUATION[cfg["controller"]["type"]],
                  gripper_stiffness=rob.get("gripper_stiffness"), gripper_damping=rob.get("gripper_damping"))
    overrides = cfg.get("environment_overrides") or {}
    env = cfg["environment"]
    if res["kind"] == "pick_place":
        overrides = _merge(overrides, case_overrides(cfg["pick_place"], res["case"]))
        env = res["case"]["environment"]
    return SimulationEngine(env, robot, overrides)


class _View:
    """Render helper bound to one model (lives on the worker thread)."""

    def __init__(self, model: mujoco.MjModel, size: tuple[int, int], effects: bool = False):
        w, h = size
        self.effects = effects
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, h)
        self.model = model
        self.renderer = mujoco.Renderer(model, h, w)
        self.size = size
        self.scratch = mujoco.MjData(model)
        self.polys: list[tuple[np.ndarray, tuple, float]] = []
        self._paths_key = None

    def close(self) -> None:
        self.renderer.close()

    def set_paths(self, paths: list[dict], base_qpos: np.ndarray, qpos_adr, site_id: int) -> None:
        key = (len(paths), sum(len(p["path"]) for p in paths))
        if key == self._paths_key:
            return
        self._paths_key = key
        polys = []

        def fk(q):
            s = self.scratch
            s.qpos[:] = base_qpos
            s.qpos[qpos_adr] = q
            mujoco.mj_kinematics(self.model, s)
            return s.site_xpos[site_id].copy()
        for p in paths:
            path = [np.asarray(q, float) for q in p["path"]]
            pts = [fk(path[0])]
            for a, b in zip(path[:-1], path[1:]):
                for s in np.linspace(0, 1, 6)[1:]:
                    pts.append(fk(a + (b - a) * s))
            color = (0.1, 0.85, 0.3, 0.9) if p["kind"] == "rrt" else (0.2, 0.6, 1.0, 0.9)
            polys.append((np.array(pts), color, 0.003))
        self.polys = polys

    def render(self, data: mujoco.MjData, cam: mujoco.MjvCamera, extra=None) -> np.ndarray:
        self.renderer.update_scene(data, cam)
        scn = self.renderer.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.effects
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.effects
        for pts, rgba, width in self.polys:
            add_polyline(scn, pts, rgba, width)
        if extra is not None:
            extra(scn)
        return self.renderer.render()


def _arrow(scene, origin, vector, rgba=(1.0, 0.15, 0.6, 1.0), scale=0.01, width=0.008):
    """A force arrow from origin along vector (N) with length scale * |F| (m), visual only."""
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(vector) < 1e-9:
        return
    g = scene.geoms[scene.ngeom]
    end = np.asarray(origin) + scale * np.asarray(vector)
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(origin, float), end)
    scene.ngeom += 1


class Session:
    def __init__(self, quality: str = "medium", fps: float = 20.0, runs_dir=RUNS_DIR, autostart: bool = True,
                 show_initial_scene: bool = True):
        self.lock = threading.Lock()
        self.request = RunRequest()
        self.status = RunStatus(message="ready: choose a scenario and press Run")
        self.speed = 1.0
        self.fps = fps
        self.quality = quality
        self.view_mode = "live"            # "live" | "replay"
        self.camera = dict(DEFAULT_CAMERA)
        self.runs_dir = runs_dir
        self.messages: list[dict] = []
        self.frame_id, self.frame = 0, b""
        self.live: LiveRun | None = None
        self.replay: Recording | None = None
        self.replay_time, self.replay_playing, self.replay_speed = 0.0, False, 1.0
        self.last_result = None
        self.busy_since: float | None = None
        self._actions: queue.Queue = queue.Queue()
        self._shutdown = threading.Event()
        self._views: dict[int, _View] = {}
        self._replay_engine: SimulationEngine | None = None
        self._dirty = True
        self._step_budget = 0
        self.on_run_finished: list = []   # callbacks(session, RunResult), called on the worker thread
        self._preview_pending = False
        self.experiment: dict | None = None
        self._thread = threading.Thread(target=self._loop, name="workbench-sim", daemon=True)
        if autostart:
            self._thread.start()
            if show_initial_scene:
                self.act("reset")   # build the default scene so the viewport shows the robot from the start

    # ------------------------------------------------------------------ public API (any thread)

    def configure(self, **changes) -> RunRequest:
        """Validate and store the pending request (applied by the next reset/run)."""
        req = self.request.replace(**changes).validate()
        with self.lock:
            self.request = req
            # Preview only when no run is active; edits during a run wait for the next run/reset.
            self._preview_pending = self.status.state not in ("running", "paused", "building")
        self.log("config", "pending settings: " + "; ".join(req.summary()))
        return req

    def set_request(self, req: RunRequest) -> RunRequest:
        req = RunRequest.from_dict(req.to_dict()).validate()
        with self.lock:
            self.request = req
            self._preview_pending = self.status.state not in ("running", "paused", "building")
        return req

    def act(self, action: str, **kw) -> None:
        if action not in ("reset", "run", "pause", "resume", "step", "stop", "replay", "seek", "replay_play",
                          "replay_pause", "live", "camera", "camera_reset", "quality", "speed", "run_request"):
            raise RequestError(f"unknown action '{action}'")
        self._actions.put((action, kw))

    def log(self, kind: str, text: str) -> None:
        with self.lock:
            self.messages.append({"t": time.time(), "kind": kind, "text": text})
            del self.messages[:-200]

    def snapshot_status(self) -> dict:
        with self.lock:
            st = asdict(self.status)
            busy = self.busy_since is not None and time.perf_counter() - self.busy_since > 0.25
            st.update(request=self.request.to_dict(), pending_summary=self.request.summary(), speed=self.speed,
                      view=self.view_mode, replay_time=self.replay_time, replay_playing=self.replay_playing,
                      replay_speed=self.replay_speed, frame_id=self.frame_id, quality=self.quality,
                      busy=busy, camera=self.camera,
                      replay_run=self.replay.meta["run_id"] if self.replay else None,
                      replay_duration=float(self.replay.t[-1]) if self.replay is not None and len(self.replay.t) else 0.0,
                      last_result=self.last_result)
        return st

    def latest_frame(self) -> tuple[int, bytes]:
        with self.lock:
            return self.frame_id, self.frame

    def recording(self) -> Recording | None:
        """The recording shown in the inspectors: the replay if in replay mode, else the live run's."""
        if self.view_mode == "replay" and self.replay is not None:
            return self.replay
        live = self.live
        return Recording.from_live(live) if live is not None else self.replay

    def close(self, timeout: float = 10.0) -> None:
        self._shutdown.set()
        self._actions.put(("noop", {}))
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------ worker thread

    def _set(self, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(self.status, k, v)
        self._dirty = True

    def _loop(self) -> None:
        last_render, render_cost = 0.0, 0.0
        try:
            while not self._shutdown.is_set():
                self._drain()
                if self._shutdown.is_set():
                    break
                ran = False
                # Preview: when no run is active, show the pending scene (rebuilt once after a burst of edits).
                # An active (running/paused) run is never changed; its settings apply at the next run/reset.
                if self._preview_pending and self._actions.empty() and self.status.state in (
                        "idle", "ready", "finished", "failed") and self.view_mode == "live":
                    self._preview_pending = False
                    if self.live is None or self.live.request.to_dict() != self.request.to_dict():
                        self._build()
                if self.live is not None and (self.status.state == "running" or self._step_budget > 0):
                    self._advance()
                    ran = True
                if self.view_mode == "replay" and self.replay_playing and self.replay is not None:
                    self._advance_replay()
                    ran = True
                now = time.perf_counter()
                # Display rate: at most fps, and rendering may take at most ~1/3 of the worker's time while
                # physics runs (at 'max' speed at most 5 frames/s), so the UI never slows the simulation much.
                gap = max(1.0 / self.fps, 2.0 * render_cost if ran else 0.0,
                          0.2 if ran and self.speed <= 0 and self.view_mode == "live" else 0.0)
                if (self._dirty or ran) and now - last_render >= gap:
                    self._render()
                    render_cost = time.perf_counter() - now
                    last_render = now
                    self._dirty = False
                if not ran:
                    time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001 - surface worker crashes to the UI instead of dying silently
            self.log("error", f"worker crashed: {exc!r}\n{traceback.format_exc()}")
            self._set(state="failed", message=f"internal error: {exc!r}")
        finally:
            if self.live is not None and not self.live.done:
                self.live.stop("application closed")
            for v in self._views.values():
                v.close()
            self._views.clear()

    def _drain(self) -> None:
        while True:
            try:
                action, kw = self._actions.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(action, kw)
            except RequestError as exc:
                self.log("error", str(exc))
                self._set(message=f"rejected: {exc}")
            except Exception as exc:  # noqa: BLE001
                self.log("error", f"{action} failed: {exc!r}")
                self._set(state="failed", message=f"{action} failed: {exc}")
            self._dirty = True

    def _handle(self, action: str, kw: dict) -> None:
        st = self.status.state
        if action == "noop":
            return
        if action == "run_request":   # configure + reset + run in one step (experiments, API)
            self.set_request(RunRequest.from_dict(kw["request"]))
            action = "run"
        if action == "reset":
            self._build()
        elif action == "run":
            stale = self.live is not None and self.live.request.to_dict() != self.request.to_dict()                 and self.status.state == "ready"   # settings changed after the last reset: rebuild
            if self.live is None or self.live.done or stale or self.status.state not in ("ready", "paused"):
                self._build()
            if self.live is not None:
                self.view_mode = "live"
                with self.lock:
                    self.last_result = None   # the previous result stays visible until a new run starts
                self._preview_pending = False
                self._pace_reset()
                self._set(state="running", message="running")
                self.log("run", f"run {self.live.run_id} started")
        elif action == "pause" and st == "running":
            self._set(state="paused", message="paused")
        elif action == "resume" and st == "paused":
            self._pace_reset()
            self._set(state="running", message="running")
        elif action == "step":
            if self.live is None or self.live.done:
                self._build()
            if self.status.state in ("ready", "paused"):
                n = int(kw.get("physics_steps", 1))
                if kw.get("unit") == "control":
                    n = max(1, int(round(self.live.control_period / self.live.engine.dt)))
                self._step_budget = n
                self._set(state="paused", message=f"stepping {n} physics step(s) of {1e3 * self.live.engine.dt:.0f} ms")
        elif action == "stop":
            if self.live is not None and not self.live.done:
                self.live.stop("stopped by the user")
                self._finish_run()
        elif action == "replay":
            self._load_replay(kw.get("run_id"))
        elif action == "seek":
            if self.replay is None and self.live is not None:
                self._load_replay(None)
            self.view_mode = "replay"
            self.replay_time = float(kw["t"])
            self.replay_playing = False
        elif action == "replay_play":
            if self.replay is not None:
                self.view_mode, self.replay_playing = "replay", True
                self.replay_speed = float(kw.get("speed", self.replay_speed))
                if self.replay_time >= float(self.replay.t[-1]) - 1e-9:
                    self.replay_time = 0.0
                self._replay_wall = time.perf_counter()
        elif action == "replay_pause":
            self.replay_playing = False
        elif action == "live":
            self.view_mode, self.replay_playing = "live", False
        elif action == "camera":
            for k in ("azimuth", "elevation", "distance"):
                if k in kw:
                    self.camera[k] = float(kw[k])
            if "lookat" in kw:
                self.camera["lookat"] = [float(v) for v in kw["lookat"]]
            self.camera["elevation"] = float(np.clip(self.camera["elevation"], -89, 89))
            self.camera["distance"] = float(np.clip(self.camera["distance"], 0.3, 6.0))
        elif action == "camera_reset":
            self.camera = dict(DEFAULT_CAMERA)
        elif action == "quality":
            if kw["level"] not in QUALITY:
                raise RequestError(f"quality must be one of {list(QUALITY)}")
            self.quality = kw["level"]
            for v in self._views.values():
                v.close()
            self._views.clear()
        elif action == "speed":
            self.speed = float(kw["value"])
            self._pace_reset()

    def _pace_reset(self) -> None:
        self._wall_ref = time.perf_counter()
        self._sim_ref = self.live.engine.time if self.live is not None else 0.0

    def _build(self) -> None:
        if self.live is not None and not self.live.done:
            if self.live.steps == 0:   # built but never started: nothing to record, discard silently
                self.live.done = True
            else:
                self.live.stop("reset by the user")
                self._finish_run(save=True)
        req = self.request
        if self.live is not None:  # the old run's renderer is not needed any more
            v = self._views.pop(id(self.live.engine.model), None)
            if v is not None:
                v.close()
        self._set(state="building", message="building scene, controller and plans ...", phase="", sim_time=0.0)
        self.busy_since = time.perf_counter()
        try:
            self.live = LiveRun(req)
        except Exception as exc:  # noqa: BLE001 - e.g. planning failed, infeasible trajectory
            self.live = None
            self.busy_since = None
            self._set(state="failed", message=f"could not build the run: {exc}")
            self.log("error", f"could not build the run: {exc}")
            return
        self.busy_since = None
        self.view_mode = "live"
        self._set(state="ready", run_id=self.live.run_id, message="ready (press Run, or Step)",
                  phase=self.live.phase, steps=0, sim_time=float(self.live.engine.time),
                  active_controller=self._active_name())
        self.log("build", f"built run {self.live.run_id}: " + "; ".join(req.summary()))

    def _active_name(self) -> str:
        c = self.live.controller if self.live else None
        if c is None:
            return ""
        name = type(c).__name__.replace("Controller", "")
        return f"{name} (CTC fallback)" if getattr(c, "using_fallback", False) else name

    def _advance(self) -> None:
        live = self.live
        wall0 = time.perf_counter()
        if self._step_budget > 0:
            n = self._step_budget
            self._step_budget = 0
            for _ in range(n):
                self.busy_since = time.perf_counter()
                if not live.step():
                    break
            self.busy_since = None
        else:
            target = None if self.speed <= 0 else self._sim_ref + (wall0 - self._wall_ref) * self.speed
            while time.perf_counter() - wall0 < 0.04:
                if target is not None and live.engine.time >= target:
                    break
                self.busy_since = time.perf_counter()
                alive = live.step()
                self.busy_since = None
                if not alive:
                    break
        self._set(sim_time=float(live.engine.time), phase=live.phase, steps=live.steps,
                  active_controller=self._active_name())
        if live.done:
            self._finish_run()

    def _finish_run(self, save: bool = True) -> None:
        live = self.live
        res = live.finish()
        directory = save_run(live, self.runs_dir) if save else None
        with self.lock:
            self.last_result = {"run_id": res.run_id, "outcome": res.outcome, "success": res.success,
                                "failure_stage": res.failure_stage, "failure_reason": res.failure_reason,
                                "sim_time_s": res.sim_time_s, "wall_time_s": res.wall_time_s,
                                "directory": str(directory) if directory else None,
                                "metrics": {k: v for k, v in res.metrics.items()
                                            if not isinstance(v, (list, dict)) or k in ("targets", "mpc")}}
        msg = (f"finished: {res.outcome.upper()}" + (f" at {res.failure_stage}: {res.failure_reason}"
                                                     if res.failure_reason else "") + f" (saved as {res.run_id})")
        self._set(state="finished" if res.outcome != "failure" else "failed", message=msg)
        self.log("result", msg)
        for callback in list(self.on_run_finished):  # e.g. a guided experiment queues its next run
            callback(self, res)

    def _load_replay(self, run_id: str | None) -> None:
        if run_id in (None, "", "live", "last"):
            if self.live is None:
                raise RequestError("no run to replay yet")
            rec = Recording.from_live(self.live)
        else:
            d = self.runs_dir / run_id
            if not d.is_dir():
                raise RequestError(f"no saved run '{run_id}'")
            rec = Recording.load(d)
        req = RunRequest.from_dict(rec.meta["request"])
        if self._replay_engine is not None:
            v = self._views.pop(id(self._replay_engine.model), None)
            if v:
                v.close()
        self._replay_engine = scene_engine(req)
        self.replay = rec
        self.replay_time, self.replay_playing, self.view_mode = 0.0, False, "replay"
        self.log("replay", f"replaying recorded run {rec.meta['run_id']} ({len(rec.t)} samples, recorded data)")

    def _advance_replay(self) -> None:
        now = time.perf_counter()
        self.replay_time += (now - getattr(self, "_replay_wall", now)) * self.replay_speed
        self._replay_wall = now
        end = float(self.replay.t[-1])
        if self.replay_time >= end:
            self.replay_time, self.replay_playing = end, False

    # ------------------------------------------------------------------ rendering (worker thread only)

    def _view(self, model) -> _View:
        w, h, effects = QUALITY[self.quality]
        v = self._views.get(id(model))
        if v is None or v.size != (w, h) or v.effects != effects:
            if v is not None:
                v.close()
            v = self._views[id(model)] = _View(model, (w, h), effects)
        return v

    def _cam(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.camera["lookat"]
        cam.distance, cam.azimuth, cam.elevation = self.camera["distance"], self.camera["azimuth"], self.camera["elevation"]
        return cam

    def _render(self) -> None:
        label, rgb = None, None
        if self.view_mode == "replay" and self.replay is not None and self._replay_engine is not None:
            e = self._replay_engine
            rec = self.replay
            i = rec.index(self.replay_time)
            e.data.qpos[:] = rec.data["qpos"][i]
            mujoco.mj_forward(e.model, e.data)
            view = self._view(e.model)
            view.set_paths(rec.paths, rec.data["qpos"][0], e.arm_qpos_adr, e.ee_site_id)
            force = rec.data["force"][i]
            body = (rec.meta["request"].get("force_pulse") or {}).get("body")

            def extra(scn):
                self._targets(scn, RunRequest.from_dict(rec.meta["request"]))
                if body and np.any(force):
                    _arrow(scn, e.data.xpos[e.model.body(body).id], force)
            rgb = view.render(e.data, self._cam(), extra)
            label = (f"RECORDED  run {rec.meta['run_id']}  t = {rec.t[i]:.3f} s  phase {rec.data['phase'][i]}"
                     f"{'  FORCE ' + format(np.linalg.norm(force), '.0f') + ' N' if np.any(force) else ''}")
        elif self.live is not None:
            live = self.live
            e = live.engine
            view = self._view(e.model)
            view.set_paths(live.planned_paths(), e.data.qpos.copy(), e.arm_qpos_adr, e.ee_site_id)

            def extra(scn):
                self._targets(scn, live.request)
                if live.current_force.any():
                    _arrow(scn, e.data.xpos[live._force_body], live.current_force)
            rgb = view.render(e.data, self._cam(), extra)
            label = (f"LIVE  t = {e.time:.3f} s  {self.status.state}  phase {live.phase}  {self._active_name()}"
                     f"{'  FORCE ' + format(np.linalg.norm(live.current_force), '.0f') + ' N' if live.current_force.any() else ''}")
        if rgb is None:
            w, h, _ = QUALITY[self.quality]
            img = Image.new("RGB", (w, h), (22, 27, 34))
        else:
            img = Image.fromarray(rgb)
        if label:
            d = ImageDraw.Draw(img, "RGBA")
            replay = label.startswith("RECORDED")
            d.rectangle((0, 0, img.width, 26), fill=(160, 90, 0, 190) if replay else (0, 0, 0, 150))
            d.text((10, 6), label, fill=(255, 255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=82)
        with self.lock:
            self.frame_id += 1
            self.frame = buf.getvalue()

    @staticmethod
    def _targets(scn, req: RunRequest) -> None:
        if req.spec.kind == "reach":
            add_points(scn, np.array(req.resolved_targets()), (1.0, 0.2, 0.2, 0.7), 0.015)
