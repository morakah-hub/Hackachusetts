"""Simulation worker: the single owner of MjModel/MjData, the controller and the renderer contexts.

HTTP handlers never touch MuJoCo. They edit the scene store (while idle) or submit commands through a bounded
queue, and read the latest published snapshot / camera frame (both replaced, never queued). One run at a
time; a run is a task plan made of pick-and-place legs executed one after another.

Physics keeps stepping in every state except paused (the arm holds its last reference); pause freezes physics
time. The speed setting scales simulated time per wall-clock second (0 = as fast as possible).
"""

from __future__ import annotations

import io
import platform
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import Future

import mujoco
import numpy as np
from PIL import Image

from . import __version__
from . import scene as sc
from . import geometry
from .camera import RGBDCamera
from .controller import ComputedTorqueController
from .evaluator import Evaluator
from .kinematics import Kinematics
from .planner import TaskPlan, scene_config
from .robot import GRIPPER_OPEN, HOME_Q
from .rules import PLACE_TOL_MM
from .store import StudioStore
from .task import PHASE_LABELS, PickPlaceLeg, TaskFailure
from .trajectory import Hold

ACTIVE = ("starting", "running", "paused", "stopping")
FAILURE_TEXT = {
    "PERCEPTION_UNAVAILABLE": ("The camera could not find {obj}.",
                               "Keep the object inside the shaded work area and away from others, or set "
                               "'Locate objects with' to Simulator in the Camera panel."),
    "PLAN_FAILED": ("The arm could not plan a safe motion for {obj}.",
                    "Leave more room around the object or choose another spot."),
    "GRASP_FAILED": ("The gripper did not hold {obj}.", "The object may have moved; try the task again."),
    "PLACE_FAILED": ("{obj} did not end up where planned.", "Check the result and try again."),
    "CONTACT_VIOLATION": ("The robot touched something it should not have while moving {obj}.",
                          "Leave more space between objects."),
    "CONTROL_TIMEOUT": ("The arm did not finish a motion in time while moving {obj}.", ""),
    "BACKEND_ERROR": ("The simulation hit an internal error while moving {obj}.", "See the event log for details."),
}


class CommandError(Exception):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _depth_colormap(depth: np.ndarray, near=0.6, far=1.6) -> np.ndarray:
    t = np.clip((depth - near) / (far - near), 0, 1)
    stops = np.array([[250, 236, 200], [240, 170, 70], [40, 160, 160], [20, 50, 80], [10, 14, 22]], float)
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    rgb[depth <= 0] = 0
    return rgb.astype(np.uint8)


class SimWorker:
    def __init__(self, store: StudioStore, realtime: bool = True, render: bool = True, speed: float = 1.0):
        self.store = store
        self.realtime = realtime
        self.render_enabled = render
        self.commands: queue.Queue = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot: dict = {"status": "loading"}
        self._frame: dict | None = None
        self.events: deque = deque(maxlen=80)
        self.ready = threading.Event()
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        self.speed = float(speed)
        self.mode = "camera_rgbd"
        self.faults = {"camera_masked": False, "gripper_disabled": False}
        self.run: dict | None = None
        self.results: dict[str, dict] = {}
        self.layout_before_run: dict | None = None
        self.thinking = False
        self.on_step = None             # optional hook (recording), called after every physics step
        self.camera = None
        self.model = None

    # ------------------------------------------------------------------ setup (worker thread)

    def _setup(self) -> None:
        t0 = time.perf_counter()
        snap = self.store.snapshot()
        self.cfg = scene_config(snap["objects"])
        self._build(self.cfg, HOME_Q)
        self.synced_rev = snap["revision"]
        self.scene_revision = snap["revision"]
        names = [self.model.body(int(b)).name for b in sorted(self.ids.robot_bodies)]
        self.pose_bodies = ["table", "camera_rig", *names]
        self.geometry = geometry.export(self.model, self.pose_bodies, {"camera_rig": 3})
        self.build_s = time.perf_counter() - t0
        self.step_ms = deque(maxlen=200)
        self.rtf_window: deque = deque(maxlen=60)
        self.seq = 0
        self.log(f"scene compiled in {self.build_s:.2f} s (MuJoCo {mujoco.__version__}, "
                 f"{len(self.cfg.objects)} objects)")

    def _build(self, cfg: sc.SceneConfig, arm_q) -> None:
        self.model, self.ids = sc.build_model(cfg)
        self.data = mujoco.MjData(self.model)
        sc.set_initial_state(self.model, self.data, self.ids, cfg, np.asarray(arm_q, float))
        self.kin = Kinematics(self.model, self.ids)
        self.ctrl = ComputedTorqueController(self.model, self.ids)
        self.evaluator = Evaluator(self.model, self.ids)
        if self.render_enabled:
            if self.camera is not None:
                self.camera.close()
            self.camera = RGBDCamera(self.model, cfg.camera, self.ids.camera)
        self.cfg = cfg
        self.task: PickPlaceLeg | None = None
        self.task_iter = None
        self._set_hold(arm_q)
        self.gripper_cmd = GRIPPER_OPEN
        self.estimate = self.plan_info = None

    def _sync_scene(self, force: bool = False) -> None:
        """Bring the simulation up to the store's scene (only while no run is active)."""
        rev = self.store.revision
        if rev == self.synced_rev and not force:
            return
        snap = self.store.snapshot()
        cfg = scene_config(snap["objects"])
        if cfg.structure_key() != self.cfg.structure_key():
            self._build(cfg, self.arm_q())
            self.log(f"scene rebuilt with {len(cfg.objects)} objects")
        else:
            sc.place_objects(self.model, self.data, self.ids, cfg.objects)
            self.cfg = cfg
            mujoco.mj_forward(self.model, self.data)
        self.synced_rev = snap["revision"]
        self.scene_revision = snap["revision"]
        self.estimate = None

    # ------------------------------------------------------------------ helpers used by the leg

    def log(self, msg: str, level: str = "info") -> None:
        t = float(self.data.time) if getattr(self, "data", None) is not None else 0.0
        self.events.append({"t_sim": round(t, 3), "t_wall": time.time(), "level": level, "msg": msg})

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.ids.arm_qadr].copy()

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.ids.arm_dadr].copy()

    def finger_width(self) -> float:
        return float(self.data.qpos[self.ids.finger_qadr].sum())

    def set_gripper(self, value: float) -> None:
        if self.faults["gripper_disabled"] and value < GRIPPER_OPEN:
            self.log("fault injection: gripper close command ignored (gripper disabled)", "warn")
            return
        self.gripper_cmd = float(value)

    def _set_hold(self, q) -> None:
        self.segment = Hold(q)
        self.seg_t0 = float(self.data.time)

    def set_reference(self, seg) -> None:
        self.segment = seg
        self.seg_t0 = float(self.data.time)

    def reference_finished(self) -> bool:
        return self.data.time - self.seg_t0 >= self.segment.duration

    def reference_end(self) -> np.ndarray:
        return np.asarray(self.segment.end, float).copy()

    def observe(self):
        if self.camera is None:
            raise TaskFailure("observe", "PERCEPTION_UNAVAILABLE", "rendering disabled: no camera available")
        self.camera.masked = self.faults["camera_masked"]
        obs = self.camera.capture(self.data, self.scene_revision)
        self._publish_frame(obs)
        return obs

    def truth_pose(self, oid: str):
        """Simulator truth: known-world development mode, display and evaluation only."""
        b = self.ids.objects[oid].body
        p = self.data.xpos[b].copy()
        R = self.data.xmat[b].reshape(3, 3)
        return p, float(np.arctan2(R[1, 0], R[0, 0]))

    def publish_estimate(self, est: dict) -> None:
        self.estimate = est

    def publish_plan(self, plan: dict) -> None:
        self.plan_info = plan

    def object_poses(self) -> dict[str, tuple[list[float], list[float]]]:
        d = self.data
        return {oid: (d.xpos[o.body].tolist(), d.xquat[o.body].tolist()) for oid, o in self.ids.objects.items()}

    def robot_at_home(self) -> bool:
        return bool(np.abs(self.arm_q() - HOME_Q).max() < 0.03)

    # ------------------------------------------------------------------ physics

    def _physics_step(self) -> None:
        t0 = time.perf_counter()
        d = self.data
        if self.task_iter is not None:
            try:
                next(self.task_iter)
            except StopIteration:
                self._finish_leg(None)
            except TaskFailure as f:
                self._finish_leg(f)
            except Exception as exc:  # keep the worker alive and report the bug as a failed leg
                self.log(traceback.format_exc(limit=4), "error")
                self._finish_leg(TaskFailure(self.task.phase if self.task else "run", "BACKEND_ERROR",
                                             f"{type(exc).__name__}: {exc}"))
        q_ref, qd_ref, qdd_ref = self.segment.sample(d.time - self.seg_t0)
        tau = self.ctrl.compute(d, q_ref, qd_ref, qdd_ref, self.model.opt.timestep)
        d.ctrl[self.ids.arm_act] = tau
        d.ctrl[self.ids.grip_act] = self.gripper_cmd
        mujoco.mj_step(self.model, d)
        if self.task is not None and self.task_iter is not None:
            self.evaluator.step(d, self.task.phase)
            if self.run is not None and len(self.trace) < 40000 and int(round(d.time / self.model.opt.timestep)) % 25 == 0:
                self.trace.append([round(float(d.time), 3), self.run["leg_index"], self.task.phase,
                                   *np.round(self.arm_q(), 5).tolist(), round(self.finger_width(), 5)])
        self.step_ms.append(1e3 * (time.perf_counter() - t0))
        if self.on_step is not None:
            self.on_step(self)

    # ------------------------------------------------------------------ runs

    def _start_run(self, plan: TaskPlan) -> dict:
        if self.run is not None and self.run["state"] in ACTIVE:
            raise CommandError("BUSY", "A task is already running.")
        if plan.revision != self.store.revision:
            raise CommandError("STALE_SCENE", "The table changed after the task was checked. Check it again.")
        self._sync_scene()
        if not plan.ok:
            raise CommandError("NOT_READY", "This task cannot run as set up.", 422)
        self.layout_before_run = self.store.snapshot()
        legs = []
        for leg in plan.legs:
            d = leg.to_dict()
            d.update(status="pending", result=None)
            legs.append(d)
        self.plan_legs = list(plan.legs)
        self.run = {"id": uuid.uuid4().hex[:8], "task_id": plan.task_id, "title": plan.title, "summary": plan.summary,
                    "state": "running", "started_at": time.time(), "finished_at": None, "leg_index": 0,
                    "phase": "observe", "sim_time": 0.0, "message": "", "failure": None, "notices": [], "legs": legs,
                    "perception_mode": self.mode}
        self.run_sim0 = float(self.data.time)
        self.run_wall0 = time.perf_counter()
        self.trace = []
        self.paused = False
        self.log(f"run {self.run['id']} started: {plan.summary} ({len(legs)} moves, "
                 f"{'camera RGB-D' if self.mode == 'camera_rgbd' else 'KNOWN-WORLD dev'} mode)")
        self._start_leg(0)
        return self.run

    def _start_leg(self, i: int) -> None:
        leg = self.plan_legs[i]
        self.run["leg_index"] = i
        self.run["legs"][i]["status"] = "active"
        self.task = PickPlaceLeg(self, leg, self.mode)
        self.task_iter = self.task.program()
        self.leg_wall0, self.leg_sim0 = time.perf_counter(), float(self.data.time)
        self.evaluator.reset(self.data, leg.object_id)
        self.ctrl.saturated_steps = 0

    def _finish_leg(self, failure: TaskFailure | None, stopped: bool = False) -> None:
        task, run = self.task, self.run
        if task is None or run is None:
            return
        self.task_iter = None
        self.thinking = False
        i = run["leg_index"]
        leg = self.plan_legs[i]
        self._set_hold(self.arm_q() if (failure or stopped) else self.reference_end())
        if task.phase_log and task.phase_log[-1]["t_end"] is None:
            task.phase_log[-1]["t_end"] = float(self.data.time)
        ev = self.evaluator.final(self.data, leg.target)
        if stopped:
            ok, stage, code, reason = False, task.phase, "STOPPED", "stopped by the user"
        elif failure is not None:
            ok, stage, code, reason = False, failure.stage, failure.code, failure.reason
        elif ev["forbidden_contacts"]:
            ok, stage, code = False, "evaluation", "CONTACT_VIOLATION"
            reason = f"{ev['forbidden_contacts']} steps with forbidden contacts (first: {ev['first_forbidden_contact']})"
        elif not ev["placed"]:
            ok, stage, code = False, "evaluation", "PLACE_FAILED"
            reason = "; ".join(ev["reasons"])
        else:
            ok, stage, code, reason = True, None, None, "placed"
        est_err = None
        if task.estimate and task.estimate.get("valid") and task.mode == "camera_rgbd":
            est_err = 1e3 * float(np.linalg.norm(np.asarray(task.estimate["position_m"][:2]) - np.asarray(leg.start_pos[:2])))
        msg, hint = FAILURE_TEXT.get(code or "", ("", ""))
        result = {"success": ok, "stage": stage, "reason": None if ok else reason, "code": code,
                  "message": msg.format(obj=leg.object_name) if msg else ("Done" if ok else reason),
                  "hint": hint, "placement_error_mm": round(1e3 * ev["placement_error_xy_m"], 2),
                  "height_error_mm": round(1e3 * ev["placement_error_z_m"], 2),
                  "tilt_deg": round(ev["final_tilt_deg"], 2), "lift_mm": round(1e3 * ev["max_lift_m"], 1),
                  "slip_mm": None if ev["object_slip_m"] is None else round(1e3 * ev["object_slip_m"], 2),
                  "sim_time_s": round(float(self.data.time) - self.leg_sim0, 2),
                  "wall_time_s": round(time.perf_counter() - self.leg_wall0, 2),
                  "controller": self.ctrl.name, "phases": task.phase_log,
                  "transport": task.plan_info.get("transport"), "perception": task.mode,
                  "estimate_error_mm": None if est_err is None else round(est_err, 2),
                  "grasp_width_mm": None if task.grasp_width is None else round(1e3 * task.grasp_width, 1),
                  "torque_saturation_steps": self.ctrl.saturated_steps, "nudged_mm":
                      {k: round(1e3 * v, 1) for k, v in ev["nudged_m"].items()}}
        legd = run["legs"][i]
        legd["result"] = result
        legd["status"] = "done" if ok else ("stopped" if stopped else "failed")
        for name, dist in ev["nudged_m"].items():
            run["notices"].append(f"{name} was nudged by {1e3 * dist:.0f} mm while moving {leg.object_name}.")
        # The measured poses become the scene (objects can now rest on others).
        self.store.set_poses(self.object_poses())
        self.synced_rev = self.scene_revision = self.store.revision
        self.log(f"move {i + 1} ({leg.object_name}): " + (f"placed within {result['placement_error_mm']:.1f} mm"
                                                          if ok else f"{code} at {stage}: {reason}"),
                 "info" if ok else "warn")
        if ok and i + 1 < len(self.plan_legs):
            self._start_leg(i + 1)
            return
        for rest in run["legs"][i + 1:]:
            rest["status"] = "skipped"
        if ok:
            run["state"], run["message"] = "succeeded", "Task completed."
        elif stopped:
            run["state"], run["message"] = "stopped", "Stopped."
        else:
            run["state"] = "failed"
            run["failure"] = {"leg": i, "stage": stage, "reason": reason, "code": code,
                              "message": result["message"], "hint": hint}
            run["message"] = result["message"]
        run["finished_at"] = time.time()
        run["sim_time"] = float(self.data.time) - self.run_sim0
        run["wall_time"] = time.perf_counter() - self.run_wall0
        self.paused = False
        self.results[run["id"]] = {"run": run, "trace": self.trace,
                                   "trace_columns": ["t", "leg", "phase", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                                                     "gripper_width_m"],
                                   "versions": {"armstrong": __version__, "mujoco": mujoco.__version__,
                                                "python": platform.python_version()}}
        while len(self.results) > 20:
            self.results.pop(next(iter(self.results)))
        self.log(f"run {run['id']} {run['state']}", "info" if ok else "warn")

    # ------------------------------------------------------------------ commands

    def submit(self, name: str, **kwargs) -> Future:
        fut: Future = Future()
        try:
            self.commands.put_nowait((name, kwargs, fut))
        except queue.Full:
            fut.set_exception(CommandError("BUSY", "command queue full"))
        return fut

    def busy(self) -> bool:
        return self.run is not None and self.run["state"] in ACTIVE

    def _handle(self, name: str, kw: dict):
        run = self.run
        if name == "run":
            return self._start_run(kw["plan"])
        if name == "pause":
            if run is None or run["state"] != "running":
                raise CommandError("BUSY", "Nothing is running.")
            self.paused, run["state"] = True, "paused"
            self.log("paused (physics time frozen)")
            return {}
        if name == "resume":
            if run is None or run["state"] != "paused":
                raise CommandError("BUSY", "The task is not paused.")
            self.paused, run["state"] = False, "running"
            self.log("resumed")
            return {}
        if name == "stop":
            if self.busy():
                self.paused = False
                self._finish_leg(None, stopped=True)
            return {}
        if name == "dismiss":
            if self.busy():
                raise CommandError("BUSY", "Stop the task first.")
            self.run = None
            return {}
        if name == "speed":
            self.speed = float(kw["value"])
            self._reanchor = True
            return {}
        if name == "reset_robot":
            if self.busy():
                raise CommandError("BUSY", "A task is running.")
            self._sync_scene()
            arm = self.ids.arm_qadr
            self.data.qpos[arm] = HOME_Q
            self.data.qvel[self.ids.arm_dadr] = 0.0
            self.data.qpos[self.ids.finger_qadr] = 0.04
            self.gripper_cmd = GRIPPER_OPEN
            self.ctrl.reset()
            self._set_hold(HOME_Q)
            mujoco.mj_forward(self.model, self.data)
            return {}
        if name == "sync":
            if self.busy():
                raise CommandError("BUSY", "A task is running.")
            self._sync_scene()
            return {}
        if name == "faults":
            for k in ("camera_masked", "gripper_disabled"):
                if kw.get(k) is not None:
                    self.faults[k] = bool(kw[k])
            self.log(f"faults: {self.faults}", "warn" if any(self.faults.values()) else "info")
            return dict(self.faults)
        if name == "mode":
            if self.busy():
                raise CommandError("BUSY", "Change the mode when no task is running.")
            if kw["mode"] not in ("camera_rgbd", "known_world"):
                raise CommandError("BAD_MODE", "Unknown mode.", 400)
            self.mode = kw["mode"]
            self.log(f"objects are located with: {'the RGB-D camera' if self.mode == 'camera_rgbd' else 'simulator poses (dev)'}")
            return {"perception_mode": self.mode}
        raise CommandError("BACKEND_ERROR", f"unknown command {name}", 400)

    def _drain(self) -> None:
        while True:
            try:
                name, kw, fut = self.commands.get_nowait()
            except queue.Empty:
                return
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(self._handle(name, kw))
                except CommandError as e:
                    fut.set_exception(e)
                except Exception as e:
                    self.log(f"command {name} failed: {e}", "error")
                    fut.set_exception(CommandError("BACKEND_ERROR", f"{type(e).__name__}: {e}", 500))

    # ------------------------------------------------------------------ publishing

    def _publish_frame(self, obs) -> None:
        buf = io.BytesIO()
        Image.fromarray(obs.rgb).save(buf, format="JPEG", quality=85)
        dbuf = io.BytesIO()
        Image.fromarray(_depth_colormap(obs.depth_m)).save(dbuf, format="PNG")
        with self._lock:
            self._frame = {"meta": obs.describe(), "rgb": buf.getvalue(), "depth": dbuf.getvalue(),
                           "captured_monotonic_s": obs.captured_monotonic_s}

    def _publish_snapshot(self) -> None:
        d, m = self.data, self.model
        self.seq += 1
        bodies = {}
        for name in self.pose_bodies:
            b = m.body(name).id
            bodies[name] = [round(float(v), 6) for v in (*d.xpos[b], *d.xquat[b])]
        objects = {oid: [round(float(v), 6) for v in (*d.xpos[o.body], *d.xquat[o.body])]
                   for oid, o in self.ids.objects.items()}
        run = self.run
        if run is not None and run["state"] in ACTIVE:
            run["sim_time"] = float(d.time) - self.run_sim0
            run["phase"] = self.task.phase if self.task is not None else run["phase"]
        frame_meta = None
        with self._lock:
            if self._frame is not None:
                fm = self._frame["meta"]
                frame_meta = {"frame_id": fm["frame_id"], "scene_revision": fm["scene_revision"],
                              "sim_time_s": fm["sim_time_s"], "masked": fm["masked"], "width": fm["width"],
                              "height": fm["height"],
                              "age_s": time.monotonic() - self._frame["captured_monotonic_s"]}
        snap = {
            "time": time.time(), "revision": self.store.revision, "run": run, "speed": self.speed,
            "thinking": self.thinking, "robot_at_home": self.robot_at_home(), "error": self.error,
            "frame": {"seq": self.seq, "bodies": bodies, "objects": objects, "gripper_width": self.finger_width(),
                      "live": self.busy()},
            "sim_time": float(d.time), "realtime_factor": self._rtf(),
            "perception_mode": self.mode, "faults": dict(self.faults), "camera": frame_meta,
            "estimate": self.estimate, "plan": self.plan_info,
            "controller": {"name": self.ctrl.name, "gains": self.ctrl.gains(), "rate_hz": 1.0 / m.opt.timestep},
            "events": list(self.events)[-30:],
            "timing": {"step_ms_mean": float(np.mean(self.step_ms)) if self.step_ms else None,
                       "build_s": self.build_s},
        }
        with self._lock:
            self._snapshot = snap

    def _rtf(self):
        w = self.rtf_window
        if len(w) < 2 or w[-1][0] - w[0][0] < 0.2:
            return None
        return (w[-1][1] - w[0][1]) / (w[-1][0] - w[0][0])

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot

    def frame(self) -> dict | None:
        with self._lock:
            return self._frame

    # ------------------------------------------------------------------ main loop

    def run_forever(self) -> None:
        try:
            self._setup()
            if self.camera is not None:
                self.observe()
            self._publish_snapshot()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._snapshot = {"status": "error", "error": self.error}
            self.ready.set()
            raise
        self.ready.set()
        dt = self.model.opt.timestep
        anchor_wall, anchor_sim = time.perf_counter(), float(self.data.time)
        self._reanchor = False
        self.paused = False
        last_cam = last_snap = 0.0
        cam_period = 1.0 / self.cfg.camera.rate_hz
        try:
            while not self._stop.is_set():
                self._drain()
                if not self.busy():
                    try:
                        self._sync_scene()
                    except Exception as exc:  # a bad scene must not kill the worker
                        self.log(f"scene sync failed: {exc}", "error")
                now = time.perf_counter()
                if self.paused or self._reanchor:
                    anchor_wall, anchor_sim = now, float(self.data.time)
                    self._reanchor = False
                if not self.paused:
                    if self.speed <= 0:
                        while time.perf_counter() - now < 0.025 and not self.paused:
                            self._physics_step()
                        anchor_wall, anchor_sim = time.perf_counter(), float(self.data.time)
                    else:
                        target = anchor_sim + (now - anchor_wall) * self.speed
                        if target - self.data.time > 0.1:      # fell behind (planning): re-anchor
                            anchor_wall, anchor_sim = now, float(self.data.time)
                            target = anchor_sim + dt
                        n = 0
                        while self.data.time + 0.5 * dt < target and n < 120 and not self.paused:
                            self._physics_step()
                            n += 1
                self.rtf_window.append((time.perf_counter(), float(self.data.time)))
                if self.camera is not None and now - last_cam >= cam_period:
                    last_cam = now
                    try:
                        self.observe()
                    except Exception as exc:
                        self.log(f"camera failed: {exc}", "error")
                if now - last_snap >= 1 / 30:
                    last_snap = now
                    self._publish_snapshot()
                time.sleep(0.001)
        finally:
            if self.camera is not None:
                self.camera.close()

    def start_thread(self) -> None:
        self.thread = threading.Thread(target=self.run_forever, name="armstrong-sim", daemon=True)
        self.thread.start()
        self.ready.wait(60)

    def shutdown(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(5)

    # ------------------------------------------------------------------ synchronous driver (tests/CLI)

    def setup_sync(self) -> None:
        self._setup()
        self._reanchor = False
        self.paused = False

    def run_sync(self, plan: TaskPlan, max_sim_s: float = 300.0) -> dict:
        """Start a run and step as fast as possible until it ends (same code path as the live loop)."""
        self._start_run(plan)
        t_end = self.data.time + max_sim_s
        while self.busy() and self.data.time < t_end:
            self._physics_step()
        if self.busy():
            self._finish_leg(TaskFailure(self.task.phase, "CONTROL_TIMEOUT", f"run exceeded {max_sim_s:.0f} s"))
        return self.run

    def command_sync(self, name: str, **kw):
        return self._handle(name, kw)


__all__ = ["SimWorker", "CommandError", "PHASE_LABELS", "PLACE_TOL_MM"]
