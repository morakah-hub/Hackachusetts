"""The simulation worker: runs task plans leg by leg, paced to the wall clock, and publishes poses.

Thread contract (as in workbench/session.py): one worker thread creates, steps and drops every live MuJoCo
object. HTTP threads only call start/pause/resume/stop/set_speed/reset_robot (which set fields under the lock
or queue a command) and read copies through status()/frame().

Each leg is one PickPlaceEpisode built from the current scene (all objects as free boxes at their measured
poses), starting from the arm configuration the previous leg ended in. After a leg the measured object poses
are written back to the scene. A failed or stopped leg first lets go and waits for the objects to settle, so
the scene is left physically consistent.
"""

import copy
import queue
import threading
import time
import traceback
import uuid

import mujoco
import numpy as np

from robots.panda import Panda
from simulation.engine import SimulationEngine
from simulation.manipulation import PickPlaceEpisode
from studio.geometry import body_poses, robot_body_names
from studio.planner import Leg, TaskPlan, scene_overrides
from studio.scene import SceneObject, StudioStore, World
from studio.settings import StudioSettings

PHASE_LABELS = {"init": "Preparing", "approach": "Moving to the object", "descend": "Lowering the gripper",
                "close": "Gripping", "lift": "Lifting", "hold": "Checking the grip", "transport": "Carrying",
                "lower": "Setting it down", "release": "Letting go", "retreat": "Backing off",
                "verify": "Checking the placement", "home": "Returning to rest", "settle": "Letting objects settle"}
PHASE_ORDER = ["approach", "descend", "close", "lift", "hold", "transport", "lower", "release", "retreat", "verify"]


def explain_failure(stage: str | None, reason: str | None, name: str) -> tuple[str, str]:
    """(message, hint) for a failed leg, for the person using the app."""
    reason = reason or ""
    if stage == "stopped":
        return "Stopped by you.", ""
    if stage == "contacts":
        return (f"The arm or {name} touched something it should not have.",
                "Leave more room between objects, or move the target spot.")
    if "planning failed" in reason or "start configuration invalid" in reason:
        where = "to the object" if stage == "approach" else ("there" if stage == "transport" else "home")
        return f"The arm could not find a collision-free way {where}.", "Move nearby objects apart and try again."
    if "Cartesian line" in reason:
        return "The arm could not move straight up or down at this spot.", "Try a spot closer to the middle of the ring."
    if stage == "close":
        return f"The gripper did not close on {name} as expected.", ""
    if stage == "release":
        return f"The gripper could not open wide enough to let go of {name} cleanly.", reason
    if stage == "lift":
        return f"{name} did not come up with the gripper.", "It may be heavier than the grip can hold."
    if stage in ("hold", "transport", "lower") and ("slipped" in reason or "dropped" in reason or "fell" in reason):
        return f"{name} slipped in the gripper.", "Heavy or tall objects are more likely to slip."
    if stage == "verify":
        if "tilted" in reason:
            return f"{name} tipped over when it was put down.", ""
        return f"{name} did not end up where it should.", reason
    if stage == "home":
        return "The object was placed, but the arm could not return to its rest position.", reason
    return f"Stopped during '{PHASE_LABELS.get(stage or '', stage)}'.", reason


def _transport_kind(planning: list[dict]) -> str | None:
    """How the object was carried: along the arc, or along a planned joint-space path (fallback)."""
    moves = [p for p in planning if p.get("phase") == "transport"]
    if any(p.get("kind") == "arc" and p.get("collision_free") and not p.get("reason") for p in moves):
        return "arc"
    if any(p.get("kind") == "rrt" and p.get("success") for p in moves):
        return "joint path"
    return None


class Runner:
    def __init__(self, settings: StudioSettings, store: StudioStore, world: World):
        self.settings, self.store, self.world = settings, store, world
        self.home = settings.home.copy()
        self.lock = threading.Lock()
        self._commands: queue.Queue = queue.Queue()
        self._frame: dict = {"seq": 0, "bodies": {}, "objects": {}, "gripper_width": 0.08, "live": False}
        self._run: dict | None = None
        self._plan: TaskPlan | None = None
        self._busy_since: float | None = None
        self._stop = self._pause = False
        self.speed = float(settings.studio["run"]["speed"])
        self.snapshot_dt = 1.0 / float(settings.studio["run"]["snapshot_hz"])
        self.idle_q = self.home.copy()
        self.layout_before_run: dict | None = None
        self.robot_bodies: list[str] = []
        self.error: str | None = None
        self._ready = threading.Event()
        self._closing = False
        self.thread = threading.Thread(target=self._loop, name="studio-sim", daemon=True)
        self.thread.start()
        self._ready.wait(60)

    # ------------------------------------------------------------------ public API (any thread)

    @property
    def busy(self) -> bool:
        with self.lock:
            return self._run is not None and self._run["state"] in ("starting", "running", "paused", "stopping",
                                                                      "settling")

    def start(self, plan: TaskPlan) -> dict:
        with self.lock:
            if self._run is not None and self._run["state"] in ("starting", "running", "paused", "stopping", "settling"):
                raise RuntimeError("A task is already running.")
            run = {"id": time.strftime("%H%M%S-") + uuid.uuid4().hex[:4], "task_id": plan.task_id,
                   "title": plan.title, "summary": plan.summary, "state": "starting", "started_at": time.time(),
                   "finished_at": None, "leg_index": 0, "phase": "init", "sim_time": 0.0, "message": "",
                   "failure": None, "notices": [],
                   "legs": [{**leg.to_dict(), "status": "pending", "result": None} for leg in plan.legs]}
            self._run, self._plan = run, plan
            self._stop = self._pause = False
            self._commands.put(("run", plan))
            return copy.deepcopy(run)

    def pause(self) -> None:
        with self.lock:
            if self._run and self._run["state"] == "running":
                self._pause = True
                self._run["state"] = "paused"

    def resume(self) -> None:
        with self.lock:
            if self._run and self._run["state"] == "paused":
                self._pause = False
                self._run["state"] = "running"

    def stop(self) -> None:
        with self.lock:
            if self._run and self._run["state"] in ("starting", "running", "paused"):
                self._stop, self._pause = True, False
                self._run["state"] = "stopping"

    def set_speed(self, value: float) -> None:
        with self.lock:
            self.speed = float(value)

    def reset_robot(self) -> None:
        if self.busy:
            raise RuntimeError("A task is running.")
        self._commands.put(("reset_robot", None))

    def dismiss(self) -> None:
        """Forget a finished run (the panel is closed)."""
        with self.lock:
            if self._run and self._run["state"] in ("succeeded", "failed", "stopped"):
                self._run = None

    def status(self) -> dict:
        with self.lock:
            run = copy.deepcopy(self._run)
            busy = self._busy_since is not None and time.perf_counter() - self._busy_since > 0.3
            return {"run": run, "speed": self.speed, "thinking": bool(busy and run and run["state"] == "running"),
                    "robot_at_home": bool(np.allclose(self.idle_q, self.home, atol=1e-3)), "error": self.error}

    def frame(self) -> dict:
        with self.lock:
            return dict(self._frame)

    def close(self) -> None:
        self._closing = True
        self.stop()
        self._commands.put(("quit", None))
        self.thread.join(10)

    # ------------------------------------------------------------------ worker

    def _loop(self) -> None:
        try:
            self._display = SimulationEngine(self.settings.environment, Panda(actuation="torque"),
                                             scene_overrides(self.world, []))
            self.robot_bodies = robot_body_names(self._display)
            self._publish_idle()
        except Exception:  # noqa: BLE001 - reported through status()
            self.error = traceback.format_exc()
        finally:
            self._ready.set()
        while not self._closing:
            try:
                cmd, arg = self._commands.get(timeout=0.25)
            except queue.Empty:
                self._publish_idle(objects_only=True)
                continue
            if cmd == "quit":
                break
            if cmd == "reset_robot":
                self.idle_q = self.home.copy()
                self._publish_idle()
            elif cmd == "run":
                try:
                    self._execute(arg)
                except Exception as exc:  # noqa: BLE001 - a crash ends the run, not the worker
                    traceback.print_exc()
                    with self.lock:
                        if self._run is not None:
                            self._run.update(state="failed", finished_at=time.time(),
                                             failure={"message": "Internal error while running the task.",
                                                      "hint": str(exc), "stage": "internal", "reason": repr(exc)})
                    self._publish_idle()

    def _publish_idle(self, objects_only: bool = False) -> None:
        objs = self.store.objects_copy()
        objects = {o.id: [*map(float, o.pos), *map(float, o.quat)] for o in objs}
        with self.lock:
            if objects_only and self._frame.get("live"):
                return
            frame = dict(self._frame)
        if not objects_only or not frame.get("bodies"):
            e = self._display
            e.data.qpos[e.arm_qpos_adr] = self.idle_q
            mujoco.mj_kinematics(e.model, e.data)
            frame["bodies"] = body_poses(e, self.robot_bodies)
            frame["gripper_width"] = float(e.data.qpos[e.model.joint("finger_joint1").qposadr[0]] * 2)
        frame.update(objects=objects, live=False, seq=frame.get("seq", 0) + 1)
        with self.lock:
            self._frame = frame

    def _publish(self, ep: PickPlaceEpisode, sim_offset: float, phase: str) -> None:
        e = ep.engine
        objs = self._object_ids
        frame = {"bodies": body_poses(e, self.robot_bodies), "objects": body_poses(e, objs),
                 "gripper_width": ep.finger_width(), "live": True}
        with self.lock:
            frame["seq"] = self._frame.get("seq", 0) + 1
            self._frame = frame
            if self._run is not None:
                self._run["phase"] = phase
                self._run["sim_time"] = round(sim_offset + float(e.time), 3)

    def _flags(self) -> tuple[float, bool, bool]:
        with self.lock:
            return self.speed, self._pause, self._stop

    def _drive(self, ep: PickPlaceEpisode, step, sim_offset: float, phase_of) -> bool:
        """Step `step()` (one physics step; False when finished) paced to the wall clock. Returns False if the
        user stopped the run."""
        wall_ref, sim_ref = time.perf_counter(), ep.engine.time
        last_pub = 0.0
        while True:
            speed, paused, stop = self._flags()
            if stop:
                return False
            now = time.perf_counter()
            if paused:
                if now - last_pub > 0.1:
                    self._publish(ep, sim_offset, phase_of())
                    last_pub = now
                time.sleep(0.02)
                wall_ref, sim_ref = time.perf_counter(), ep.engine.time
                continue
            target = np.inf if speed <= 0 else sim_ref + (now - wall_ref) * speed
            deadline = now + 0.02
            alive = True
            while ep.engine.time < target and time.perf_counter() < deadline:
                t0 = time.perf_counter()
                with self.lock:
                    self._busy_since = t0
                alive = step()
                with self.lock:
                    self._busy_since = None
                if time.perf_counter() - t0 > 0.05:  # planning: continue from here instead of catching up
                    wall_ref, sim_ref = time.perf_counter(), ep.engine.time
                    target = np.inf if speed <= 0 else sim_ref
                if not alive:
                    break
            now = time.perf_counter()
            if now - last_pub >= self.snapshot_dt or not alive:
                self._publish(ep, sim_offset, phase_of())
                last_pub = now
            if not alive:
                return True
            time.sleep(0.002)

    def _settle(self, ep: PickPlaceEpisode, sim_offset: float, max_time: float = 3.0) -> None:
        """Open the gripper, hold the arm where it is and wait until every object is at rest."""
        e = ep.engine
        ep.ref.hold(e.arm_qpos())
        e.set_gripper(ep.pp["gripper"]["open"])
        m, d = e.model, e.data
        dofs = [int(m.jnt_dofadr[m.joint(f"{oid}_free").id]) for oid in self._object_ids]
        t_end = e.time + max_time
        quiet = {"since": None}

        def step() -> bool:
            ref = ep.ref.reference(e.time)
            ctrl = ep.controller.compute(e.arm_qpos(), e.arm_qvel(), ref.q, ref.qd, e.dt, t=e.time, qdd_ref=ref.qdd)
            e.set_arm_command(ctrl)
            e.step()
            mujoco.mj_kinematics(m, d)
            fast = any(np.linalg.norm(d.qvel[a:a + 3]) > 0.005 for a in dofs)
            if fast:
                quiet["since"] = None
            elif quiet["since"] is None:
                quiet["since"] = e.time
            settled = quiet["since"] is not None and e.time - quiet["since"] > 0.3
            return not settled and e.time < t_end

        with self.lock:
            if self._run is not None and self._run["state"] in ("stopping", "running", "paused"):
                self._run["state"] = "settling"
            self._stop = False
            self._pause = False
        self._drive(ep, step, sim_offset, lambda: "settle")

    def _episode(self, leg: Leg, objects: list[SceneObject], q_start: np.ndarray, last: bool, k: int):
        cfg = self.settings.episode_config()
        pp = cfg["pick_place"]
        o = next(obj for obj in objects if obj.id == leg.object_id)
        half = 0.5 * np.asarray(o.size, float)
        pp["object"] = {"body": o.id, "geom": f"{o.id}_geom", "half_size": half.tolist(), "mass": float(o.mass),
                        "grasp_width": float(leg.grasp_width)}
        pp["grasp"]["yaw"] = float(leg.grasp_yaw)
        pp["grasp"]["height_above_center"] = float(half[2] - leg.depth)
        pp["grasp"]["lift_height"] = float(leg.lift)
        pp["start"] = [float(v) for v in q_start]
        pp["return_home"] = [float(v) for v in self.home] if last else None
        cfg["controller_model"]["nominal_masses"] = {obj.id: 0.4 for obj in objects}
        cfg["environment_overrides"] = scene_overrides(self.world, objects)
        cfg["label"] = "studio"
        case = {"name": f"leg{k + 1}", "object_pose": {"pos": list(o.pos), "quat": list(o.quat)},
                "place": [float(leg.target[0]), float(leg.target[1])], "place_z": float(leg.target[2]),
                "mass": float(o.mass), "seed": k, "supports": sorted(set(leg.supports_from) | set(leg.supports_to))}
        return PickPlaceEpisode(cfg, case)

    def _execute(self, plan: TaskPlan) -> None:
        self.layout_before_run = self.store.snapshot()
        notices = []
        if not np.allclose(self.idle_q, self.home, atol=1e-3):
            notices.append("The robot was moved back to its rest position before starting.")
        q = self.home.copy()
        sim_offset = 0.0
        with self.lock:
            run = self._run
            run["notices"] = notices
            if run["state"] == "starting":
                run["state"] = "running"
        outcome = "succeeded"
        for k, leg in enumerate(plan.legs):
            if self._flags()[2]:
                outcome = "stopped"
                break
            objects = self.store.objects_copy()
            self._object_ids = [o.id for o in objects]
            with self.lock:
                run["leg_index"] = k
                run["legs"][k]["status"] = "active"
                run["phase"] = "init"
            ep = self._episode(leg, objects, q, k == len(plan.legs) - 1, k)
            ep.start()
            finished = self._drive(ep, ep.advance, sim_offset, lambda: ep.phase)
            if not finished:
                ep.abort("stopped by the user")
            r = ep.finish()
            ok = r["success"] or r["failure_stage"] == "home"
            if not ok:
                self._settle(ep, sim_offset)
            e = ep.engine
            poses = {oid: (e.data.xpos[e.model.body(oid).id].tolist(), e.data.xquat[e.model.body(oid).id].tolist())
                     for oid in self._object_ids}
            self.store.set_poses(poses)
            q = e.arm_qpos()
            sim_offset += float(e.time)
            msg, hint = ("", "") if r["success"] else explain_failure(r["failure_stage"], r["failure_reason"],
                                                                     leg.object_name)
            result = {"success": bool(r["success"]), "stage": r["failure_stage"], "reason": r["failure_reason"],
                      "message": msg, "hint": hint, "placement_error_mm": r.get("placement_error_mm"),
                      "tilt_deg": r.get("final_tilt_deg"), "sim_time_s": round(float(e.time), 2),
                      "wall_time_s": round(float(r["wall_time_s"]), 2), "controller": r["controller_type"],
                      "phases": [{"phase": p["phase"], "t_start": p["t_start"], "t_end": p["t_end"]} for p in r["phases"]],
                      "transport": _transport_kind(r["planning"])}
            with self.lock:
                run["legs"][k]["status"] = "done" if ok else ("stopped" if r["failure_stage"] == "stopped" else "failed")
                run["legs"][k]["result"] = result
                run["sim_time"] = round(sim_offset, 3)
                if r["failure_stage"] == "home":
                    run["notices"].append(msg)
            if not ok:
                outcome = "stopped" if r["failure_stage"] == "stopped" else "failed"
                with self.lock:
                    run["failure"] = {"leg": k, "stage": r["failure_stage"], "reason": r["failure_reason"],
                                      "message": msg, "hint": hint}
                break
        self.idle_q = np.asarray(q, float)
        with self.lock:
            for leg_rec in run["legs"]:
                if leg_rec["status"] == "pending":
                    leg_rec["status"] = "skipped"
            run["state"] = outcome
            run["finished_at"] = time.time()
            run["phase"] = "done"
            self._stop = self._pause = False
        self._publish_idle()
