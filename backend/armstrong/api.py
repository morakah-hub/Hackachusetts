"""Local HTTP API (FastAPI) and static hosting of the built frontend.

Everything is local: the server binds to 127.0.0.1 and has no authentication. Scene edits go to the store
(the simulation worker picks them up while idle); run control goes through the worker's command queue;
live state and camera frames are read from what the worker last published.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import mujoco
import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .planner import TASKS, PlanError, TaskPlanner
from .rules import PLACE_TOL_MM, Problem, RobotRules, World
from .sim import CommandError, SimWorker
from .store import MAX_ASSET_BYTES, StoreError, StudioStore
from .task import PHASE_LABELS, PHASES

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "frontend" / "dist"


def _clean(obj):
    """JSON-safe: non-finite floats become null; numpy scalars/arrays become Python types."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _clean(obj.tolist())
    if hasattr(obj, "item"):
        return _clean(obj.item())
    return obj


def _error(message: str, status: int = 400, problems: list[Problem] | None = None, **extra) -> JSONResponse:
    return JSONResponse(_clean({"error": message, "problems": [p.to_dict() for p in problems or []], **extra}),
                        status_code=status)


class Studio:
    """Everything the routes need."""

    def __init__(self, worker: SimWorker, store: StudioStore, world: World, rules: RobotRules):
        self.worker, self.store, self.world, self.rules = worker, store, world, rules
        self.planner = TaskPlanner(store, world, rules)
        self._plans: dict[tuple, object] = {}
        self._plan_lock = threading.Lock()

    def config(self) -> dict:
        return {"version": __version__, "robot": self.rules.profile(), "world": self.world.to_dict(),
                "gap_m": self.store.gap, "tasks": TASKS,
                "phases": [{"id": p, "label": PHASE_LABELS[p]} for p in PHASES], "phase_labels": PHASE_LABELS,
                "controller": "ctc", "controller_name": "Computed torque control (500 Hz)",
                "physics": {"timestep_s": 0.002, "engine": f"MuJoCo {mujoco.__version__}"},
                "place_tol_mm": PLACE_TOL_MM,
                "camera": {"width": self.worker.cfg.camera.width if self.worker.model else 480,
                           "height": self.worker.cfg.camera.height if self.worker.model else 360},
                "workspace": {"measured_at": None, "wall_s": None, "profile": [], "probe_heights_m": [],
                              "measured_theta_max_deg": self.world.theta_max_deg},
                "limits": {"max_asset_mb": MAX_ASSET_BYTES // (1024 * 1024)}}

    def plan(self, task_id: str, params: dict):
        key = (task_id, json.dumps(params, sort_keys=True), self.store.revision)
        with self._plan_lock:
            cached = self._plans.get(key)
            if cached is not None:
                return cached
            plan = self.planner.plan(task_id, params)
            self._plans = {key: plan}
            return plan

    def command(self, name: str, timeout: float = 20.0, **kw):
        if not self.worker.ready.is_set():
            raise CommandError("LOADING", "The simulation is still starting.", 503)
        return self.worker.submit(name, **kw).result(timeout=timeout)


def create_app(studio: Studio) -> FastAPI:
    app = FastAPI(title="Armstrong Studio", version=__version__)
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    S = studio

    def editable():
        if S.worker.busy():
            raise StoreError("A task is running. Wait for it to finish or stop it first.", status=409)

    @app.exception_handler(StoreError)
    async def _store_error(_req, exc: StoreError):
        return _error(exc.message, exc.status, exc.problems)

    @app.exception_handler(PlanError)
    async def _plan_error(_req, exc: PlanError):
        return _error(exc.message, 400, exc.problems)

    @app.exception_handler(CommandError)
    async def _command_error(_req, exc: CommandError):
        return _error(exc.message, exc.status, code=exc.code)

    # ------------------------------------------------------------------ info

    @app.get("/api/health")
    def health():
        return {"ok": S.worker.ready.is_set() and S.worker.error is None, "version": __version__,
                "worker_error": S.worker.error}

    @app.get("/api/config")
    def config():
        if not S.worker.ready.wait(60):
            return _error("The simulation is still starting.", 503)
        return _clean(S.config())

    @app.get("/api/geometry")
    def geometry_manifest():
        if not S.worker.ready.wait(60) or S.worker.error:
            return _error(S.worker.error or "loading", 503)
        return S.worker.geometry[0]

    @app.get("/api/geometry.bin")
    def geometry_blob():
        if not S.worker.ready.wait(60) or S.worker.error:
            return _error(S.worker.error or "loading", 503)
        return Response(S.worker.geometry[1], media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/state")
    def state():
        return JSONResponse(_clean(S.worker.snapshot()))

    # ------------------------------------------------------------------ scene

    @app.get("/api/scene")
    def scene():
        return _clean(S.store.scene_dict())

    @app.post("/api/scene/objects")
    def add_object(body: dict = Body(...)):
        editable()
        return _clean(S.store.add_object(body))

    @app.patch("/api/scene/objects/{object_id}")
    def update_object(object_id: str, body: dict = Body(...)):
        editable()
        return _clean(S.store.update_object(object_id, body))

    @app.delete("/api/scene/objects/{object_id}")
    def delete_object(object_id: str):
        editable()
        S.store.remove_object(object_id)
        return {"ok": True}

    @app.post("/api/scene/reset")
    def reset_scene(body: dict = Body(default={})):
        editable()
        mode = (body or {}).get("mode")
        if mode == "empty":
            S.store.clear()
        else:
            S.store.reset_layout("objects" if mode == "objects" else "default")
        S.command("reset_robot")
        return _clean(S.store.scene_dict())

    @app.post("/api/scene/restore")
    def restore_scene():
        editable()
        snap = S.worker.layout_before_run
        if snap is None:
            raise StoreError("There is no earlier layout to restore.")
        S.store.restore(snap)
        S.command("reset_robot")
        return _clean(S.store.scene_dict())

    # ------------------------------------------------------------------ library and uploads

    @app.get("/api/library")
    def library():
        return _clean(S.store.library_list())

    @app.post("/api/library")
    def add_library_item(body: dict = Body(...)):
        return _clean(S.store.add_library_item(body))

    @app.delete("/api/library/{item_id}")
    def delete_library_item(item_id: str):
        S.store.delete_library_item(item_id)
        return {"ok": True}

    @app.post("/api/assets")
    async def upload_asset(request: Request, filename: str):
        size = request.headers.get("content-length")
        if size is not None and int(size) > MAX_ASSET_BYTES:
            raise StoreError(f"The file is larger than {MAX_ASSET_BYTES // (1024 * 1024)} MB.", status=413)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_ASSET_BYTES:
                raise StoreError(f"The file is larger than {MAX_ASSET_BYTES // (1024 * 1024)} MB.", status=413)
        return S.store.save_asset(filename, bytes(data))

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str):
        path, media = S.store.asset_file(asset_id)
        return FileResponse(path, media_type=media, headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------------ tasks and runs

    @app.post("/api/tasks/check")
    def check_task(body: dict = Body(...)):
        plan = S.plan(str(body.get("task_id")), body.get("params") or {})
        return _clean(plan.to_dict())

    @app.post("/api/tasks/run")
    def run_task(body: dict = Body(...)):
        editable()
        plan = S.plan(str(body.get("task_id")), body.get("params") or {})
        if not plan.ok:
            out = plan.to_dict()
            return _error("This task cannot run as set up.", 422, plan.all_problems(), plan=out)
        run = S.command("run", plan=plan)
        return _clean({"plan": plan.to_dict(), "run": run})

    @app.post("/api/run/{action}")
    def run_control(action: str, body: dict = Body(default={})):
        if action in ("pause", "resume", "stop", "dismiss"):
            S.command(action)
        elif action == "speed":
            try:
                value = float((body or {}).get("value"))
            except (TypeError, ValueError):
                raise StoreError("Speed must be a number.") from None
            if not np.isfinite(value) or value < 0 or value > 8:
                raise StoreError("Speed must be between 0 (as fast as possible) and 8.")
            S.command("speed", value=value)
        else:
            return _error(f"Unknown action '{action}'.", 404)
        return _clean(S.worker.snapshot())

    @app.post("/api/robot/reset")
    def reset_robot():
        S.command("reset_robot")
        return {"ok": True}

    @app.get("/api/runs/{run_id}/export")
    def run_export(run_id: str):
        rec = S.worker.results.get(run_id)
        if rec is None:
            return _error(f"Unknown run {run_id}.", 404)
        return Response(json.dumps(_clean(rec), indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="armstrong_run_{run_id}.json"'})

    # ------------------------------------------------------------------ perception (Armstrong)

    @app.post("/api/settings")
    def settings(body: dict = Body(...)):
        out = {}
        if "perception_mode" in body:
            out.update(S.command("mode", mode=str(body["perception_mode"])))
        return out

    @app.post("/api/faults")
    def faults(body: dict = Body(...)):
        return S.command("faults", camera_masked=body.get("camera_masked"), gripper_disabled=body.get("gripper_disabled"))

    def _frame_response(kind: str):
        fr = S.worker.frame()
        if fr is None:
            return _error("No camera frame yet.", 503)
        meta = fr["meta"]
        return Response(fr[kind], media_type="image/jpeg" if kind == "rgb" else "image/png", headers={
            "Cache-Control": "no-store", "X-Frame-Id": str(meta["frame_id"]),
            "X-Sim-Time": f"{meta['sim_time_s']:.3f}"})

    @app.get("/api/camera/rgb")
    def cam_rgb():
        return _frame_response("rgb")

    @app.get("/api/camera/depth")
    def cam_depth():
        return _frame_response("depth")

    @app.get("/api/camera/meta")
    def cam_meta():
        fr = S.worker.frame()
        return JSONResponse(_clean(fr["meta"] if fr else None))

    # ------------------------------------------------------------------ frontend

    if (DIST / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):
                return _error("Not found.", 404)
            f = (DIST / path).resolve()
            if path and f.is_file() and DIST.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/")
        def no_frontend():
            return JSONResponse({"message": "The frontend is not built. Run start.bat (it builds frontend/dist)."})

    return app
