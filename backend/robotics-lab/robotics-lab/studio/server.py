"""ArmStrong HTTP API (FastAPI) and static hosting of the built frontend.

Everything is local: the server binds to 127.0.0.1 by default and has no authentication, so do not expose it
on a shared network. Handlers are plain functions (FastAPI runs them in its thread pool); the simulation runs
on the runner's worker thread, and the planner's checks run on private scene copies.
"""

import json
import time
from contextlib import asynccontextmanager

import mujoco
import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from robots.panda import Panda
from simulation.engine import SimulationEngine
from studio import geometry
from studio.planner import TASKS, PlanError, TaskPlanner, scene_overrides
from studio.robot import Problem, RobotRules
from studio.runner import PHASE_LABELS, PHASE_ORDER, Runner
from studio.scene import MAX_ASSET_BYTES, StoreError, StudioStore, World
from studio.settings import StudioSettings
from studio.workspace import load_or_measure

VERSION = "0.1.0"


class Studio:
    """Everything the routes need, built once at startup."""

    def __init__(self, settings: StudioSettings, log=print):
        self.settings = settings
        ws = load_or_measure(settings, log)
        zone = ws["zone"]
        theta = min(float(zone["theta_max_deg"]), float(settings.studio["workspace"].get("max_azimuth_deg", 180.0)))
        self.workspace = ws
        self.world = World(table_height=settings.table_height, table_radius=float(ws["table_radius"]),
                           r_in=float(zone["r_in"]), r_out=float(zone["r_out"]), theta_max_deg=theta)
        self.rules = RobotRules(settings)
        self.store = StudioStore(settings.data_dir, self.rules, self.world, settings.limits["gap_m"])
        self.planner = TaskPlanner(settings, self.store, self.world, self.rules)
        engine = SimulationEngine(settings.environment, Panda(actuation="torque"), scene_overrides(self.world, []))
        self.geometry_manifest, self.geometry_blob = geometry.export(
            engine, ["table", *geometry.robot_body_names(engine)])
        self.runner = Runner(settings, self.store, self.world)
        self._plans: dict[tuple, object] = {}

    def config(self) -> dict:
        s = self.settings.studio
        return {"version": VERSION, "robot": self.rules.profile(), "world": self.world.to_dict(),
                "grasp_depth": s["grasp_depth"], "gap_m": self.settings.limits["gap_m"],
                "tasks": TASKS, "phases": [{"id": p, "label": PHASE_LABELS[p]} for p in PHASE_ORDER + ["home"]],
                "phase_labels": PHASE_LABELS, "controller": self.settings.cfg["controller"]["type"],
                "physics": {"timestep_s": 0.002, "engine": f"MuJoCo {mujoco.__version__}"},
                "workspace": {"measured_at": self.workspace.get("measured_at"), "wall_s": self.workspace.get("wall_s"),
                              "profile": self.workspace.get("profile"), "probe_heights_m":
                                  s["workspace"]["probe_heights_m"], "measured_theta_max_deg":
                                  self.workspace["zone"]["theta_max_deg"]},
                "limits": {"max_asset_mb": MAX_ASSET_BYTES // (1024 * 1024)}}

    def plan(self, task_id: str, params: dict):
        rev = self.store.revision
        key = (task_id, json.dumps(params, sort_keys=True), rev)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        plan = self.planner.plan(task_id, params)
        self._plans = {key: plan}  # keep only the latest
        return plan


def _error(message: str, status: int = 400, problems: list[Problem] | None = None) -> JSONResponse:
    return JSONResponse({"error": message, "problems": [p.to_dict() for p in problems or []]}, status_code=status)


def create_app(settings: StudioSettings | None = None, studio: Studio | None = None) -> FastAPI:
    settings = settings or StudioSettings.load()
    holder: dict = {"studio": studio}

    @asynccontextmanager
    async def lifespan(_app):
        if holder["studio"] is None:
            holder["studio"] = Studio(settings)
        yield
        holder["studio"].runner.close()

    app = FastAPI(title="ArmStrong", version=VERSION, lifespan=lifespan)

    def S() -> Studio:  # noqa: N802 - short accessor
        return holder["studio"]

    def editable():
        if S().runner.busy:
            raise StoreError("A task is running. Wait for it to finish or stop it first.", status=409)

    @app.exception_handler(StoreError)
    async def _store_error(_req, exc: StoreError):
        return _error(exc.message, exc.status, exc.problems)

    @app.exception_handler(PlanError)
    async def _plan_error(_req, exc: PlanError):
        return _error(exc.message, 400, exc.problems)

    # ------------------------------------------------------------------ info

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": VERSION, "worker_error": S().runner.error}

    @app.get("/api/config")
    def config():
        return S().config()

    @app.get("/api/geometry")
    def geometry_manifest():
        return S().geometry_manifest

    @app.get("/api/geometry.bin")
    def geometry_blob():
        return Response(S().geometry_blob, media_type="application/octet-stream",
                        headers={"Cache-Control": "max-age=3600"})

    # ------------------------------------------------------------------ scene

    @app.get("/api/scene")
    def scene():
        return S().store.scene_dict()

    @app.post("/api/scene/objects")
    def add_object(body: dict = Body(...)):
        editable()
        return S().store.add_object(body)

    @app.patch("/api/scene/objects/{object_id}")
    def update_object(object_id: str, body: dict = Body(...)):
        editable()
        return S().store.update_object(object_id, body)

    @app.delete("/api/scene/objects/{object_id}")
    def delete_object(object_id: str):
        editable()
        S().store.remove_object(object_id)
        return {"ok": True}

    @app.post("/api/scene/reset")
    def reset_scene(body: dict = Body(default={})):
        editable()
        if (body or {}).get("mode") == "empty":
            S().store.clear()
        else:
            S().store.reset_default()
        S().runner.reset_robot()
        return S().store.scene_dict()

    @app.post("/api/scene/restore")
    def restore_scene():
        editable()
        snap = S().runner.layout_before_run
        if snap is None:
            raise StoreError("There is no earlier layout to restore.")
        S().store.restore(snap)
        S().runner.reset_robot()
        return S().store.scene_dict()

    # ------------------------------------------------------------------ library and uploads

    @app.get("/api/library")
    def library():
        return S().store.library_list()

    @app.post("/api/library")
    def add_library_item(body: dict = Body(...)):
        return S().store.add_library_item(body)

    @app.delete("/api/library/{item_id}")
    def delete_library_item(item_id: str):
        S().store.delete_library_item(item_id)
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
        return S().store.save_asset(filename, bytes(data))

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str):
        path, media = S().store.asset_file(asset_id)
        return FileResponse(path, media_type=media, headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------------ tasks and runs

    @app.post("/api/tasks/check")
    def check_task(body: dict = Body(...)):
        plan = S().plan(str(body.get("task_id")), body.get("params") or {})
        return plan.to_dict()

    @app.post("/api/tasks/run")
    def run_task(body: dict = Body(...)):
        editable()
        plan = S().plan(str(body.get("task_id")), body.get("params") or {})
        if not plan.ok:
            out = plan.to_dict()
            return JSONResponse({"error": "This task cannot run as set up.", "plan": out,
                                 "problems": out["problems"]}, status_code=422)
        try:
            run = S().runner.start(plan)
        except RuntimeError as exc:
            raise StoreError(str(exc), status=409) from None
        return {"plan": plan.to_dict(), "run": run}

    @app.get("/api/state")
    def state():
        st = S().runner.status()
        return {"time": time.time(), "revision": S().store.revision, **st, "frame": S().runner.frame()}

    @app.post("/api/run/pause")
    def pause():
        S().runner.pause()
        return S().runner.status()

    @app.post("/api/run/resume")
    def resume():
        S().runner.resume()
        return S().runner.status()

    @app.post("/api/run/stop")
    def stop():
        S().runner.stop()
        return S().runner.status()

    @app.post("/api/run/dismiss")
    def dismiss():
        S().runner.dismiss()
        return S().runner.status()

    @app.post("/api/run/speed")
    def speed(body: dict = Body(...)):
        try:
            value = float(body.get("value"))
        except (TypeError, ValueError):
            raise StoreError("Speed must be a number.") from None
        if not np.isfinite(value) or value < 0 or value > 8:
            raise StoreError("Speed must be between 0 (as fast as possible) and 8.")
        S().runner.set_speed(value)
        return S().runner.status()

    @app.post("/api/robot/reset")
    def reset_robot():
        try:
            S().runner.reset_robot()
        except RuntimeError as exc:
            raise StoreError(str(exc), status=409) from None
        return {"ok": True}

    # ------------------------------------------------------------------ frontend

    dist = settings.frontend_dist
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def no_frontend():
            return (f"<h2>ArmStrong API is running.</h2><p>The frontend is not built yet. In "
                    f"<code>frontend/</code> run <code>npm install</code> and <code>npm run build</code>, then reload "
                    f"(or use <code>npm run dev</code> for development). Expected: <code>{dist}</code></p>"
                    f"<p>API docs: <a href='/docs'>/docs</a></p>")
    return app
