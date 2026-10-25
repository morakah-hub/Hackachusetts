r"""Recorded assembly runs: about two minutes of simulated robot work, saved to results/.

One table, one session, three runs with the camera-guided pick-and-place (the same code path as the app):

    1. Assemble a bridge  (two pillars, then a beam laid across them)
    2. Assemble a tower   (base block to its spot, middle block on it, top block on top)
    3. Crown the bridge   (take the top block off the tower and stack it on the bridge's beam)

    .venv\Scripts\python.exe scripts\run_assembly_suite.py            # writes results\assembly_<date>\
    .venv\Scripts\python.exe scripts\run_assembly_suite.py --no-video # faster, no GIF/PNG frames

Outputs: summary.md (table of every move), moves.csv, runs.json (full run records with phase timings and
joint traces), session.gif (third-person video of the whole session) and one PNG of the table after each run.
Physics runs as fast as the computer allows; the reported times are simulated seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from armstrong import __version__  # noqa: E402
from armstrong.planner import TaskPlanner  # noqa: E402
from armstrong.rules import RobotRules, World  # noqa: E402
from armstrong.sim import SimWorker  # noqa: E402
from armstrong.store import StudioStore  # noqa: E402

BRIDGE_SPOT = (0.56, 0.13)
TOWER_SPOT = (0.48, -0.15)
FPS = 5                       # video frames per simulated second
VIDEO_W, VIDEO_H = 480, 360


def names(store: StudioStore) -> dict[str, str]:
    return {o.name: o.id for o in store.objects.values()}


class Recorder:
    """Third-person frames every 1/FPS simulated seconds (the RGB-D camera used for perception is separate)."""

    def __init__(self, worker: SimWorker, enabled: bool):
        self.enabled = enabled
        self.frames: list[Image.Image] = []
        self.next_t = 0.0
        self.renderer = None
        self.model = None
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.40, 0.0, 1.0]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.75, 200.0, -24.0
        self.caption = ""
        self.worker = worker

    def __call__(self, w: SimWorker) -> None:
        if not self.enabled or w.data.time < self.next_t:
            return
        self.next_t = w.data.time + 1.0 / FPS
        self.frames.append(self.snapshot())

    def snapshot(self) -> Image.Image:
        w = self.worker
        if self.renderer is None or self.model is not w.model:   # the model is rebuilt only if objects change
            self.renderer, self.model = mujoco.Renderer(w.model, VIDEO_H, VIDEO_W), w.model
        self.renderer.update_scene(w.data, camera=self.cam)
        img = Image.fromarray(self.renderer.render())
        if self.caption:
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, VIDEO_W, 22], fill=(22, 25, 29))
            phase = w.task.phase if (w.task is not None and w.task_iter is not None) else ""
            d.text((8, 5), f"{self.caption}   {phase}   t = {w.data.time:6.1f} s", fill=(232, 234, 237))
        return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output folder (default: results/assembly_<date>)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--mode", default="camera_rgbd", choices=["camera_rgbd", "known_world"])
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "results" / f"assembly_{datetime.now():%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)

    rules, world = RobotRules(), World()
    store = StudioStore(None, rules, world, persist=False)       # the "Assembly kit" layout
    planner = TaskPlanner(store, world, rules)
    worker = SimWorker(store, realtime=False)
    worker.setup_sync()
    worker.mode = args.mode
    rec = Recorder(worker, not args.no_video)
    worker.on_step = rec
    t_session0 = float(worker.data.time)

    scenarios = [
        ("Bridge", "Assemble a bridge: two pillars, then a beam across them",
         "assemble", lambda: {"recipe": "bridge", "target": list(BRIDGE_SPOT)}),
        ("Tower", "Assemble a tower of three blocks",
         "assemble", lambda: {"recipe": "tower", "target": list(TOWER_SPOT)}),
        ("Crown", "Crown the bridge: move the tower's top block onto the beam",
         "stack", lambda: {"object": names(store)["Top block"], "base": names(store)["Beam"]}),
    ]
    runs, rows = [], []
    for i, (short, title, task, params) in enumerate(scenarios, start=1):
        worker._sync_scene()
        plan = planner.plan(task, params())
        print(f"[{i}/{len(scenarios)}] {title}: {len(plan.legs)} moves planned in {plan.check_s:.2f} s"
              + ("" if plan.ok else f"  NOT READY: {[p.message for p in plan.all_problems()]}"), flush=True)
        if not plan.ok:
            runs.append({"scenario": title, "planned": False, "problems": [p.to_dict() for p in plan.all_problems()]})
            continue
        rec.caption = f"Run {i}: {short}"
        wall0 = time.perf_counter()
        run = worker.run_sync(plan)
        wall = time.perf_counter() - wall0
        record = worker.results[run["id"]]
        runs.append({"scenario": title, "planned": True, "state": run["state"], "sim_time_s": round(run["sim_time"], 2),
                     "wall_time_s": round(wall, 2), "run": record["run"], "trace_columns": record["trace_columns"],
                     "trace": record["trace"]})
        for k, leg in enumerate(run["legs"], start=1):
            r = leg["result"] or {}
            rows.append({"run": i, "scenario": short, "move": k, "object": leg["object_name"], "label": leg["label"],
                         "status": leg["status"], "placement_error_mm": r.get("placement_error_mm"),
                         "height_error_mm": r.get("height_error_mm"), "tilt_deg": r.get("tilt_deg"),
                         "camera_estimate_error_mm": r.get("estimate_error_mm"), "lift_mm": r.get("lift_mm"),
                         "grasp_width_mm": r.get("grasp_width_mm"), "sim_time_s": r.get("sim_time_s"),
                         "transport": r.get("transport"), "failure": r.get("reason")})
        print(f"      -> {run['state']}: {run['sim_time']:.1f} s simulated ({wall:.1f} s on this computer)", flush=True)
        if rec.enabled:
            rec.snapshot().save(out / f"after_run{i}_{short.lower()}.png")
    total_sim = float(worker.data.time) - t_session0
    ok_moves = sum(1 for r in rows if r["status"] == "done")

    with open(out / "moves.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "armstrong": __version__,
            "mujoco": mujoco.__version__, "python": platform.python_version(), "perception_mode": args.mode,
            "total_simulated_s": round(total_sim, 2), "runs": runs}
    (out / "runs.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")

    if rec.enabled and rec.frames:
        frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in rec.frames]
        frames[0].save(out / "session.gif", save_all=True, append_images=frames[1:], duration=int(1000 / FPS),
                       loop=0, optimize=True)

    errs = [r["placement_error_mm"] for r in rows if r["status"] == "done"]
    cam = [r["camera_estimate_error_mm"] for r in rows if r["camera_estimate_error_mm"] is not None]
    lines = [f"# Assembly runs ({datetime.now():%d %b %Y})", "",
             f"Total simulated robot time: **{total_sim:.1f} s** ({total_sim / 60:.2f} min) in {len(runs)} runs, "
             f"{ok_moves}/{len(rows)} moves succeeded. Objects located with: "
             f"{'the simulated RGB-D camera' if args.mode == 'camera_rgbd' else 'simulator poses'}.", ""]
    if errs:
        lines += [f"Placement error: mean {np.mean(errs):.2f} mm, max {np.max(errs):.2f} mm (pass limit 15 mm)."]
    if cam:
        lines += [f"Camera localization error (before each pick): mean {np.mean(cam):.2f} mm, max {np.max(cam):.2f} mm.", ""]
    lines += ["| Run | Scenario | Result | Simulated time |", "|---|---|---|---|"]
    for i, r in enumerate(runs, start=1):
        lines.append(f"| {i} | {r['scenario']} | {r.get('state', 'not planned')} | {r.get('sim_time_s', 0):.1f} s |")
    lines += ["", "| Run | Move | Object | Result | Placement error | Tilt | Camera error | Time |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ce = "-" if r["camera_estimate_error_mm"] is None else f"{r['camera_estimate_error_mm']:.1f} mm"
        lines.append(f"| {r['run']} | {r['label']} | {r['object']} | {r['status']} | {r['placement_error_mm']:.1f} mm | "
                     f"{r['tilt_deg']:.1f} deg | {ce} | {r['sim_time_s']:.1f} s |")
    if rec.enabled:
        lines += ["", "![Assembly session](session.gif)", "",
                  "Snapshots after each run: " + ", ".join(f"[run {i}](after_run{i}_{s[0].lower()}.png)"
                                                          for i, s in enumerate(scenarios, start=1))]
    lines += ["", "Files: `moves.csv` (one row per move), `runs.json` (full run records with phase timings and joint "
                  "traces).", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTotal simulated time {total_sim:.1f} s; {ok_moves}/{len(rows)} moves succeeded. Results in {out}")
    if worker.camera is not None:
        worker.camera.close()
    return 0 if ok_moves == len(rows) and all(r.get("state") == "succeeded" for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
