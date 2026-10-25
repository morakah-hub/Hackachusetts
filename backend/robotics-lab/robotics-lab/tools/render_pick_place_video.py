r"""Demo video (v0.6): known-world physical pick-and-place around an obstacle, rendered offscreen.

    .\.venv\Scripts\python.exe tools\render_pick_place_video.py      # -> results\video\v0.6_pick_place.webp

Clip 1: evaluation case eval_06 (object on one side of the obstacle, target on the other), torque MPC
with the disturbance observer, 1.15 kg steel block (unmodelled payload). Clip 2: negative control,
the Menagerie gripper gains (2.5 N squeeze) with a 1.0 kg block: the fingers close on the block, the
lift check sees that it did not rise, and the episode is rejected.

Each clip is an ordinary episode (simulation/manipulation.py): the object moves only through contact
and friction. Frames are rendered every 1/fps s of simulated time, so playback is real time (1x).
Every frame shows the version, the active controller (MPC or its CTC fallback) and the phase.
Existing files are never overwritten unless --overwrite is given (earlier videos are kept).
"""

import argparse
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_demo_video as base  # noqa: E402
from simulation.engine import PROJECT_ROOT  # noqa: E402
from simulation.manipulation import PickPlaceEpisode  # noqa: E402
from simulation.runner import load_config  # noqa: E402

VERSION = "v0.6"
STUDY = PROJECT_ROOT / "configs" / "pick_place_study.yaml"


def configured(cfg: dict, kind: str, observer: bool) -> dict:
    c = copy.deepcopy(cfg)
    c["controller"]["type"] = kind
    for sec in ("ctc", "torque_mpc"):
        c["controller"][sec].setdefault("disturbance_observer", {})["enabled"] = observer
    return c


def render_episode(video: "base.Video", ep: PickPlaceEpisode, title: str, lines: list[str]) -> dict:
    e = ep.engine
    m = e.model
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, video.w), max(
        m.vis.global_.offheight, video.h)
    renderer = mujoco.Renderer(m, video.h, video.w)
    cam = mujoco.MjvCamera()
    a = video.args
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = a.lookat, a.distance, a.azimuth, a.elevation
    every = max(1, int(round(1.0 / (video.fps * e.dt))))
    z0 = float(e.data.xpos[ep.obj_body][2])

    def frame(episode):
        if episode.steps % every:
            return
        renderer.update_scene(e.data, cam)
        mpc = hasattr(episode.controller, "using_fallback")
        active = ("torque MPC" if not episode.controller.using_fallback else "CTC fallback") if mpc else "CTC"
        lift = 1e3 * (float(e.data.xpos[episode.obj_body][2]) - z0)
        status = (f"t = {e.time:4.1f} s  phase: {episode.phase:<9}  active: {active}  object +{lift:5.1f} mm  "
                  f"grip {1e3 * episode.finger_width():4.1f} mm  forbidden contacts {episode.forbidden_steps}")
        video.emit(video.annotate(renderer.render(), title, lines, status))

    try:
        result = ep.run(on_step=frame)
        verdict = "SUCCESS" if result["success"] else f"REJECTED at {result['failure_stage']}: {result['failure_reason']}"
        renderer.update_scene(e.data, cam)
        last = video.annotate(renderer.render(), title, lines, verdict)
        for _ in range(int(2.5 * video.fps)):
            video.emit(last)
    finally:
        renderer.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=f"Render the {VERSION} pick-and-place demo video")
    p.add_argument("--output", default=str(PROJECT_ROOT / "results" / "video" / f"{VERSION}_pick_place.webp"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--frames-dir", default=None)
    p.add_argument("--azimuth", type=float, default=200.0)
    p.add_argument("--elevation", type=float, default=-28.0)
    p.add_argument("--distance", type=float, default=1.45)
    p.add_argument("--lookat", type=float, nargs=3, default=[0.45, 0.0, 0.50])
    p.add_argument("--case", default="eval_06")
    args = p.parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists (earlier videos are kept as evidence); use --output or --overwrite")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    base.VERSION = VERSION  # the version stamp drawn on every frame
    video = base.Video(args)
    cfg = load_config(STUDY)
    case = next(c for c in cfg["cases"] if c["name"] == args.case)

    video.card(f"Known-world physical pick-and-place (Robotics Lab {VERSION})",
               ["Franka Panda in MuJoCo, torque-actuated. Playback: real time (1x).",
                "KNOWN WORLD: object pose, size and obstacle are read from the simulator (no perception).",
                "The block moves only through simulated contact and friction (no weld, no teleport).",
                "Clip 1: torque MPC + disturbance observer, 1.15 kg steel block unknown to the controller.",
                "Clip 2: negative control - original Menagerie gripper gains (2.5 N squeeze), 1.0 kg block."], 6.0)
    r1 = render_episode(video, PickPlaceEpisode(configured(cfg, "torque_mpc", True), dict(case, mass=1.15)),
                        f"{VERSION} pick-and-place, torque MPC + observer (50 Hz QP), 1.15 kg, known world, 1x",
                        ["Phases: approach -> descend -> close -> lift -> hold 1 s -> transport (planned around the",
                         "blue obstacle, carried block checked with 20 mm clearance) -> lower -> release -> retreat ->",
                         "verify: lift >= 50 mm, no slip/drop, 0 forbidden contacts, final position within 10 mm."])
    print(f"clip 1: success={r1['success']} placement {r1.get('placement_error_mm')} mm, "
          f"MPC fallback steps {r1['mpc']['fallback_steps']}")
    weak = dict(case, mass=1.0, fault={"robot": {"gripper_stiffness": 100.0, "gripper_damping": 10.0}})
    r2 = render_episode(video, PickPlaceEpisode(configured(cfg, "torque_ctc", True), weak),
                        f"{VERSION} negative control: weak grip (Menagerie gains), 1.0 kg, CTC + observer, 1x",
                        ["The fingers close on the block (width check passes), but 2.5 N of squeeze cannot",
                         "carry 9.8 N: the measured lift is ~0 mm, so the supervisor rejects the episode."])
    print(f"clip 2: success={r2['success']} stage={r2['failure_stage']}")
    video.stream.close()
    print(f"wrote {args.output} ({video.stream.frames} frames, {video.stream.frames / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
