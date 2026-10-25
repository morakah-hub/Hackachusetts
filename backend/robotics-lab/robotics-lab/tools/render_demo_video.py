"""Demo video (v0.5.1, re-labelled v0.5.2): the Panda reaching over the wall, rendered offscreen from MuJoCo.

    python tools/render_demo_video.py                         # -> results/video/v0.5.2_demo.webp
    python tools/render_demo_video.py --include-unsafe        # add the unchecked direct move (collides)
    python tools/render_demo_video.py --output demo.webp --fps 25 --width 960 --height 540
    python tools/render_demo_video.py --frames-dir frames     # also save every frame as PNG (for ffmpeg)

Every clip is a normal SimulationRunner run: the planner runs once at task
creation, the controller (torque MPC from configs/planning_demo.yaml) produces
torques, and the robot moves only through mj_step. Frames are rendered every
1/fps s of simulated time, so the video plays at real-time speed. Overlays are
drawn into the render scene only (visual geoms, never in the model).

Clips:
  1. planning on, nominal wall: blocked direct move, RRT-Connect path, executed path
  2. planning off: the v0.3/v0.4 straight-path check rejects the blocked target (red line)
  3. (--include-unsafe) planning off and path/torque checks deliberately disabled: the arm hits the wall
  4. a new episode with a taller wall (configs/robustness_wall.yaml, variant `taller`): the
     old plan is re-checked against the known new geometry, found invalid, and replaced

Output is an animated WebP written frame by frame with Pillow (constant memory;
no ffmpeg needed). Convert to MP4 with ffmpeg if available, or record the viewer
(see docs/research/v0.5.1-demo-brief.md).

v0.5.2: the planner uses the corrected collision checker (exact distances for the
fingertip boxes, docs/research/v0.5.2-results.md); the title card says so. The
v0.5.1 video (results/video/v0.5.1_demo.webp) is historical evidence: existing
files are never overwritten unless --overwrite is given.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.visualization import add_polyline, add_points, draw_planned_paths, ee_polyline  # noqa: E402
from simulation.engine import PROJECT_ROOT  # noqa: E402
from simulation.kinematics import ArmKinematics  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

DEMO = PROJECT_ROOT / "configs" / "planning_demo.yaml"
ROBUSTNESS = PROJECT_ROOT / "configs" / "robustness_wall.yaml"
BLOCKED_RGBA = (0.95, 0.15, 0.15, 0.85)
VERSION = "v0.5.2"


class WebPStream:
    """Animated WebP written one frame at a time (Pillow's encoder keeps only compressed frames)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int, quality: int = 80):
        from PIL import _webp  # Pillow's libwebp binding; Image.save(save_all=True) needs every frame in memory

        self.path, self.dt_ms, self.quality, self.t, self.frames = path, 1000.0 / fps, quality, 0.0, 0
        self.enc = _webp.WebPAnimEncoder(size, 0, 0, False, 3, 5, False, False)

    def add(self, img: Image.Image) -> None:
        self.enc.add(img.convert("RGB").getim(), round(self.t), False, self.quality, 100, 4)
        self.t += self.dt_ms
        self.frames += 1

    def close(self) -> None:
        self.enc.add(None, round(self.t), False, self.quality, 100, 0)
        data = self.enc.assemble("", "", "")
        if data is None:
            raise OSError("WebP encoder returned no data")
        self.path.write_bytes(data)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class Video:
    def __init__(self, args):
        self.args, self.w, self.h, self.fps = args, args.width, args.height, args.fps
        self.stream = WebPStream(Path(args.output), (self.w, self.h), self.fps)
        self.frames_dir = Path(args.frames_dir) if args.frames_dir else None
        if self.frames_dir:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.big, self.small = font(max(14, self.h // 22)), font(max(11, self.h // 30))

    def emit(self, img: Image.Image) -> None:
        if self.frames_dir:
            img.save(self.frames_dir / f"frame_{self.stream.frames:05d}.png")
        self.stream.add(img)

    def annotate(self, rgb: np.ndarray, title: str, lines: list[str], status: str = "") -> Image.Image:
        img = Image.fromarray(rgb).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        pad, line_h = 10, self.small.size + 5
        draw.rectangle((0, 0, self.w, self.big.size + 2 * pad), fill=(0, 0, 0, 150))
        draw.text((pad, pad), title, font=self.big, fill=(255, 255, 255, 255))
        box_h = line_h * len(lines) + 2 * pad
        draw.rectangle((0, self.h - box_h, self.w, self.h), fill=(0, 0, 0, 150))
        for i, line in enumerate(lines):
            draw.text((pad, self.h - box_h + pad + i * line_h), line, font=self.small, fill=(235, 235, 235, 255))
        if status:
            tw = draw.textlength(status, font=self.small)
            draw.rectangle((self.w - tw - 3 * pad, self.big.size + 2 * pad, self.w, self.big.size + 2 * pad + line_h + pad),
                           fill=(0, 0, 0, 150))
            draw.text((self.w - tw - 2 * pad, self.big.size + 2 * pad + pad // 2), status, font=self.small,
                      fill=(255, 255, 255, 255))
        tw = draw.textlength(VERSION, font=self.small)  # every frame says which version rendered it
        y = self.h - box_h - line_h - pad
        draw.rectangle((self.w - tw - 2 * pad, y, self.w, y + line_h + pad // 2), fill=(0, 0, 0, 150))
        draw.text((self.w - tw - pad, y + pad // 4), VERSION, font=self.small, fill=(255, 220, 120, 255))
        return Image.alpha_composite(img, layer).convert("RGB")

    def card(self, title: str, lines: list[str], seconds: float) -> None:
        img = Image.new("RGB", (self.w, self.h), (18, 28, 40))
        draw = ImageDraw.Draw(img)
        y = self.h // 3
        draw.text((40, y), title, font=self.big, fill=(255, 255, 255))
        for i, line in enumerate(lines):
            draw.text((40, y + self.big.size + 20 + i * (self.small.size + 8)), line, font=self.small, fill=(210, 220, 230))
        for _ in range(int(round(seconds * self.fps))):
            self.emit(img)

    def clip(self, runner: SimulationRunner, title: str, lines: list[str], overlay, duration: float | None = None) -> dict:
        """Simulate `runner` for `duration` (default: the whole task) and render it in real time."""
        e = runner.engine
        m = e.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, self.w), max(
            m.vis.global_.offheight, self.h)
        renderer = mujoco.Renderer(m, self.h, self.w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.args.lookat
        cam.distance, cam.azimuth, cam.elevation = self.args.distance, self.args.azimuth, self.args.elevation
        duration = runner.task.duration if duration is None else duration
        steps_per_frame = max(1, int(round(1.0 / (self.fps * e.dt))))
        contact_steps = 0
        targets = runner.waypoint_ee
        try:
            for _ in range(int(round(duration * self.fps))):
                renderer.update_scene(e.data, cam)
                overlay(renderer.scene)
                ref = runner.task.reference(e.time)
                goal = targets[ref.waypoint_index]
                err = 1e3 * np.linalg.norm(e.forward_kinematics(e.arm_qpos()) - goal)
                status = f"t = {e.time:4.1f} s   EE to target {err:6.1f} mm   contact steps {contact_steps}"
                self.emit(self.annotate(renderer.render(), title, lines, status))
                for _ in range(steps_per_frame):
                    runner.step()
                    if e.data.ncon and e.robot_contacts():
                        contact_steps += 1
        finally:
            renderer.close()
        return {"contact_steps": contact_steps, "final_error_mm": err}


def demo_config(planning: bool) -> dict:
    cfg = load_config(DEMO)
    cfg["task"]["planning"]["enabled"] = planning
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=f"Render the {VERSION} motion-planning demo video (offscreen, MuJoCo)")
    p.add_argument("--output", default=str(PROJECT_ROOT / "results" / "video" / f"{VERSION}_demo.webp"))
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--include-unsafe", action="store_true",
                   help="Add a clip with planning off AND the path/torque checks disabled (the arm hits the wall)")
    p.add_argument("--variant", default="taller", help="Wall variant from configs/robustness_wall.yaml for clip 4")
    p.add_argument("--seed", type=int, default=0, help="Planning seed for the planned clips")
    p.add_argument("--frames-dir", default=None, help="Also write every frame as PNG here")
    p.add_argument("--azimuth", type=float, default=170.0)
    p.add_argument("--elevation", type=float, default=-25.0)
    p.add_argument("--distance", type=float, default=1.55)
    p.add_argument("--lookat", type=float, nargs=3, default=[0.40, 0.0, 0.62])
    args = p.parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists (earlier videos are kept as evidence); choose another --output "
                         "or pass --overwrite")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    video = Video(args)
    legend = "green: planned path followed (end effector)   orange: raw RRT-Connect path   red: blocked / invalid path"

    exact = load_config(DEMO)["task"]["planning"].get("exact_box_box", True)
    video.card(f"Collision-aware motion planning in MuJoCo (Robotics Lab {VERSION})",
               ["Franka Panda, torque-actuated, torque MPC controller, real-time playback.",
                "Task: move the hand from the left of the wall to the right of it (0.45, -0.25, 0.50) m.",
                "Planner: joint-space RRT-Connect + shortcutting, 20 mm clearance, run once before motion.",
                "Collision checker: exact distances for the fingertip boxes (v0.5.2 fix)" if exact else
                "Collision checker: MuJoCo margin contacts only (v0.5 / v0.5.1 behaviour)"], 4.0)

    # 1. Planning on.
    cfg = demo_config(True)
    cfg["task"]["planning"]["seed"] = args.seed
    planned = SimulationRunner(cfg)
    first = planned.task.plans[0]
    st = first.stats
    nominal_path = first.path
    print(f"clip 1: {first.name} {first.status}, {1e3 * st['total_time_s']:.0f} ms, {len(first.path)} vertices")
    kin = ArmKinematics(planned.engine)
    start = np.asarray(cfg["task"]["start"], float)
    direct_line = ee_polyline(kin.fk, [start, first.q], 40)

    def overlay_planned(scene):
        add_polyline(scene, direct_line, BLOCKED_RGBA, 0.003)
        planned.draw_planned_paths(scene)

    r = video.clip(planned, "1. Planning ON: the direct joint move is blocked, the planner goes over the wall",
                   [f"Planned in {1e3 * st['total_time_s']:.0f} ms (seed {args.seed}): path {st['raw_length_rad']:.2f} -> "
                    f"{st['smoothed_length_rad']:.2f} rad after shortcutting, {len(first.path)} vertices; "
                    f"then a second target back on the left.", legend], overlay_planned)
    print(f"  contact steps {r['contact_steps']}, final error {r['final_error_mm']:.2f} mm")

    # 2. Planning off: the existing straight-path check rejects the blocked target.
    off = SimulationRunner(demo_config(False))
    rejected = [t for t in off.task.plans if not t.reachable]
    reason = rejected[0].reason if rejected else "(not rejected)"
    print(f"clip 2: rejected {[t.name for t in rejected]}: {reason}")
    video.card("2. Same targets, planning OFF",
               ["Without the planner, only the straight joint-space move is considered (v0.3/v0.4 behaviour).",
                f"It is collision-checked first: right_of_wall REJECTED ({reason}).",
                "The robot never attempts it; the red line shows the hand path it would have taken."], 3.0)
    video.clip(off, "2. Planning OFF: blocked target rejected, robot stays on the left",
               [f"right_of_wall rejected: {reason}. Only back_left (= start pose) remains.", legend],
               lambda scene: add_polyline(scene, direct_line, BLOCKED_RGBA, 0.004))

    # 3. Optional unsafe baseline.
    if args.include_unsafe:
        cfg = demo_config(False)
        cfg["task"]["ik"]["check_path"] = False
        cfg["task"]["trajectory"]["torque_scale"] = None
        unsafe = SimulationRunner(cfg)
        video.card("3. UNSAFE baseline: planning off and safety checks deliberately disabled",
                   ["Path collision check and torque-feasibility check turned off for comparison only.",
                    "The same controller follows the straight joint move into the wall."], 3.0)
        r = video.clip(unsafe, "3. UNSAFE baseline (checks disabled): the straight move hits the wall",
                       ["Comparison only. Contacts are real MuJoCo contacts between the arm and the wall.", legend],
                       lambda scene: add_polyline(scene, direct_line, BLOCKED_RGBA, 0.004),
                       duration=unsafe.task.hold_end_time(unsafe.task.plans[0]))
        print(f"clip 3: contact steps {r['contact_steps']}, final error {r['final_error_mm']:.1f} mm")

    # 4. New episode, changed wall: re-check the old plan, replan with the known new geometry.
    import yaml

    with open(ROBUSTNESS, encoding="utf-8") as f:
        rob = yaml.safe_load(f)
    variant = rob["variants"][args.variant]
    cfg = demo_config(True)
    cfg["task"]["planning"]["seed"] = args.seed
    cfg["task"]["targets"] = cfg["task"]["targets"][:1]
    cfg["environment_overrides"] = variant["overrides"]
    episode = SimulationRunner(cfg)
    ok, why = episode.planner.checker.path_valid(nominal_path)
    new = episode.task.plans[0]
    print(f"clip 4: old plan valid in '{args.variant}': {ok} ({why}); replanned {new.status}")
    video.card(f"4. New episode: wall changed ({variant['description']})",
               ["The new wall geometry is known at the start of the episode (no perception, no online replanning).",
                f"The previous plan is re-checked against it: {'still valid' if ok else 'INVALID - ' + why}.",
                "So the robot plans again with the updated scene before moving."], 4.0)
    old_line = ee_polyline(kin.fk, nominal_path, 20)

    def overlay_episode(scene):
        add_polyline(scene, old_line, BLOCKED_RGBA, 0.004)
        add_points(scene, np.array([kin.fk(q) for q in nominal_path[1:-1]]), BLOCKED_RGBA, 0.01)
        draw_planned_paths(scene, kin.fk, episode.task.planned_paths, show_raw=False)

    s = new.stats or {}
    r = video.clip(episode, f"4. Wall changed ({args.variant}): old path invalid (red), new plan (green)",
                   [f"Replanned in {1e3 * s.get('total_time_s', float('nan')):.0f} ms: "
                    f"{s.get('smoothed_length_rad', float('nan')):.2f} rad, {len(new.path) if new.path else 0} vertices. "
                    "Same planner and controller, no code change between episodes.", legend], overlay_episode)
    print(f"  contact steps {r['contact_steps']}, final error {r['final_error_mm']:.2f} mm")

    video.stream.close()
    secs = video.stream.frames / args.fps
    print(f"Saved {args.output}: {video.stream.frames} frames, {secs:.1f} s at {args.fps} fps, "
          f"{Path(args.output).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
