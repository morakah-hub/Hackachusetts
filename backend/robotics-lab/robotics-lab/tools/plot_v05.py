"""Figures for docs/research/v0.5-results.md: the planned obstacle-avoidance demo.

    python tools/plot_v05.py [output_dir]      # default docs/research/figures

Runs the demo (configs/planning_demo.yaml) headless with torque MPC, planned and
with the unchecked direct move, and plots
  v0.5_demo_paths.png      end-effector paths in the y-z plane against the wall
                           (direct joint interpolation, raw RRT-Connect, shortcut, executed)
  v0.5_demo_clearance.png  robot-environment distance over time, planned vs unchecked direct
"""

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker  # noqa: E402
from planning.visualization import ee_polyline  # noqa: E402
from simulation.engine import PROJECT_ROOT  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

DEMO = PROJECT_ROOT / "configs" / "planning_demo.yaml"


def run(cfg):
    runner = SimulationRunner(cfg)
    log = runner.run_headless(runner.task.duration, log_every=5).arrays()
    return runner, log


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "docs" / "research" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(DEMO)
    planned, log = run(base)
    cfg = copy.deepcopy(base)
    cfg["task"]["planning"]["enabled"] = False
    cfg["task"]["ik"]["check_path"] = False
    cfg["task"]["trajectory"]["torque_scale"] = None
    direct, log_d = run(cfg)

    fk = planned.planner.kin.fk
    first = planned.task.plans[0]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.add_patch(plt.Rectangle((-0.02, 0.40), 0.04, 0.26, color="#2a6fdb", alpha=0.5, label="wall (x 0.25-0.80)"))
    ax.add_patch(plt.Rectangle((-0.5, 0.36), 1.0, 0.04, color="#8b5a2b", alpha=0.5, label="table top"))
    start = np.asarray(base["task"]["start"], float)
    straight = ee_polyline(fk, [start, first.q], 60)
    raw = ee_polyline(fk, first.raw_path, 20)
    path = ee_polyline(fk, first.path, 20)
    end = int(np.searchsorted(log["time"], planned.task.hold_end_time(first)))
    ee = np.array([planned.engine.forward_kinematics(q) for q in log["q"][:end]])
    ax.plot(straight[:, 1], straight[:, 2], "--", color="0.4", lw=1.2, label="direct joint move (blocked)")
    ax.plot(raw[:, 1], raw[:, 2], color="#e0663d", lw=1.2, alpha=0.8, label=f"RRT-Connect raw ({len(first.raw_path)} vertices)")
    ax.plot(path[:, 1], path[:, 2], color="#2e9e6a", lw=2.4, alpha=0.6, label=f"shortcut ({len(first.path)} vertices)")
    ax.plot(ee[:, 1], ee[:, 2], color="k", lw=0.8, label="executed (torque MPC)")
    via = np.array([fk(q) for q in first.path])
    ax.plot(via[:, 1], via[:, 2], "o", color="#2e9e6a", ms=5)
    ax.set_xlabel("y (m)  [end-effector, side view]")
    ax.set_ylabel("z (m)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Start (y = +0.25) to right_of_wall (y = -0.25), seed 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "v0.5_demo_paths.png", dpi=130)
    plt.close(fig)

    mon = CollisionChecker.from_engine(planned.engine, clearance=0.10, limit_margin=0.0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for lg, label, color in ((log, "planned (torque MPC)", "#2e9e6a"), (log_d, "unchecked direct (torque MPC)", "#e0663d")):
        d = np.array([mon.min_distance(q) for q in lg["q"]])
        ax.plot(lg["time"], 1e3 * d, color=color, label=label)
    s = planned.planner.settings
    ax.axhline(1e3 * s.clearance, color="0.3", ls="--", lw=0.8, label="planning clearance 20 mm")
    ax.axhline(1e3 * (s.clearance - s.cartesian_step / 2), color="0.3", ls=":", lw=0.8, label="certified 15 mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("robot-environment\ndistance (mm, capped 100)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "v0.5_demo_clearance.png", dpi=130)
    plt.close(fig)
    print(f"Saved v0.5_demo_paths.png and v0.5_demo_clearance.png to {out}")


if __name__ == "__main__":
    main()
