"""Wall-geometry robustness experiment (v0.5.1).

    python robustness_experiment.py                                   # 6 wall variants x 5 seeds, torque MPC + CTC
    python robustness_experiment.py --variants nominal taller --seeds 0 1
    python robustness_experiment.py --no-execution                    # planning and old-plan checks only
    python robustness_experiment.py --controllers torque_mpc --output results/rob --no-plots

Settings in configs/robustness_wall.yaml. The wall geometry of each episode is
known at its start (no perception, no replanning during motion). Results go to
results/robustness_<timestamp>/ (results.json, planning_runs.csv, execution_runs.csv,
summary.csv, summary.md, robustness_paths.png). Runs are sequential.
"""

import argparse
import time
from pathlib import Path

from simulation.engine import PROJECT_ROOT
from simulation.robustness import ROBUSTNESS_CONFIG, run_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Replanning under changed (known) wall geometry")
    parser.add_argument("--config", default=str(ROBUSTNESS_CONFIG), help="Experiment YAML")
    parser.add_argument("--variants", nargs="+", default=None, help="Subset of wall variants")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Planning seeds (default from config)")
    parser.add_argument("--controllers", nargs="+", default=None, help="Subset of execution controllers")
    parser.add_argument("--no-execution", action="store_true", help="Skip execution in MuJoCo")
    parser.add_argument("--output", default=None, help="Output directory (default results/robustness_<timestamp>)")
    parser.add_argument("--no-plots", action="store_true", help="Skip the PNG plot")
    args = parser.parse_args()

    out = Path(args.output) if args.output else PROJECT_ROOT / "results" / time.strftime("robustness_%Y%m%d_%H%M%S")
    print(f"Running wall robustness experiment -> {out}")
    run_robustness(Path(args.config), out, args.variants, args.seeds, args.controllers,
                   execution=not args.no_execution, plots=not args.no_plots)
    print()
    print((out / "summary.md").read_text(encoding="utf-8"))
    print("Old plan = the same seed's plan for the reference wall, re-checked (never executed) in the new geometry;")
    print("negative distance = penetration. Times include IK, RRT-Connect and shortcutting. Contact steps = physics")
    print("steps with any robot contact in the plant. Clearance measured up to 100 mm.")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
