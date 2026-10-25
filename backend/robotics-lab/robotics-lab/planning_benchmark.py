"""Motion-planning benchmark (v0.5).

    python planning_benchmark.py                                  # all scenarios, 20 seeds, + execution
    python planning_benchmark.py --scenarios wall narrow_passage --seeds 5
    python planning_benchmark.py --no-execution                   # planning statistics only
    python planning_benchmark.py --execution-only --scenarios wall  # closed-loop execution only
    python planning_benchmark.py --output results/my_run --no-plots

Scenarios are defined in configs/benchmark_planning.yaml. Results go to
results/planning_<timestamp>/ (results.json, planning_runs.csv, planning_summary.csv,
execution_runs.csv, summary.md, plots).
"""

import argparse
import time
from pathlib import Path

from simulation.engine import PROJECT_ROOT
from simulation.planning_benchmark import PLANNING_BENCHMARK_CONFIG, run_planning_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark collision-aware motion planning")
    parser.add_argument("--config", default=str(PLANNING_BENCHMARK_CONFIG), help="Benchmark YAML")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Subset of scenarios")
    parser.add_argument("--seeds", type=int, default=None, help="Planning seeds per scenario (default from config)")
    part = parser.add_mutually_exclusive_group()
    part.add_argument("--no-execution", action="store_true", help="Skip the closed-loop execution part")
    part.add_argument("--execution-only", action="store_true", help="Skip the planning statistics part")
    parser.add_argument("--output", default=None, help="Output directory (default results/planning_<timestamp>)")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    args = parser.parse_args()

    out = Path(args.output) if args.output else PROJECT_ROOT / "results" / time.strftime("planning_%Y%m%d_%H%M%S")
    print(f"Running planning benchmark -> {out}")
    report = run_planning_benchmark(Path(args.config), out, args.scenarios, args.seeds,
                                    execution=not args.no_execution, plots=not args.no_plots,
                                    planning=not args.execution_only)
    print()
    print((out / "summary.md").read_text(encoding="utf-8"))
    print("Times include IK, all RRT-Connect attempts and shortcutting. Clearance = smallest robot-environment")
    print("distance, measured up to 100 mm. Contact steps = physics steps with any robot contact in the plant.")
    print(f"Saved results.json, CSV files, summary.md{' and plots' if report.get('plots') else ''} to {out}")


if __name__ == "__main__":
    main()
