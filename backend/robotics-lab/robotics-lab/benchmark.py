"""Headless controller benchmark.

    python benchmark.py                                   # all controllers x all scenarios
    python benchmark.py --controllers pid mpc --scenarios demo fast
    python benchmark.py --output results/my_run --no-plots
    python benchmark.py --config configs/benchmark_robustness.yaml   # v0.4 mismatch / disturbance sweep

Scenarios and controller variants are defined in configs/benchmark.yaml.
Results go to results/benchmark_<timestamp>/ (results.json, results.csv, plots).
"""

import argparse
import math
import time
from pathlib import Path

from simulation.benchmark import BENCHMARK_CONFIG, run_benchmark
from simulation.engine import PROJECT_ROOT

TABLE_COLUMNS = [
    ("rms_error_rad", "RMS err", "{:.2e}"),
    ("max_error_rad", "max err", "{:.2e}"),
    ("ee_max_error_mm", "EE mm", "{:.3f}"),
    ("ee_orientation_max_error_deg", "ori deg", "{:.3f}"),
    ("reference_feasible", "ref ok", "{}"),
    ("max_velocity_ratio", "v/vmax", "{:.4f}"),
    ("mpc_max_predicted_velocity_ratio", "pred v", "{:.4f}"),
    ("settling_time_mean_s", "settle", "{:.2f}"),
    ("rms_torque_Nm", "RMS tau", "{:.2f}"),
    ("torque_saturation_fraction", "sat", "{:.2f}"),
    ("velocity_limit_violation_steps", "v-viol", "{}"),
    ("joint_limit_violation_steps", "q-viol", "{}"),
    ("mpc_solve_time_mean_ms", "solve ms", "{:.2f}"),
    ("mpc_solve_time_p95_ms", "p95 ms", "{:.2f}"),
    ("deadline_overruns", "overrun", "{}"),
    ("mpc_fallbacks", "fallbk", "{}"),
    ("wall_time_s", "wall s", "{:.1f}"),
]


def fmt(value, spec):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return spec.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controllers headlessly on identical scenarios")
    parser.add_argument("--controllers", nargs="+", default=None, help="Subset of controllers from the benchmark config")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Subset of scenarios from the benchmark config")
    parser.add_argument("--config", default=str(BENCHMARK_CONFIG), help="Benchmark YAML")
    parser.add_argument("--output", default=None, help="Output directory (default results/benchmark_<timestamp>)")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots")
    args = parser.parse_args()

    out = Path(args.output) if args.output else PROJECT_ROOT / "results" / time.strftime("benchmark_%Y%m%d_%H%M%S")
    print(f"Running benchmark -> {out}")
    report = run_benchmark(args.controllers, args.scenarios, out, plots=not args.no_plots, bench_path=Path(args.config))

    header = f"{'scenario':13s} {'controller':11s} {'actuator':8s} " + " ".join(f"{h:>8s}" for _, h, _ in TABLE_COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for r in report["results"]:
        row = " ".join(f"{fmt(r.get(k), spec):>8s}" for k, _, spec in TABLE_COLUMNS)
        print(f"{r['scenario']:13s} {r['controller']:11s} {r['actuation']:8s} {row}")
    print("\nUnits: joint errors rad, EE = max end-effector path error (mm), v/vmax = peak joint speed / limit,")
    print("settle s (mean over moves), tau Nm; sat = fraction of physics steps with an arm actuator at its force")
    print("limit; v-viol/q-viol = physics steps above the speed limit / outside the joint range; ori deg = max")
    print("orientation tracking error (joint tasks only); ref ok = reference within speed/accel/torque limits;")
    print("pred v = max |qd|/v_max the MPC plans predicted.")
    print("Position-servo and torque-motor controllers use different actuator models (see configs/benchmark.yaml).")
    print(f"Saved results.json, results.csv{' and plots' if report.get('plots') else ''} to {out}")


if __name__ == "__main__":
    main()
