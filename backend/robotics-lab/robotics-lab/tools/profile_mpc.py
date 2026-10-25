"""Profile MPC updates (servo MPC and torque MPC) at 50 and 100 Hz.

    python tools/profile_mpc.py                       # default matrix, demo motion, 6 s each
    python tools/profile_mpc.py --duration 12 --scenario wrist_feasible
    python tools/profile_mpc.py --cprofile            # also print the top cProfile entries
    python tools/profile_mpc.py --memory              # tracemalloc: allocation hotspots per update

For each variant it reports per-update wall time split into linearization
(mjd_transitionFD + composition + one rollout for c), gravity torque, OSQP
update (new QP data), OSQP solve (incl. warm start), and the rest (QP data
assembly in numpy, bookkeeping), plus total mean / p95 / p99 / max and
deadline misses (update time > control period). Results also go to
results/v0.4_profile/profile.json. Timing is wall-clock and machine/load
dependent: compare variants from the same run.
"""

import argparse
import cProfile
import json
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import osqp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.benchmark import BENCHMARK_CONFIG, deep_merge, scenario_config  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

VARIANTS = {
    "servo_mpc_50hz": ("joint_mpc", {}),
    "servo_mpc_100hz": ("joint_mpc", {"control_rate_hz": 100, "horizon": 15, "solver": {"max_iter": 250}}),
    "torque_mpc_50hz": ("torque_mpc", {}),
    "torque_mpc_100hz_N25": ("torque_mpc", {"control_rate_hz": 100, "horizon": 25}),
    "torque_mpc_100hz_N50": ("torque_mpc", {"control_rate_hz": 100, "horizon": 50}),
}


class Timer:
    """Wraps a callable and accumulates its wall time per MPC update."""

    def __init__(self):
        self.total = 0.0

    def wrap(self, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.total += time.perf_counter() - t0
        return inner


def profile_variant(name, kind, overrides, scenario, duration, use_cprofile=False, memory=False):
    with open(BENCHMARK_CONFIG, encoding="utf-8") as f:
        bench = yaml.safe_load(f)
    cfg = scenario_config(load_config(), bench, "torque_mpc" if kind == "torque_mpc" else "mpc", scenario)
    section = "torque_mpc" if kind == "torque_mpc" else "mpc"
    cfg["controller"][section] = deep_merge(cfg["controller"].get(section, {}), overrides)
    r = SimulationRunner(cfg)
    c = r.controller
    timers = {k: Timer() for k in ("linearize", "gravity", "osqp_update", "osqp_solve", "observer")}
    c.model.linearize = timers["linearize"].wrap(c.model.linearize)
    c.model.gravity_torque = timers["gravity"].wrap(c.model.gravity_torque)
    if getattr(c, "observer", None) is not None:
        c.observer.update = timers["observer"].wrap(c.observer.update)
    orig_update, orig_solve, orig_warm = osqp.OSQP.update, osqp.OSQP.solve, osqp.OSQP.warm_start
    osqp.OSQP.update = timers["osqp_update"].wrap(orig_update)
    osqp.OSQP.solve = timers["osqp_solve"].wrap(orig_solve)
    osqp.OSQP.warm_start = timers["osqp_solve"].wrap(orig_warm)
    per_update = {k: [] for k in timers}
    orig_mpc_update = c._update

    def update(*a, **k):
        before = {k2: t.total for k2, t in timers.items() if k2 != "observer"}
        orig_mpc_update(*a, **k)
        for k2, v in before.items():
            per_update[k2].append(timers[k2].total - v)

    c._update = update
    # Machine-speed calibration: mean mj_step time on a scratch copy right before the run, so
    # update costs can also be read in "physics steps" (robust to the 2-3x speed swings seen).
    import mujoco
    scratch = mujoco.MjData(r.engine.model)
    t0 = time.perf_counter()
    mujoco.mj_step(r.engine.model, scratch, 2000)
    step_us = 1e6 * (time.perf_counter() - t0) / 2000
    prof = cProfile.Profile() if use_cprofile else None
    if memory:
        tracemalloc.start(10)
        snap0 = tracemalloc.take_snapshot()
    if prof:
        prof.enable()
    wall = time.perf_counter()
    r.run_headless(duration, log_every=1000)
    wall = time.perf_counter() - wall
    if prof:
        prof.disable()
    mem = None
    if memory:
        snap1 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = snap1.compare_to(snap0, "lineno")
        n = len(c.stats.updates)
        mem = [(str(s.traceback[0]), s.count_diff, s.size_diff) for s in stats[:8]]
        print(f"  tracemalloc (net, top lines over {n} updates):")
        for where, cnt, size in mem:
            print(f"    {size / 1024:8.1f} KiB {cnt:7d} blocks  {where}")
    osqp.OSQP.update, osqp.OSQP.solve, osqp.OSQP.warm_start = orig_update, orig_solve, orig_warm

    upd = np.array([u.update_time for u in c.stats.updates])
    parts = {k: 1e3 * float(np.mean(v)) for k, v in per_update.items() if v}
    rest = 1e3 * upd.mean() - sum(parts.values())
    obs_steps = max(1, int(round(duration / r.engine.dt)))
    row = {
        "variant": name, "scenario": scenario, "rate_hz": 1 / c.control_period, "horizon": c.N,
        "updates": len(upd), "mean_ms": 1e3 * upd.mean(), "p95_ms": 1e3 * np.percentile(upd, 95),
        "p99_ms": 1e3 * np.percentile(upd, 99), "max_ms": 1e3 * upd.max(),
        "deadline_ms": 1e3 * c.control_period, "deadline_misses": int(np.sum(upd > c.control_period)),
        "linearize_ms": parts.get("linearize", 0.0), "gravity_ms": parts.get("gravity", 0.0),
        "osqp_update_ms": parts.get("osqp_update", 0.0), "osqp_solve_ms": parts.get("osqp_solve", 0.0),
        "rest_ms": rest, "observer_us_per_step": 1e6 * timers["observer"].total / obs_steps,
        "mean_iterations": float(np.mean([u.iterations for u in c.stats.updates])),
        "fallbacks": int(sum(u.fallback for u in c.stats.updates)),
        "realtime_factor": duration / wall,
        "mj_step_us": step_us, "mean_in_mj_steps": 1e6 * upd.mean() / step_us,
        "p99_in_mj_steps": 1e6 * np.percentile(upd, 99) / step_us,
    }
    print(f"{name:22s} {row['rate_hz']:4.0f} Hz N={row['horizon']:2d} | total mean {row['mean_ms']:5.2f} p95 {row['p95_ms']:5.2f}"
          f" p99 {row['p99_ms']:5.2f} max {row['max_ms']:6.2f} ms (deadline {row['deadline_ms']:.0f}, misses {row['deadline_misses']})"
          f" | lin {row['linearize_ms']:.2f} grav {row['gravity_ms']:.2f} osqp-upd {row['osqp_update_ms']:.2f}"
          f" osqp-solve {row['osqp_solve_ms']:.2f} rest {row['rest_ms']:.2f} ms | obs {row['observer_us_per_step']:.0f} us/step"
          f" | iters {row['mean_iterations']:.0f} fb {row['fallbacks']} | mj_step {step_us:.1f} us ->"
          f" mean {row['mean_in_mj_steps']:.0f} / p99 {row['p99_in_mj_steps']:.0f} steps", flush=True)
    if prof:
        pstats.Stats(prof).sort_stats("tottime").print_stats(12)
    row["memory_top"] = mem
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--scenario", default="demo")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--output", default="results/v0.4_profile")
    args = ap.parse_args()
    rows = [profile_variant(v, *VARIANTS[v], args.scenario, args.duration, args.cprofile, args.memory)
            for v in args.variants]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"profile_{args.scenario}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
