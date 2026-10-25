"""Motion-planning benchmark (v0.5): planning statistics over seeds, then closed-loop execution.

Run via planning_benchmark.py at the repo root; scenarios in configs/benchmark_planning.yaml.

Part 1 (planning only) plans every scenario with every seed and records the
outcome of every run, failures included. Each successful path is re-validated
with an independently built checker (always the corrected v0.5.2 checker, also
when the planner runs with `exact_box_box: false`) and its clearance is measured:
`path_min_clearance_mm` with the checker's distance (exact for box-box pairs
since v0.5.2), `path_min_exact_clearance_*` with mj_geomDistance over every
robot-environment pair, at the planner's own edge samples and densely (every
1 mm of robot motion) in between.

Part 2 (execution) builds the normal SimulationRunner for each executable
scenario (planning happens there, at task creation, exactly as in main.py),
runs MuJoCo headless and measures the simulated robot: tracking, final
end-effector error, limit violations, contacts in the plant, clearance and
controller health. Planning failures and execution failures are reported
separately (`outcome`).
"""

import copy
import csv
import json
import platform
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import yaml

from controllers.mpc import JointMPCController
from controllers.torque_mpc import TorqueMPCController
from planning.collision import CollisionChecker
from planning.motion_planner import MotionPlanner, PlanningSettings
from simulation.benchmark import deep_merge
from simulation.engine import PROJECT_ROOT, SimulationEngine
from simulation.kinematics import ArmKinematics
from simulation.runner import SimulationRunner, load_config

PLANNING_BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark_planning.yaml"


def load_bench(path: Path = PLANNING_BENCHMARK_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def scenario_planning(bench: dict, sc: dict) -> dict:
    return deep_merge(bench["planning"], sc.get("planning") or {})


def _percentile(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


# ------------------------------------------------------------------ part 1: planning only


def path_clearance(checker: CollisionChecker, path: list[np.ndarray]) -> float:
    """Smallest robot-environment distance at every edge sample of the path (capped)."""
    best = checker.clearance
    for a, b in zip(path[:-1], path[1:]):
        n = checker.edge_samples(a, b)
        for i in range(n + 1):
            best = min(best, checker.min_distance(a + (b - a) * (i / n)))
    return best


def plan_case(engine: SimulationEngine, sc_name: str, sc: dict, settings: dict, seed: int) -> dict:
    planner = MotionPlanner.from_engine(engine, PlanningSettings.from_config(settings))
    start = np.asarray(sc["start"], float)
    if "target" in sc:
        plan = planner.plan_to_position(np.asarray(sc["target"], float), start, seed=seed)
    else:
        plan = planner.plan_joint(start, np.asarray(sc["goal"], float), seed=seed)
    st = plan.stats
    row = {
        "scenario": sc_name, "seed": seed, "environment": sc["environment"],
        "goal_type": "cartesian" if "target" in sc else "joint",
        "success": plan.success, "status": plan.status, "reason": plan.reason,
        "total_time_s": st.get("total_time_s", float("nan")),
        "rrt_time_s": st.get("planning_time_s", float("nan")),
        "nodes": st.get("nodes_all_attempts", st.get("nodes", 0)),
        "collision_checks": st.get("collision_checks", 0),
        "direct_edge_valid": st.get("direct_edge_valid"),
        "ik_converged": st.get("ik_converged"), "ik_valid": st.get("ik_valid"),
        "candidate_index": st.get("candidate_index"), "candidates_tried": st.get("candidates_tried"),
        "raw_length_rad": st.get("raw_length_rad", float("nan")) if plan.success else float("nan"),
        "smoothed_length_rad": st.get("smoothed_length_rad", float("nan")) if plan.success else float("nan"),
        "raw_vertices": st.get("raw_vertices"), "smoothed_vertices": st.get("smoothed_vertices"),
    }
    if plan.success:
        # Independent re-check with a freshly built checker, and the path's true clearance.
        from simulation.robustness import ExactClearance  # (robustness imports this module)

        fresh = CollisionChecker.from_engine(engine, clearance=settings["clearance"],
                                             cartesian_step=settings["cartesian_step"],
                                             max_joint_step=settings["max_joint_step"],
                                             limit_margin=settings["joint_limit_margin"], exact_box_box=True)
        ok, why = fresh.path_valid(plan.path)
        monitor = CollisionChecker.from_engine(engine, clearance=0.10, cartesian_step=settings["cartesian_step"] / 4,
                                               limit_margin=0.0)
        exact = ExactClearance(engine)
        row |= {"path_revalidated": ok, "revalidation_reason": why,
                "path_min_clearance_mm": 1e3 * path_clearance(monitor, plan.path),
                "path_min_exact_clearance_samples_mm": 1e3 * exact.at_checked_samples(plan.path, planner.checker),
                "path_min_exact_clearance_mm": 1e3 * exact.path(plan.path, step=0.001)}
        if "target" in sc:
            row["goal_ee_error_mm"] = 1e3 * float(np.linalg.norm(planner.kin.fk(plan.q) - np.asarray(sc["target"])))
    return row


def ik_only_baseline(engine: SimulationEngine, sc: dict) -> str:
    """What the v0.3/v0.4 IK + straight-path check does with this scenario."""
    if "target" not in sc:
        return "n/a (joint goal)"
    kin = ArmKinematics(engine)
    res = kin.plan_target(np.asarray(sc["target"], float), np.asarray(sc["start"], float))
    return "accepted" if res.success else f"rejected: {res.reason}"


def summarize(rows: list[dict], baselines: dict) -> list[dict]:
    out = []
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == sc]
        ok = [r for r in rs if r["success"]]
        t_all = [r["total_time_s"] for r in rs]
        t_ok = [r["total_time_s"] for r in ok]
        raw = [r["raw_length_rad"] for r in ok]
        smooth = [r["smoothed_length_rad"] for r in ok]
        out.append({
            "scenario": sc, "runs": len(rs), "successes": len(ok), "success_rate": len(ok) / len(rs),
            "statuses": dict(Counter(r["status"] for r in rs)),
            "direct_edge_blocked": sum(r["direct_edge_valid"] is False for r in rs),
            "time_mean_s": float(np.mean(t_all)), "time_median_s": float(np.median(t_all)),
            "time_p95_s": _percentile(t_all, 95),
            "time_success_median_s": float(np.median(t_ok)) if ok else float("nan"),
            "nodes_mean": float(np.mean([r["nodes"] for r in rs])),
            "collision_checks_mean": float(np.mean([r["collision_checks"] for r in rs])),
            "raw_length_mean_rad": float(np.mean(raw)) if ok else float("nan"),
            "smoothed_length_mean_rad": float(np.mean(smooth)) if ok else float("nan"),
            "length_reduction_mean": float(np.mean([1 - s / r for r, s in zip(raw, smooth) if r > 0])) if ok else float("nan"),
            "smoothed_vertices_mean": float(np.mean([r["smoothed_vertices"] for r in ok])) if ok else float("nan"),
            "ik_valid_mean": float(np.mean([r["ik_valid"] for r in rs if r["ik_valid"] is not None]))
            if any(r["ik_valid"] is not None for r in rs) else float("nan"),
            "revalidated": sum(bool(r.get("path_revalidated")) for r in ok),
            "path_min_clearance_mm": float(min(r["path_min_clearance_mm"] for r in ok)) if ok else float("nan"),
            "path_min_exact_clearance_samples_mm": float(min(r["path_min_exact_clearance_samples_mm"] for r in ok))
            if ok else float("nan"),
            "path_min_exact_clearance_mm": float(min(r["path_min_exact_clearance_mm"] for r in ok)) if ok else float("nan"),
            "v04_ik_only": baselines.get(sc, ""),
        })
    return out


# ------------------------------------------------------------------ part 2: execution


def execution_config(base: dict, bench: dict, sc: dict, controller: dict, seed: int, mode: str) -> dict:
    ex = bench["execution"]
    cfg = copy.deepcopy(base)
    cfg["environment"] = sc["environment"]
    cfg["controller"] = deep_merge(base["controller"], controller)
    planning = {**scenario_planning(bench, sc), "enabled": mode == "planned", "seed": seed}
    cfg["task"] = {
        "type": "cartesian_reach", "move_time": 3.0, "hold_time": ex["hold_time"],
        "trajectory": copy.deepcopy(ex["trajectory"]), "start": list(sc["start"]),
        "targets": [{"name": "goal", "position": list(sc["target"])}],
        "ik": {"tolerance": planning.get("ik", {}).get("tolerance", 1e-4)}, "planning": planning,
    }
    if mode == "unchecked_direct":  # explicit unsafe baseline: no path check, no torque check
        cfg["task"]["ik"]["check_path"] = False
        cfg["task"]["trajectory"]["torque_scale"] = None
    return cfg


def execute_case(cfg: dict, bench: dict, sc_name: str, ctrl_name: str, seed: int, mode: str,
                 measure=None) -> dict:
    """`measure(engine, q_samples) -> dict`, if given, adds metrics computed on the logged joints."""
    ex = bench["execution"]
    t_build = time.perf_counter()
    runner = SimulationRunner(cfg)
    build_s = time.perf_counter() - t_build
    task, engine = runner.task, runner.engine
    p = task.plans[0]
    row = {"scenario": sc_name, "controller": ctrl_name, "seed": seed, "mode": mode,
           "actuation": engine.robot.actuation, "plan_success": p.reachable, "plan_status": p.status or
           ("success" if p.reachable else "rejected"), "plan_reason": p.reason, "task_build_s": build_s}
    st = p.stats or {}
    row |= {"planning_time_s": st.get("total_time_s", float("nan")),
            "raw_length_rad": st.get("raw_length_rad", float("nan")),
            "smoothed_length_rad": st.get("smoothed_length_rad", float("nan")),
            "path_vertices": len(p.path) if p.path is not None else (2 if p.reachable else 0)}
    if not p.reachable:
        return row | {"outcome": "planning_failed"}

    contact_steps, events, pairs, prev = 0, 0, set(), False

    def before_step(r):
        nonlocal contact_steps, events, prev
        found = r.engine.robot_contacts() if r.engine.data.ncon else []
        if found:
            contact_steps += 1
            pairs.update(found)
            events += not prev
        prev = bool(found)

    wall = time.perf_counter()
    log = runner.run_headless(task.duration, log_every=1, before_step=before_step).arrays()
    wall = time.perf_counter() - wall
    t, q, q_ref, qd = log["time"], log["q"], log["q_ref"], log["qd"]
    err = np.abs(q - q_ref)
    target = np.asarray(cfg["task"]["targets"][0]["position"], float)
    idx = np.arange(0, len(t), 5)
    ee = np.array([engine.forward_kinematics(q[i]) for i in idx])
    d_goal = np.linalg.norm(ee - target, axis=1)
    far = np.nonzero(d_goal > ex["reach_tolerance"])[0]
    reach_t = float("nan") if (len(far) and far[-1] == len(idx) - 1) else float(t[idx[far[-1] + 1]] if len(far) else t[0])
    monitor = CollisionChecker.from_engine(engine, clearance=ex["clearance_monitor"], limit_margin=0.0)
    clearance = min(monitor.min_distance(q[i]) for i in idx)
    lo, hi = engine.joint_limits[:, 0], engine.joint_limits[:, 1]
    vmax = np.asarray(engine.robot.velocity_limits)
    i_end = np.searchsorted(t, task.hold_end_time(p)) - 1
    final = float(np.linalg.norm(engine.forward_kinematics(q[i_end]) - target))
    row |= {
        "planned_motion_s": task.segment_bounds(task.num_segments - 1)[1],
        "executed_reach_s": reach_t,
        "tracking_rms_rad": float(np.sqrt(np.mean(err**2))), "tracking_max_rad": float(err.max()),
        "final_ee_error_mm": 1e3 * final,
        "joint_limit_violation_steps": int(np.sum(np.any((q < lo) | (q > hi), axis=1))),
        "velocity_violation_steps": int(np.sum(np.any(np.abs(qd) > vmax, axis=1))),
        "max_velocity_ratio": float(np.max(np.abs(qd) / vmax)),
        "min_clearance_mm": 1e3 * clearance,
        "contact_steps": contact_steps, "contact_events": events,
        "contact_pairs": ";".join(sorted(f"{a}-{b}" for a, b in pairs)),
        "sim_time_s": float(engine.time), "wall_time_s": wall,
    }
    ctrl = runner.controller
    if isinstance(ctrl, (JointMPCController, TorqueMPCController)):
        s = ctrl.stats.summary(ctrl.control_period)
        row |= {"mpc_fallbacks": s["fallbacks"], "deadline_overruns": s["deadline_overruns"],
                "mpc_solve_time_p95_ms": s["solve_time_p95_ms"], "mpc_update_time_max_ms": s["update_time_max_ms"]}
    else:
        row |= {"mpc_fallbacks": 0, "deadline_overruns": 0}
    ok = final <= ex["reach_tolerance"] and contact_steps == 0
    row["outcome"] = "success" if ok else "execution_failed"
    if measure is not None:
        row |= measure(engine, q[idx])
    return row


# ------------------------------------------------------------------ driver


def run_planning_benchmark(bench_path: Path = PLANNING_BENCHMARK_CONFIG, out_dir: Path | None = None,
                           scenarios: list[str] | None = None, seeds: int | None = None,
                           execution: bool = True, plots: bool = True, verbose: bool = True,
                           planning: bool = True) -> dict:
    bench = load_bench(bench_path)
    base = load_config()
    names = scenarios or list(bench["scenarios"])
    n_seeds = int(seeds if seeds is not None else bench["seeds"])
    engines: dict[str, SimulationEngine] = {}

    rows, baselines = [], {}
    for name in (names if planning else []):
        sc = bench["scenarios"][name]
        eng = engines.setdefault(sc["environment"], SimulationEngine(sc["environment"]))
        baselines[name] = ik_only_baseline(eng, sc)
        settings = scenario_planning(bench, sc)
        for seed in range(n_seeds):
            rows.append(plan_case(eng, name, sc, settings, seed))
        if verbose:
            ok = sum(r["success"] for r in rows if r["scenario"] == name)
            st = Counter(r["status"] for r in rows if r["scenario"] == name)
            print(f"  plan {name:24s} {ok}/{n_seeds} success  {dict(st)}", flush=True)
    summary = summarize(rows, baselines)

    exec_rows = []
    if execution:
        ex = bench["execution"]
        for name in names:
            sc = bench["scenarios"][name]
            if not sc.get("execute"):
                continue
            for c_name, c_cfg in ex["controllers"].items():
                runs = [(s, "planned") for s in ex["seeds"]]
                if sc.get("unchecked_baseline"):
                    runs.append((0, "unchecked_direct"))
                for seed, mode in runs:
                    cfg = execution_config(base, bench, sc, c_cfg, seed, mode)
                    r = execute_case(cfg, bench, name, c_name, seed, mode)
                    exec_rows.append(r)
                    if verbose:
                        print(f"  exec {name:18s} {c_name:10s} seed {seed} {mode:16s} -> {r['outcome']:16s} "
                              f"final {r.get('final_ee_error_mm', float('nan')):8.3f} mm  contacts "
                              f"{r.get('contact_steps', '-')}  clearance {r.get('min_clearance_mm', float('nan')):6.1f} mm",
                              flush=True)

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                     "platform": platform.platform(), "processor": platform.processor()},
        "benchmark_config": bench, "planning_runs": rows, "planning_summary": summary, "execution_runs": exec_rows,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        for fname, data in (("planning_runs.csv", rows), ("planning_summary.csv", summary),
                            ("execution_runs.csv", exec_rows)):
            if data:
                keys = list(dict.fromkeys(k for r in data for k in r))
                with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(data)
        (out_dir / "summary.md").write_text(markdown_tables(summary, exec_rows), encoding="utf-8")
        if plots and rows:
            report["plots"] = save_plots(out_dir, rows, names)
    return report


def _f(x, spec="{:.3f}"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return spec.format(x)


def markdown_tables(summary: list[dict], exec_rows: list[dict]) -> str:
    lines = ["## Planning (all seeds)", "",
             "| scenario | success | statuses | direct blocked | time mean / median / p95 (s) | nodes | checks | "
             "raw -> smoothed (rad) | vertices | min clearance (mm) | exact: samples / dense (mm) | v0.4 IK-only |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        st = ", ".join(f"{k} {v}" for k, v in s["statuses"].items())
        lines.append(
            f"| {s['scenario']} | {s['successes']}/{s['runs']} | {st} | {s['direct_edge_blocked']}/{s['runs']} | "
            f"{_f(s['time_mean_s'])} / {_f(s['time_median_s'])} / {_f(s['time_p95_s'])} | {_f(s['nodes_mean'], '{:.0f}')} | "
            f"{_f(s['collision_checks_mean'], '{:.0f}')} | {_f(s['raw_length_mean_rad'], '{:.2f}')} -> "
            f"{_f(s['smoothed_length_mean_rad'], '{:.2f}')} | {_f(s['smoothed_vertices_mean'], '{:.1f}')} | "
            f"{_f(s['path_min_clearance_mm'], '{:.1f}')} | {_f(s['path_min_exact_clearance_samples_mm'], '{:.1f}')} / "
            f"{_f(s['path_min_exact_clearance_mm'], '{:.1f}')} | {s['v04_ik_only']} |")
    if exec_rows:
        lines += ["", "## Execution in MuJoCo", "",
                  "| scenario | controller | seed | mode | outcome | planned / reached (s) | RMS / max err (rad) | "
                  "final EE (mm) | contact steps | min clearance (mm) | q / v viol | fallbacks | overruns |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in exec_rows:
            if r["outcome"] == "planning_failed":
                lines.append(f"| {r['scenario']} | {r['controller']} | {r['seed']} | {r['mode']} | planning_failed "
                             f"({r['plan_status']}) | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {r['scenario']} | {r['controller']} | {r['seed']} | {r['mode']} | {r['outcome']} | "
                f"{_f(r['planned_motion_s'], '{:.2f}')} / {_f(r['executed_reach_s'], '{:.2f}')} | "
                f"{_f(r['tracking_rms_rad'], '{:.1e}')} / {_f(r['tracking_max_rad'], '{:.1e}')} | "
                f"{_f(r['final_ee_error_mm'], '{:.3f}')} | {r['contact_steps']} | {_f(r['min_clearance_mm'], '{:.1f}')} | "
                f"{r['joint_limit_violation_steps']} / {r['velocity_violation_steps']} | {r['mpc_fallbacks']} | "
                f"{r['deadline_overruns']} |")
    return "\n".join(lines) + "\n"


def save_plots(out_dir: Path, rows: list[dict], names: list[str]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [[r["total_time_s"] for r in rows if r["scenario"] == n] for n in names]
    ax.boxplot(data, tick_labels=names, showfliers=True)
    for i, n in enumerate(names):
        fails = [r["total_time_s"] for r in rows if r["scenario"] == n and not r["success"]]
        ax.plot([i + 1] * len(fails), fails, "x", color="#e0663d", ms=5)
    ax.set_yscale("log")
    ax.set_ylabel("planning time (s), x = failed run")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "planning_time.png", dpi=120)
    plt.close(fig)
    files.append("planning_time.png")

    with_paths = [n for n in names if any(r["success"] and r["raw_vertices"] for r in rows if r["scenario"] == n)]
    if with_paths:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(with_paths))
        raw = [np.mean([r["raw_length_rad"] for r in rows if r["scenario"] == n and r["success"]]) for n in with_paths]
        smo = [np.mean([r["smoothed_length_rad"] for r in rows if r["scenario"] == n and r["success"]]) for n in with_paths]
        ax.bar(x - 0.2, raw, 0.4, label="RRT-Connect (raw)", color="#e0663d")
        ax.bar(x + 0.2, smo, 0.4, label="shortcut", color="#2e9e6a")
        ax.set_xticks(x, with_paths, rotation=20, fontsize=8)
        ax.set_ylabel("mean joint-space path length (rad)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "path_length.png", dpi=120)
        plt.close(fig)
        files.append("path_length.png")
    return files
