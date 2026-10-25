"""Wall-geometry robustness experiment (v0.5.1).

Run via robustness_experiment.py at the repo root; settings in configs/robustness_wall.yaml.

Each (variant, seed) is an episode whose wall geometry is known at the start
(MjSpec overrides of environments/planning_demo.xml). Per episode:

1. Old plan: the plan made with the same seed for the reference variant is
   re-checked against the new geometry with the new scene's checker, and its
   signed robot-environment distance is measured. It is never executed.
2. Replanning (planning only): a planner built from the new scene plans again;
   the path is re-validated with an independently built checker.
3. Execution: for each controller, the normal SimulationRunner (planning at task
   creation, exactly as in main.py) runs the episode in MuJoCo, measured by
   simulation.planning_benchmark.execute_case (contacts, clearance, final error).

Clearance is reported twice. `*_clearance_mm` is the checker's measure
(`CollisionChecker.min_distance` under a 100 mm margin): MuJoCo's `contact.dist`, and since
v0.5.2 `mj_geomDistance` for box-box pairs. `*_exact_clearance_mm` is an independent
measurement with `mj_geomDistance` (GJK) over every collidable robot/environment geom pair.
In v0.5.1 they differed because MuJoCo's box-box collision routine reports approximate
`contact.dist` for margin contacts (the Panda's fingertip pads are boxes): by up to ~5 mm
too small and ~1 mm too large in that experiment (docs/research/v0.5.1-demo-brief.md).

All episodes are run sequentially, one model at a time (memory-bounded).
"""

import csv
import json
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from planning.collision import CollisionChecker
from planning.motion_planner import MotionPlanner, PlanningSettings
from simulation.engine import PROJECT_ROOT, SimulationEngine
from simulation.planning_benchmark import execute_case, execution_config, path_clearance, plan_case
from simulation.runner import load_config

ROBUSTNESS_CONFIG = PROJECT_ROOT / "configs" / "robustness_wall.yaml"


class ExactClearance:
    """Robot-environment distance from mj_geomDistance (GJK) over every collidable pair that
    MuJoCo's filters allow (contype/conaffinity, minus the excluded base-table pair).
    Measurement only, on a private model copy; capped at `cap`."""

    def __init__(self, engine: SimulationEngine, cap: float = 0.10):
        self.checker = CollisionChecker.from_engine(engine, clearance=0.0, limit_margin=0.0)
        m = self.model = self.checker.model
        self.data, self.cap = self.checker.data, cap
        robot = self.checker._is_robot_geom
        live = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        excluded = {(int(m.exclude_signature[i]) >> 16, int(m.exclude_signature[i]) & 0xFFFF) for i in range(m.nexclude)}
        self.pairs = []
        for a in np.nonzero(robot & live)[0]:
            for b in np.nonzero(~robot & live)[0]:
                if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
                    continue
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                if (ba, bb) in excluded or (bb, ba) in excluded:
                    continue
                self.pairs.append((int(a), int(b)))
        self._fromto = np.zeros(6)

    def distance(self, q: np.ndarray) -> float:
        c = self.checker
        c.data.qpos[:] = c.base_qpos
        c.data.qpos[c.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return min(self.cap, min(mujoco.mj_geomDistance(self.model, self.data, a, b, self.cap, self._fromto)
                                 for a, b in self.pairs))

    def at_checked_samples(self, path: list[np.ndarray], checker: CollisionChecker) -> float:
        """Smallest distance over the configurations `checker` evaluates on the path's edges
        (vertices included)."""
        best = self.cap
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = checker.edge_samples(a, b)
            for i in range(n + 1):
                best = min(best, self.distance(a + (b - a) * (i / n)))
        return best

    def path(self, path: list[np.ndarray], step: float = 0.0025) -> float:
        """Smallest distance along the straight joint edges, sampled every `step` (m) of robot motion."""
        best = self.cap
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(1, int(np.ceil(self.checker.motion_bound(a, b) / step)))
            for i in range(n + 1):
                best = min(best, self.distance(a + (b - a) * (i / n)))
        return best

    def __call__(self, engine, q_samples: np.ndarray) -> dict:  # execute_case(measure=...)
        return {"min_exact_clearance_mm": 1e3 * min(self.distance(q) for q in q_samples)}


def load_experiment(path: Path = ROBUSTNESS_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def variant_engine(exp: dict, name: str) -> SimulationEngine:
    return SimulationEngine(exp["environment"], overrides=exp["variants"][name]["overrides"])


def scenario(exp: dict) -> dict:
    return {"environment": exp["environment"], "start": exp["start"], "target": exp["target"]}


def reference_plans(exp: dict, seeds: list[int]) -> dict[int, list[np.ndarray] | None]:
    """The "old" plans: one per seed, planned in the reference geometry."""
    planner = MotionPlanner.from_engine(variant_engine(exp, exp["reference_variant"]),
                                        PlanningSettings.from_config(exp["planning"]))
    start, target = np.asarray(exp["start"], float), np.asarray(exp["target"], float)
    plans = {}
    for seed in seeds:
        plan = planner.plan_to_position(target, start, seed=seed)
        plans[seed] = plan.path if plan.success else None
    return plans


def wall_box(engine: SimulationEngine) -> dict:
    """World-frame extent of the wall geom (for reports and plots)."""
    g = engine.model.geom("wall")
    c, h = engine.data.geom_xpos[g.id], engine.model.geom_size[g.id]
    return {f"wall_{a}_{s}": float(c[i] + sign * h[i]) for i, a in enumerate("xyz")
            for s, sign in (("min", -1), ("max", 1))}


def plan_episode(exp: dict, variant: str, seed: int, engine: SimulationEngine, old_path,
                 exact: ExactClearance | None = None) -> dict:
    s = exp["planning"]
    mujoco.mj_forward(engine.model, engine.data)
    row = {"variant": variant, **wall_box(engine)}
    checker = CollisionChecker.from_engine(engine, clearance=s["clearance"], cartesian_step=s["cartesian_step"],
                                           max_joint_step=s["max_joint_step"], limit_margin=s["joint_limit_margin"],
                                           exact_box_box=s.get("exact_box_box", True))
    monitor = CollisionChecker.from_engine(engine, clearance=0.10, cartesian_step=s["cartesian_step"] / 4,
                                           limit_margin=0.0)
    start = np.asarray(exp["start"], float)
    exact = exact or ExactClearance(engine)
    row["start_clearance_mm"] = 1e3 * monitor.min_distance(start)
    if old_path is None:
        row |= {"old_plan_valid": None, "old_plan_reason": "no reference plan", "old_plan_min_distance_mm": float("nan")}
    else:
        ok, why = checker.path_valid(old_path)
        row |= {"old_plan_valid": ok, "old_plan_reason": why,
                "old_plan_min_distance_mm": 1e3 * path_clearance(monitor, old_path),
                "old_plan_min_exact_distance_mm": 1e3 * exact.path(old_path)}
    row |= plan_case(engine, variant, scenario(exp), s, seed)
    row["variant"] = variant
    if row["success"]:
        planner = MotionPlanner.from_engine(engine, PlanningSettings.from_config(s))
        plan = planner.plan_to_position(np.asarray(exp["target"], float), start, seed=seed)
        row["path_min_exact_clearance_mm"] = 1e3 * exact.path(plan.path)
        row["path_min_exact_clearance_samples_mm"] = 1e3 * exact.at_checked_samples(plan.path, planner.checker)
    return row


def run_robustness(exp_path: Path = ROBUSTNESS_CONFIG, out_dir: Path | None = None,
                   variants: list[str] | None = None, seeds: list[int] | None = None,
                   controllers: list[str] | None = None, execution: bool = True,
                   plots: bool = True, verbose: bool = True) -> dict:
    exp = load_experiment(exp_path)
    names = variants or list(exp["variants"])
    seeds = list(seeds if seeds is not None else exp["seeds"])
    ctrls = {k: v for k, v in exp["execution"]["controllers"].items() if controllers is None or k in controllers}
    old = reference_plans(exp, seeds)

    plan_rows = []
    for name in names:
        engine = variant_engine(exp, name)
        exact = ExactClearance(engine)
        for seed in seeds:
            plan_rows.append(plan_episode(exp, name, seed, engine, old[seed], exact))
        if verbose:
            rs = [r for r in plan_rows if r["variant"] == name]
            print(f"  plan {name:15s} replanned {sum(r['success'] for r in rs)}/{len(rs)}  "
                  f"old plan invalid {sum(r['old_plan_valid'] is False for r in rs)}/{len(rs)}  "
                  f"median {np.median([r['total_time_s'] for r in rs]):.3f} s", flush=True)
        del engine

    exec_rows = []
    if execution:
        base = load_config()
        sc = scenario(exp)
        for name in names:
            for c_name, c_cfg in ctrls.items():
                for seed in seeds:
                    cfg = execution_config(base, exp, sc, c_cfg, seed, "planned")
                    cfg["environment_overrides"] = exp["variants"][name]["overrides"]
                    measure = ExactClearance(variant_engine(exp, name))  # plant geometry of this episode
                    r = execute_case(cfg, exp, name, c_name, seed, "planned", measure=measure)
                    r["variant"] = r.pop("scenario")
                    exec_rows.append(r)
                    if verbose:
                        print(f"  exec {name:15s} {c_name:10s} seed {seed} -> {r['outcome']:16s} final "
                              f"{r.get('final_ee_error_mm', float('nan')):7.3f} mm  contacts {r.get('contact_steps', '-')}  "
                              f"clearance {r.get('min_clearance_mm', float('nan')):5.1f} mm (exact "
                              f"{r.get('min_exact_clearance_mm', float('nan')):5.1f} mm)", flush=True)

    summary = summarize(plan_rows, exec_rows, names, list(ctrls))
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
        "experiment_config": exp, "planning_runs": plan_rows, "execution_runs": exec_rows, "summary": summary,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        for fname, data in (("planning_runs.csv", plan_rows), ("execution_runs.csv", exec_rows),
                            ("summary.csv", summary)):
            if data:
                keys = list(dict.fromkeys(k for r in data for k in r))
                with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(data)
        (out_dir / "summary.md").write_text(markdown_table(summary, exp), encoding="utf-8")
        if plots:
            report["plots"] = save_plots(out_dir, exp, names, seeds, old)
    return report


def summarize(plan_rows: list[dict], exec_rows: list[dict], names: list[str], ctrls: list[str]) -> list[dict]:
    out = []
    for name in names:
        ps = [r for r in plan_rows if r["variant"] == name]
        ok = [r for r in ps if r["success"]]
        row = {
            "variant": name, "episodes": len(ps),
            "old_plan_invalid": sum(r["old_plan_valid"] is False for r in ps),
            "old_plan_min_distance_mm": float(min(r["old_plan_min_distance_mm"] for r in ps)),
            "replan_success": len(ok),
            "plan_time_median_s": float(np.median([r["total_time_s"] for r in ps])),
            "plan_time_max_s": float(np.max([r["total_time_s"] for r in ps])),
            "path_length_mean_rad": float(np.mean([r["smoothed_length_rad"] for r in ok])) if ok else float("nan"),
            "path_length_min_rad": float(np.min([r["smoothed_length_rad"] for r in ok])) if ok else float("nan"),
            "path_length_max_rad": float(np.max([r["smoothed_length_rad"] for r in ok])) if ok else float("nan"),
            "path_min_clearance_mm": float(min(r["path_min_clearance_mm"] for r in ok)) if ok else float("nan"),
            "path_min_exact_clearance_mm": float(min(r["path_min_exact_clearance_mm"] for r in ok)) if ok else float("nan"),
            "old_plan_min_exact_distance_mm": float(min(r["old_plan_min_exact_distance_mm"] for r in ps)),
            "revalidated": sum(bool(r.get("path_revalidated")) for r in ok),
        }
        for c in ctrls:
            es = [r for r in exec_rows if r["variant"] == name and r["controller"] == c]
            run = [r for r in es if r["outcome"] != "planning_failed"]
            row |= {
                f"{c}_success": sum(r["outcome"] == "success" for r in es), f"{c}_runs": len(es),
                f"{c}_contact_steps": int(sum(r["contact_steps"] for r in run)),
                f"{c}_min_clearance_mm": float(min(r["min_clearance_mm"] for r in run)) if run else float("nan"),
                f"{c}_min_exact_clearance_mm": float(min(r["min_exact_clearance_mm"] for r in run)) if run else float("nan"),
                f"{c}_final_error_max_mm": float(max(r["final_ee_error_mm"] for r in run)) if run else float("nan"),
                f"{c}_tracking_max_rad": float(max(r["tracking_max_rad"] for r in run)) if run else float("nan"),
            }
        out.append(row)
    return out


def _f(x, spec="{:.2f}"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return spec.format(x)


def markdown_table(summary: list[dict], exp: dict) -> str:
    ctrls = list(exp["execution"]["controllers"])
    ctrls = [c for c in ctrls if f"{c}_runs" in summary[0]] if summary else []
    head = ("| variant | wall | old plan invalid | old plan min dist contact / exact (mm) | replanned | plan time median / max (s) | "
            "path length mean [min-max] (rad) | path clearance contact / exact (mm) | "
            + " | ".join(f"{c}: success, contact steps, min clearance contact / exact (mm), max final err (mm), max tracking err (mrad)"
                         for c in ctrls) + " |")
    lines = ["## Wall robustness (every episode reported)", "", head,
             "|" + "---|" * (8 + len(ctrls))]
    for s in summary:
        cells = [s["variant"], exp["variants"][s["variant"]]["description"],
                 f"{s['old_plan_invalid']}/{s['episodes']}", f"{_f(s['old_plan_min_distance_mm'], '{:.1f}')} / {_f(s['old_plan_min_exact_distance_mm'], '{:.1f}')}",
                 f"{s['replan_success']}/{s['episodes']}",
                 f"{_f(s['plan_time_median_s'], '{:.3f}')} / {_f(s['plan_time_max_s'], '{:.3f}')}",
                 f"{_f(s['path_length_mean_rad'])} [{_f(s['path_length_min_rad'])}-{_f(s['path_length_max_rad'])}]",
                 f"{_f(s['path_min_clearance_mm'], '{:.1f}')} / {_f(s['path_min_exact_clearance_mm'], '{:.1f}')}"]
        for c in ctrls:
            cells.append(f"{s[f'{c}_success']}/{s[f'{c}_runs']}, {s[f'{c}_contact_steps']}, "
                         f"{_f(s[f'{c}_min_clearance_mm'], '{:.1f}')} / {_f(s[f'{c}_min_exact_clearance_mm'], '{:.1f}')}, "
                         f"{_f(s[f'{c}_final_error_max_mm'], '{:.3f}')}, "
                         f"{_f(1e3 * s[f'{c}_tracking_max_rad'], '{:.3f}')}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_plots(out_dir: Path, exp: dict, names: list[str], seeds: list[int], old: dict) -> list[str]:
    """End-effector paths of every seed per variant (side view, y-z), old plan dashed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    from planning.visualization import ee_polyline

    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    start, target = np.asarray(exp["start"], float), np.asarray(exp["target"], float)
    for ax, name in zip(axes.ravel(), names):
        engine = variant_engine(exp, name)
        mujoco.mj_forward(engine.model, engine.data)
        box = wall_box(engine)
        planner = MotionPlanner.from_engine(engine, PlanningSettings.from_config(exp["planning"]))
        fk = planner.kin.fk
        ax.add_patch(plt.Rectangle((box["wall_y_min"], box["wall_z_min"]), box["wall_y_max"] - box["wall_y_min"],
                                   box["wall_z_max"] - box["wall_z_min"], color="#2a6fdb", alpha=0.5))
        for i, seed in enumerate(seeds):
            if old[seed] is not None:
                o = ee_polyline(fk, old[seed], 15)
                ax.plot(o[:, 1], o[:, 2], "--", color="#d62728", lw=0.8, alpha=0.7,
                        label="old plan (reference wall)" if i == 0 else None)
            plan = planner.plan_to_position(target, start, seed=seed)
            if plan.success:
                p = ee_polyline(fk, plan.path, 15)
                ax.plot(p[:, 1], p[:, 2], color="#2e9e6a", lw=1.2, alpha=0.8, label="replanned" if i == 0 else None)
        ax.plot([fk(start)[1]], [fk(start)[2]], "ko", ms=4)
        ax.plot([target[1]], [target[2]], "r*", ms=8)
        ax.set_title(f"{name}: {exp['variants'][name]['description']}", fontsize=7)
        ax.set_xlim(0.45, -0.45)
        ax.set_ylim(0.35, 0.95)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        del planner, engine
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.supxlabel("end-effector y (m)", fontsize=8)
    fig.supylabel("end-effector z (m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_paths.png", dpi=130)
    plt.close(fig)
    return ["robustness_paths.png"]
