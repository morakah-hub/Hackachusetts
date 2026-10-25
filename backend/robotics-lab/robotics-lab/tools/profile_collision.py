"""Collision-checker cost and correctness, v0.5 (contacts only) vs v0.5.2 (exact box-box), on the
configurations the planner really checks.

1. Workload: the wall scene is planned with seeds 0..N-1 (v0.5.2 settings) and every configuration
   the checker evaluates is recorded (plus uniform random configurations in each scene).
2. Each recorded configuration is checked by
     v0.5      CollisionChecker(exact_box_box=False): MuJoCo margin contacts only;
     v0.5.2    CollisionChecker(exact_box_box=True): box-box pairs via mj_geomDistance, broadphase filtered;
     all-pairs v0.5.2 without the broadphase filter (every box-box pair queried);
     exhaustive mj_geomDistance over every collidable robot-environment pair (ExactClearance), the
               reference for "distance < clearance", plus mj_collision with margin 0 for self-collision;
   timed per configuration, and the answers are compared.
3. Planning time / checks per plan for both checkers on the planning scenarios (seeds 0..N-1).

    .\\.venv\\Scripts\\python.exe tools\\profile_collision.py            # ~3 min
Writes results/profile_collision/{profile.json, summary.md}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker  # noqa: E402
from planning.motion_planner import MotionPlanner, PlanningSettings  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.planning_benchmark import load_bench  # noqa: E402
from simulation.robustness import ExactClearance  # noqa: E402


def settings_for(bench: dict, sc: dict, exact: bool) -> PlanningSettings:
    s = json.loads(json.dumps(bench["planning"]))
    for k, v in (sc.get("planning") or {}).items():
        s[k] = {**s[k], **v} if isinstance(v, dict) else v
    s["exact_box_box"] = exact
    return PlanningSettings.from_config(s)


def record_workload(engine, bench, seeds):
    sc = bench["scenarios"]["wall"]
    planner = MotionPlanner.from_engine(engine, settings_for(bench, sc, True))
    log = []
    original = planner.checker._collide

    def logging_collide(q):
        log.append(np.array(q, float))
        original(q)

    planner.checker._collide = logging_collide
    for seed in seeds:
        planner.plan_to_position(np.asarray(sc["target"], float), np.asarray(sc["start"], float), seed=seed)
    return np.array(log)


def time_checks(fn, qs, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for q in qs:
            fn(q)
        best = min(best, time.perf_counter() - t0)
    return 1e6 * best / len(qs)  # us per configuration


def compare(engine, qs, clearance):
    old = CollisionChecker.from_engine(engine, clearance=clearance, exact_box_box=False)
    new = CollisionChecker.from_engine(engine, clearance=clearance)
    allp = CollisionChecker.from_engine(engine, clearance=clearance)
    allp._ex_rbound = np.full_like(allp._ex_rbound, np.inf)  # the bound never rules a pair out
    exact = ExactClearance(engine, cap=clearance)
    selfc = CollisionChecker.from_engine(engine, clearance=0.0, exact_box_box=False)
    m = selfc.model

    def self_collision(q):
        selfc._collide(q)
        d = selfc.data
        return any(selfc._is_robot_geom[c.geom1] and selfc._is_robot_geom[c.geom2] for c in d.contact[: d.ncon])

    def truth(q):
        return exact.distance(q) < clearance or self_collision(q)

    a_old = np.array([old.in_collision(q) for q in qs])
    a_new = np.array([new.in_collision(q) for q in qs])
    a_all = np.array([allp.in_collision(q) for q in qs])
    ref = np.array([truth(q) for q in qs])
    n_new = new.exact_queries / len(qs)
    n_all = allp.exact_queries / len(qs)
    out = {
        "configs": len(qs), "invalid_by_truth": int(ref.sum()), "box_box_pairs": len(new._ex_rob),
        "v05_false_safe": int((~a_old & ref).sum()), "v05_false_alarm": int((a_old & ~ref).sum()),
        "v052_false_safe": int((~a_new & ref).sum()), "v052_false_alarm": int((a_new & ~ref).sum()),
        "allpairs_equals_v052": bool(np.array_equal(a_new, a_all)),
        "geomdistance_calls_per_config_v052": n_new, "geomdistance_calls_per_config_allpairs": n_all,
        "us_v05": time_checks(old.in_collision, qs), "us_v052": time_checks(new.in_collision, qs),
        "us_allpairs": time_checks(allp.in_collision, qs),
        "us_exhaustive": time_checks(lambda q: exact.distance(q), qs, repeat=1),
        "exhaustive_pairs": len(exact.pairs),
    }
    out["us_mj_collision_only"] = time_checks(old._collide, qs)
    del m
    return out


def planning_times(bench, scenarios, seeds):
    rows = []
    for name in scenarios:
        sc = bench["scenarios"][name]
        engine = SimulationEngine(sc["environment"])
        for exact in (False, True):
            planner = MotionPlanner.from_engine(engine, settings_for(bench, sc, exact))
            for seed in seeds:
                c0, e0 = planner.checker.config_checks, planner.checker.exact_queries
                t0 = time.perf_counter()
                start = np.asarray(sc["start"], float)
                if "goal" in sc:
                    plan = planner.plan_joint(np.asarray(sc["goal"], float), start, seed=seed)
                else:
                    plan = planner.plan_to_position(np.asarray(sc["target"], float), start, seed=seed)
                rows.append({"scenario": name, "checker": "v0.5.2" if exact else "v0.5", "seed": seed,
                             "success": plan.success, "time_s": time.perf_counter() - t0,
                             "config_checks": planner.checker.config_checks - c0,
                             "geomdistance_calls": planner.checker.exact_queries - e0,
                             "length_rad": plan.stats.get("smoothed_length_rad", float("nan")) if plan.success else float("nan")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--random", type=int, default=2000, help="uniform random configurations per scene")
    ap.add_argument("--output", type=Path, default=Path("results/profile_collision"))
    args = ap.parse_args()
    bench = load_bench()
    seeds = list(range(args.seeds))
    clearance = bench["planning"]["clearance"]
    report = {"mujoco": mujoco.__version__, "clearance": clearance, "workloads": {}}

    wall_engine = SimulationEngine("planning_demo")
    qs = record_workload(wall_engine, bench, seeds)
    print(f"recorded {len(qs)} configurations checked while planning the wall scene, seeds {seeds}")
    report["workloads"]["wall planner workload"] = compare(wall_engine, qs, clearance)
    rng = np.random.default_rng(0)
    for scene in ("planning_demo", "obstacles", "planning_narrow"):
        engine = SimulationEngine(scene)
        c = CollisionChecker.from_engine(engine)
        qr = rng.uniform(c.q_min, c.q_max, size=(args.random, len(c.q_min)))
        report["workloads"][f"{scene} uniform random"] = compare(engine, qr, clearance)
    for k, v in report["workloads"].items():
        print(f"  {k:28s} " + "  ".join(f"{kk}={vv:.1f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))

    report["planning"] = planning_times(bench, ["wall", "obstacles_blocked", "narrow_passage", "multi_ik_near_obstacle"], seeds)
    lines = ["| workload | configs | invalid (truth) | v0.5 false safe / alarm | v0.5.2 false safe / alarm | "
             "geomDistance calls per config (filtered / all pairs) | us per config: mj_collision / v0.5 / v0.5.2 / all-pairs / exhaustive |",
             "|---|---|---|---|---|---|---|"]
    for k, v in report["workloads"].items():
        lines.append(f"| {k} | {v['configs']} | {v['invalid_by_truth']} | {v['v05_false_safe']} / {v['v05_false_alarm']} | "
                     f"{v['v052_false_safe']} / {v['v052_false_alarm']} | {v['geomdistance_calls_per_config_v052']:.2f} / "
                     f"{v['geomdistance_calls_per_config_allpairs']:.0f} | {v['us_mj_collision_only']:.0f} / {v['us_v05']:.0f} / "
                     f"{v['us_v052']:.0f} / {v['us_allpairs']:.0f} / {v['us_exhaustive']:.0f} |")
    lines += ["", "| scenario | checker | success | planning time median / max (s) | config checks median | "
              "geomDistance calls median | length mean (rad) |", "|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(r["scenario"] for r in report["planning"]):
        for ck in ("v0.5", "v0.5.2"):
            rs = [r for r in report["planning"] if r["scenario"] == name and r["checker"] == ck]
            ok = [r for r in rs if r["success"]]
            lines.append(f"| {name} | {ck} | {len(ok)}/{len(rs)} | {np.median([r['time_s'] for r in rs]):.3f} / "
                         f"{max(r['time_s'] for r in rs):.3f} | {np.median([r['config_checks'] for r in rs]):.0f} | "
                         f"{np.median([r['geomdistance_calls'] for r in rs]):.0f} | "
                         f"{np.mean([r['length_rad'] for r in ok]) if ok else float('nan'):.2f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "profile.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (args.output / "summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
