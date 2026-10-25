r"""v0.6 payload study: 4 controller configurations x 3 object masses x 10 frozen cases.

    .\.venv\Scripts\python.exe pick_place_study.py                       # main study -> results\v0.6\study\
    .\.venv\Scripts\python.exe pick_place_study.py --sensitivity          # bounded sensitivity check
    .\.venv\Scripts\python.exe pick_place_study.py --configs ctc_obs --masses 1.15 --cases eval_00

Episodes run sequentially (one process, bounded memory); each finished episode is appended to
episodes.jsonl at once, and a rerun skips episodes already recorded there (resumable; since v0.7 a resume
is refused when the code commit, config or library versions differ from the recorded ones). Every
attempt is counted. summary*.csv / summary*.json are rebuilt from episodes.jsonl at the end.
"""

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy

from simulation.manipulation import run_case
from simulation.runner import load_config

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "pick_place_study.yaml"


def configured(cfg: dict, name: str, spec: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["controller"]["type"] = spec["type"]
    for sec in ("ctc", "torque_mpc"):
        c["controller"][sec].setdefault("disturbance_observer", {})["enabled"] = bool(spec["observer"])
    c["label"] = name
    return c


def provenance(cfg_path: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "*.py", "configs", "environments"],
                                    cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty_code": dirty, "config": cfg_path.relative_to(ROOT).as_posix(),
            "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
            "python": sys.version.split()[0], "mujoco": mujoco.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "osqp": osqp.__version__, "platform": platform.platform()}


def load_done(path: Path) -> dict:
    done = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def _carry_ee_err_mm(r: dict) -> float | None:
    vals = [p["ee_err_max"] for p in r["phases"] if p["phase"] in ("hold", "transport", "lower")]
    return 1e3 * max(vals) if vals else None


def summarize(rows: list[dict], group: tuple[str, ...]) -> list[dict]:
    out = {}
    for r in rows:
        out.setdefault(tuple(r[g] for g in group), []).append(r)
    table = []
    for key, rs in sorted(out.items()):
        ok = [r for r in rs if r["success"]]
        stages = {}
        for r in rs:
            if not r["success"]:
                stages[r["failure_stage"]] = stages.get(r["failure_stage"], 0) + 1

        def stat(name, fn=np.max, src=rs):
            vals = [r[name] for r in src if r.get(name) is not None]
            return float(fn(vals)) if vals else None
        carry = [v for v in (_carry_ee_err_mm(r) for r in rs) if v is not None]
        row = {**dict(zip(group, key)), "episodes": len(rs), "successes": len(ok), "failure_stages": stages,
               "forbidden_contact_episodes": sum(r["forbidden_contact_steps"] > 0 for r in rs),
               "placement_error_mm_median": stat("placement_error_mm", np.median, ok),
               "placement_error_mm_max": stat("placement_error_mm", np.max, ok),
               "placement_error_z_mm_median": stat("placement_error_z_mm", np.median, ok),
               "ee_err_max_mm_median": stat("ee_err_max_mm", np.median),
               "ee_err_max_mm_max": stat("ee_err_max_mm"),
               "carry_ee_err_max_mm_median": float(np.median(carry)) if carry else None,
               "carry_ee_err_max_mm_max": float(np.max(carry)) if carry else None,
               "ee_err_rms_mm_median": stat("ee_err_rms_mm", np.median),
               "joint_err_max_rad_max": stat("joint_err_max_rad"),
               "lift_height_mm_min": stat("lift_height_mm", np.min),
               "hold_slip_mm_max": stat("hold_slip_mm"),
               "carry_slip_mm_max": stat("carry_slip_mm"),
               "min_clearance_robot_env_mm": stat("min_clearance_robot_env_mm", np.min),
               "min_clearance_object_env_mm": stat("min_clearance_object_env_mm", np.min),
               "torque_saturation_steps": int(sum(r["torque_saturation_steps"] for r in rs)),
               "position_limit_steps": int(sum(r["position_limit_steps"] for r in rs)),
               "velocity_limit_steps": int(sum(r["velocity_limit_steps"] for r in rs)),
               "sim_time_s_mean": stat("sim_time_s", np.mean), "wall_time_s_mean": stat("wall_time_s", np.mean)}
        mpc = [r["mpc"] for r in rs if "mpc" in r]
        if mpc:
            statuses = {}
            for m_ in mpc:
                for k, v in m_["statuses"].items():
                    statuses[k] = statuses.get(k, 0) + v
            row.update(mpc_updates=sum(m_["updates"] for m_ in mpc),
                       mpc_fallback_updates=sum(m_["fallback_updates"] for m_ in mpc),
                       mpc_fallback_steps=sum(m_["fallback_steps"] for m_ in mpc), mpc_statuses=statuses,
                       mpc_solve_ms_mean=float(np.mean([m_["solve_ms_mean"] for m_ in mpc])),
                       mpc_solve_ms_p95_max=float(np.max([m_["solve_ms_p95"] for m_ in mpc])),
                       mpc_update_ms_max=float(np.max([m_["update_ms_max"] for m_ in mpc])),
                       mpc_deadline_overruns=int(sum(m_["deadline_overruns"] for m_ in mpc)),
                       mpc_prim_res_max=float(np.max([m_["prim_res_max"] or 0 for m_ in mpc])),
                       mpc_dual_res_max=float(np.max([m_["dual_res_max"] or 0 for m_ in mpc])))
        row["controller_rate_hz"] = rs[0].get("controller_rate_hz")
        table.append(row)
    return table


def write_summary(out: Path, rows: list[dict], group: tuple[str, ...], name: str) -> list[dict]:
    table = summarize(rows, group)
    (out / f"{name}.json").write_text(json.dumps(table, indent=1))
    keys = list(dict.fromkeys(k for row in table for k in row))
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in table:
            w.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="v0.6 payload study")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--output", default=str(ROOT / "results" / "v0.6" / "study"))
    ap.add_argument("--sensitivity", action="store_true", help="run the bounded sensitivity check instead")
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--masses", nargs="*", type=float)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="resume even if the recorded episodes came from a different code/config version")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)
    st = cfg["study"]
    out = Path(args.output) / ("sensitivity" if args.sensitivity else "")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "episodes.jsonl"
    done = load_done(jsonl)
    meta = provenance(cfg_path)
    prov_file = out / "provenance.json"
    if done and prov_file.is_file() and not args.allow_mixed:
        old = json.loads(prov_file.read_text())
        changed = [k for k in ("git_commit", "config_sha256", "mujoco", "numpy", "scipy", "osqp")
                   if old.get(k) != meta.get(k)]
        if changed or meta.get("git_dirty_code"):
            sys.exit(f"refusing to resume {jsonl}: {len(done)} recorded episodes came from a different code/config "
                     f"version ({', '.join(changed) or 'uncommitted code changes now'}). Use a new --output folder, or "
                     f"--allow-mixed to append anyway (v0.7 guard against silently mixed studies).")
    if not done or not prov_file.is_file():
        prov_file.write_text(json.dumps(meta, indent=1))
    cases = {c["name"]: c for c in cfg["cases"]}

    jobs = []
    if args.sensitivity:
        sen = st["sensitivity"]
        for variant, physics in sen["variants"].items():
            for cname in sen["configurations"]:
                for case in sen["cases"]:
                    jobs.append((variant, physics, cname, float(sen["mass"]), case))
    else:
        for cname in args.configs or st["configurations"]:
            for mass in args.masses or st["masses"]:
                for case in args.cases or cases:
                    jobs.append(("baseline", None, cname, float(mass), case))

    t0 = time.perf_counter()
    for i, (variant, physics, cname, mass, case_name) in enumerate(jobs):
        key = f"{variant}|{cname}|{mass:g}|{case_name}"
        if key in done:
            continue
        c = configured(cfg, cname, st["configurations"][cname])
        if physics:
            c["physics"] = {**(c.get("physics") or {}), **physics}
        case = dict(cases[case_name], mass=mass)
        r = run_case(c, case)
        r.update(key=key, variant=variant, configuration=cname, physics=c.get("physics"),
                 git_commit=meta["git_commit"])
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, default=float) + "\n")
        done[key] = r
        status = "ok  " if r["success"] else f"FAIL {r['failure_stage']}: {r['failure_reason'][:70]}"
        print(f"[{i + 1}/{len(jobs)} {time.perf_counter() - t0:6.0f}s] {variant:<20} {cname:<9} m={mass:<5g} "
              f"{case_name}: {status}", flush=True)

    rows = list(done.values())
    write_summary(out, rows, ("variant", "configuration") if args.sensitivity else ("configuration", "mass_kg"),
                  "summary")
    table = write_summary(out, rows, ("variant", "configuration") if args.sensitivity else ("configuration",),
                          "summary_by_configuration")
    for row in table:
        tag = row.get("variant", "")
        print(f"{tag} {row['configuration']:<9}: {row['successes']}/{row['episodes']} complete cycles, "
              f"forbidden-contact episodes {row['forbidden_contact_episodes']}, failures {row['failure_stages']}")
    if not args.sensitivity:
        prim = next((r for r in table if r["configuration"] == st["primary"]), None)
        if prim:
            met = prim["successes"] >= st["target_successes"] and prim["forbidden_contact_episodes"] == 0
            print(f"primary {st['primary']}: {prim['successes']}/{prim['episodes']} "
                  f"(target >= {st['target_successes']}, 0 forbidden contacts): {'MET' if met else 'NOT MET'}")


if __name__ == "__main__":
    main()
