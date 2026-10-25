"""Instrumented torque-MPC runs for the v0.4 fast/wrist investigation.

    python tools/diagnose_torque_mpc.py                      # all experiments
    python tools/diagnose_torque_mpc.py baseline no_vlimit   # a subset

Each experiment runs one benchmark scenario (configs/benchmark.yaml) with the
torque MPC, overriding a few settings, and logs at every MPC update: measured
state, predicted states and torques over the horizon, references, solver status,
iterations, solve time and slack. It then reports:

  track      max / RMS joint error of the simulated robot (rad)
  1-step     prediction error of x[1] against the next measured state
             (closed loop: the applied torque is the planned tau[0])
  LTI@N      open-loop error of the linear prediction against a nonlinear MuJoCo
             rollout of the same planned torques, at the end of the horizon
             (isolates linearization error from feedback)
  v pred     max |qd| / v_max over the predicted knots (what the QP believes)
  v act      max |qd| / v_max of the simulated robot (every physics step)
  v steps    physics steps above the speed limit
  slack      max soft-constraint slack in the QP solutions
  tau sat    fraction of updates whose first planned torque is at a limit
  iters      mean OSQP iterations; status counts that are not "solved"
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.benchmark import BENCHMARK_CONFIG, deep_merge, scenario_config  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

FAR = [100.0] * 7  # "no velocity limit"

# name: (scenario, torque_mpc overrides, scenario overrides)
EXPERIMENTS = {
    "baseline_fast": ("fast", {}, {}),
    "baseline_wrist": ("wrist", {}, {}),
    "no_vlimit_fast": ("fast", {"velocity_limits": FAR}, {}),
    "no_vlimit_wrist": ("wrist", {"velocity_limits": FAR}, {}),
    "slow_fast": ("fast", {}, {"move_time": 1.5}),            # fast poses, feasible timing (peak v 0.95x)
    "slow_wrist": ("wrist", {}, {"move_time": 3.0}),          # wrist poses, feasible timing (peak v 0.96x)
    "N10_wrist": ("wrist", {"horizon": 10}, {}),
    "N40_wrist": ("wrist", {"horizon": 40}, {}),
    "hz100_N25_wrist": ("wrist", {"control_rate_hz": 100, "horizon": 25}, {}),
    "hz100_N50_wrist": ("wrist", {"control_rate_hz": 100, "horizon": 50}, {}),
    "eps1e-7_wrist": ("wrist", {"solver": {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 4000}}, {}),
    "nowarm_wrist": ("wrist", {"solver": {"warm_starting": False}}, {}),
    "Q1e4_wrist": ("wrist", {"weights": {"position": 1.0e4}}, {}),
    "Q1e6_wrist": ("wrist", {"weights": {"position": 1.0e6}}, {}),
    "vel0_wrist": ("wrist", {"weights": {"velocity": 0.0}}, {}),
    "hard_wrist": ("wrist", {"soft_constraints": False}, {}),
}


def run(name: str, scenario: str, mpc_over: dict, sc_over: dict, lti_every: int = 5) -> dict:
    with open(BENCHMARK_CONFIG, encoding="utf-8") as f:
        bench = yaml.safe_load(f)
    bench["scenarios"][scenario] = {**bench["scenarios"][scenario], **sc_over}
    cfg = scenario_config(load_config(), bench, "torque_mpc", scenario)
    cfg["controller"]["torque_mpc"] = deep_merge(cfg["controller"].get("torque_mpc", {}), mpc_over)
    r = SimulationRunner(cfg)
    c, e = r.controller, r.engine
    vmax = np.asarray(e.robot.velocity_limits)
    records = []
    orig = c._update

    def update(*a, **k):
        orig(*a, **k)
        u = c.stats.updates[-1]
        p = c.last_prediction if not u.fallback else None
        records.append({"t": u.time, "status": u.status, "iters": u.iterations, "slack": u.max_slack,
                        "solve": u.solve_time, "pred": None if p is None or p["time"] != u.time else
                        {k2: p[k2] for k2 in ("x0", "states", "commands")}})

    c._update = update
    duration = bench["scenarios"][scenario].get("duration", r.task.duration)
    log = r.run_headless(duration, log_every=1).arrays()
    q, qd, q_ref = log["q"], log["qd"], log["q_ref"]
    err = np.abs(q - q_ref)

    one_step, lti, vpred, sat = [], [], [], []
    for i, rec in enumerate(records):
        p = rec["pred"]
        if p is None:
            continue
        vpred.append(np.max(np.abs(p["states"][:, 7:]) / vmax))
        tau0 = p["commands"][0]
        sat.append(np.any(np.abs(tau0) >= 0.999 * np.abs(c.torque_limits[:, 1])))
        if i + 1 < len(records) and records[i + 1]["pred"] is not None:
            one_step.append(np.abs(records[i + 1]["pred"]["x0"] - p["states"][0]))
        if i % lti_every == 0:
            x0 = p["x0"]
            true = c.model.rollout(x0[:7], x0[7:], p["commands"])
            lti.append(np.abs(true - p["states"]))
    one_step, lti = np.array(one_step), np.array(lti)  # lti: (updates, N, 14)
    lti_k = lti[:, :, :7].max(axis=2).mean(axis=0)    # mean over updates of the max joint error at stage k
    lti_joint = np.bincount(lti[:, -1, :7].argmax(axis=1), minlength=7)
    lti = lti[:, -1, :]
    stat = Counter(rec["status"] for rec in records)
    return {
        "name": name,
        "track_max": float(err.max()), "track_rms": float(np.sqrt(np.mean(err**2))),
        "onestep_q_max": float(one_step[:, :7].max()), "onestep_qd_max": float(one_step[:, 7:].max()),
        "lti_q_max": float(lti[:, :7].max()), "lti_q_mean": float(lti[:, :7].max(1).mean()),
        "lti_q_mean_by_stage": {int(k + 1): float(lti_k[k]) for k in (0, 4, 9, len(lti_k) - 1) if k < len(lti_k)},
        "lti_worst_joint_counts": lti_joint.tolist(),
        "v_pred_max": float(np.max(vpred)), "v_act_max": float(np.max(np.abs(qd) / vmax)),
        "v_steps": int(np.sum(np.any(np.abs(qd) > vmax, axis=1))),
        "v_excess_max": float(np.max(np.abs(qd) - vmax)),
        "slack_max": float(max(rec["slack"] for rec in records)),
        "tau_sat_frac": float(np.mean(sat)),
        "iters_mean": float(np.mean([rec["iters"] for rec in records])),
        "solve_ms_mean": 1e3 * float(np.mean([rec["solve"] for rec in records])),
        "not_solved": {k: v for k, v in stat.items() if k != "solved"},
        "fallbacks": int(sum(rec["pred"] is None for rec in records)),
    }


def main() -> None:
    names = sys.argv[1:] or list(EXPERIMENTS)
    rows = []
    for name in names:
        matches = [n for n in EXPERIMENTS if n == name or n.startswith(name)]
        for n in matches:
            row = run(n, *EXPERIMENTS[n])
            rows.append(row)
            print(f"{n:18s} track max {row['track_max']:.4f} rms {row['track_rms']:.5f} | 1-step q {row['onestep_q_max']:.1e}"
                  f" qd {row['onestep_qd_max']:.1e} | LTI@N q {row['lti_q_max']:.1e} (mean {row['lti_q_mean']:.1e})"
                  f" | v pred {row['v_pred_max']:.4f} act {row['v_act_max']:.4f} steps {row['v_steps']}"
                  f" | slack {row['slack_max']:.1e} | tau sat {row['tau_sat_frac']:.2f} | iters {row['iters_mean']:.0f}"
                  f" solve {row['solve_ms_mean']:.1f} ms | {row['not_solved']} fb {row['fallbacks']}", flush=True)
            print(f"{'':18s} LTI error by stage {row['lti_q_mean_by_stage']}, worst joint counts {row['lti_worst_joint_counts']}")
    out = Path("results") / "v0.4_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "torque_mpc_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
