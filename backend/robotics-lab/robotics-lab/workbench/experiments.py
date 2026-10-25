"""Guided experiments (v0.7): one-click, sequential, matched comparisons through the normal session.

Each experiment states its question, a short physical prediction (written before any run, and *not* an
assumption about the outcome), what changes and what is held fixed. Its runs are queued one at a time through
`Session` (the same RunRequest path as the GUI and the console), every run is saved like any other run, and
every outcome - including failures - is kept in the experiment record
(results/workbench/experiments/<id>.json, with resolved settings and code provenance). The explanation is
generated from the recorded metrics after the runs; it compares numbers, it does not presume a winner.

These are exploratory single-case comparisons (one object pose, one seed), not the frozen v0.6 study: they do
not inherit its success claims, and one run per condition supports no statistical conclusion.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from simulation.engine import PROJECT_ROOT
from workbench.api import RunRequest
from workbench.runs import RUNS_DIR, Recording, provenance

EXP_DIR = PROJECT_ROOT / "results" / "workbench" / "experiments"
BASE_PICK = dict(scenario="pick_place", object_xy=[0.50, 0.20], place_xy=[0.50, -0.20], plan_seed=0,
                 contact_preset="v0.6-working", grip_preset="v0.6-strong", mpc_snapshots=True)
PULSE = {"body": "hand", "direction": [0.0, 0.0, -1.0], "magnitude_n": 30.0, "start_s": 2.0, "duration_s": 1.0}
NOISE = {"position_std": 0.0005, "velocity_std": 0.005, "seed": 7}

EXPERIMENTS = {
    "payload": {
        "title": "Light versus heavy payload",
        "question": "How much does an unmodelled payload deflect the arm when nothing estimates it?",
        "prediction": "The object's weight acts like a constant push the controller does not model; with a fixed "
                      "stiffness (CTC, wn = 25 rad/s) the hand should sag roughly in proportion to the mass while "
                      "carrying.",
        "fixed": "CTC without observer, object (0.50, 0.20) -> place (0.50, -0.20), seed 0, v0.6 contact and gripper presets",
        "varied": "payload 0.15 kg vs 1.15 kg",
        "runs": [("0.15 kg", dict(BASE_PICK, controller="torque_ctc", observer=False, payload_kg=0.15)),
                 ("1.15 kg", dict(BASE_PICK, controller="torque_ctc", observer=False, payload_kg=1.15))]},
    "observer": {
        "title": "Disturbance observer off versus on",
        "question": "Does estimating the disturbance torque remove the payload's effect on tracking?",
        "prediction": "With the observer the controller should learn the extra torque within a few 10 ms time "
                      "constants and cancel most of the sag; without it the sag stays.",
        "fixed": "CTC, 1.15 kg, same case and presets", "varied": "observer off vs on",
        "runs": [("observer OFF", dict(BASE_PICK, controller="torque_ctc", observer=False, payload_kg=1.15)),
                 ("observer ON", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.15))]},
    "controllers": {
        "title": "Computed-torque control versus torque MPC",
        "question": "With the observer on, what does the torque MPC change compared with CTC on the same task?",
        "prediction": "Both should complete the task; MPC may track more tightly (stiffer weights, preview of the "
                      "reference) at a higher computational cost. Either could do better on a given metric.",
        "fixed": "observer on, 1.15 kg, same case and presets", "varied": "CTC (500 Hz) vs torque MPC (50 Hz QP)",
        "runs": [("CTC + observer", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.15)),
                 ("MPC + observer", dict(BASE_PICK, controller="torque_mpc", observer=True, payload_kg=1.15))]},
    "contact": {
        "title": "Working contact preset versus the sensitivity condition",
        "question": "Does the grasp of the heavy block depend on MuJoCo's friction-cone setting?",
        "prediction": "v0.6 found the soft pyramidal cone lets the block creep through the grasp; the run with the "
                      "MuJoCo default may slip or drop. The controllers are not involved.",
        "fixed": "CTC + observer, 1.15 kg, same case and gripper", "varied": "elliptic cone, impratio 10 vs pyramidal cone, impratio 1",
        "runs": [("v0.6-working", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.15)),
                 ("mujoco-default", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.15,
                                         contact_preset="mujoco-default"))]},
    "grip": {
        "title": "Strong versus weak grasp",
        "question": "Can the original Menagerie gripper gains hold a 1 kg block?",
        "prediction": "Friction can carry at most mu x squeeze: ~37 N x 1.0 with the v0.6 gripper vs ~2.5 N with "
                      "the Menagerie gains, against 9.8 N of weight. The weak grasp should fail the lift check.",
        "fixed": "CTC + observer, 1.0 kg, same case and contact preset", "varied": "gripper preset v0.6-strong vs menagerie-weak",
        "runs": [("strong (v0.6)", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.0)),
                 ("weak (Menagerie)", dict(BASE_PICK, controller="torque_ctc", observer=True, payload_kg=1.0,
                                           grip_preset="menagerie-weak"))]},
    "force": {
        "title": "Force pulse: observer off versus on",
        "question": "How far does a 30 N downward push on the hand (1 s) displace it, with and without the observer?",
        "prediction": "The push is an unmeasured disturbance; the observer estimates it from the motion and should "
                      "reduce the deflection after a short transient. The controller never receives the force itself.",
        "fixed": "table reach, CTC, pulse 30 N on the hand along -z from t = 2.0 s for 1.0 s", "varied": "observer off vs on",
        "runs": [("observer OFF", dict(scenario="table_reach", controller="torque_ctc", observer=False, force_pulse=PULSE)),
                 ("observer ON", dict(scenario="table_reach", controller="torque_ctc", observer=True, force_pulse=PULSE))]},
    "noise": {
        "title": "Disturbance rejection versus noise amplification",
        "question": "What does the observer cost when the joint measurements are noisy?",
        "prediction": "The observer differentiates measured velocity, so measurement noise should enter the torque "
                      "command (more jitter) while the push is still rejected. Noise is applied to the controller's "
                      "arm observations only; metrics use the true state.",
        "fixed": "table reach, CTC, the 30 N pulse of the force experiment, noise seed 7",
        "varied": "observer off/on x noise off/on (sigma_q = 0.5 mrad, sigma_qd = 5 mrad/s)",
        "runs": [("obs OFF, no noise", dict(scenario="table_reach", controller="torque_ctc", observer=False, force_pulse=PULSE)),
                 ("obs ON, no noise", dict(scenario="table_reach", controller="torque_ctc", observer=True, force_pulse=PULSE)),
                 ("obs OFF, noise", dict(scenario="table_reach", controller="torque_ctc", observer=False, force_pulse=PULSE, noise=NOISE)),
                 ("obs ON, noise", dict(scenario="table_reach", controller="torque_ctc", observer=True, force_pulse=PULSE, noise=NOISE))]},
}

_lock = threading.Lock()
_history: list[dict] = []


def start(session, key: str) -> dict:
    spec = EXPERIMENTS[key]
    if session.experiment is not None and session.experiment["state"] == "running":
        from workbench.api import RequestError
        raise RequestError(f"experiment '{session.experiment['key']}' is still running")
    runs = [{"label": label, "request": RunRequest().replace(**changes).validate().to_dict(), "run_id": None,
             "outcome": None} for label, changes in spec["runs"]]
    exp = {"id": time.strftime("%Y%m%d-%H%M%S-") + key, "key": key, "state": "running", "runs": runs, "index": 0,
           **{k: spec[k] for k in ("title", "question", "prediction", "fixed", "varied")},
           "provenance": provenance(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "prev_speed": session.speed}
    with _lock:
        session.experiment = exp
        if _on_finished not in session.on_run_finished:
            session.on_run_finished.append(_on_finished)
    session.act("speed", value=0)  # experiments run as fast as possible (physics timestep unchanged)
    _queue(session, exp)
    session.log("experiment", f"experiment '{key}' started: {len(runs)} sequential runs")
    return exp


def _queue(session, exp):
    r = exp["runs"][exp["index"]]
    req = dict(r["request"], label=f"{exp['key']}: {r['label']}")
    session.act("run_request", request=req)


def _on_finished(session, result) -> None:
    exp = session.experiment
    if exp is None or exp["state"] != "running":
        return
    r = exp["runs"][exp["index"]]
    r.update(run_id=result.run_id, outcome=result.outcome, failure_stage=result.failure_stage,
             failure_reason=result.failure_reason)
    exp["index"] += 1
    if exp["index"] < len(exp["runs"]) and result.outcome != "stopped":
        _queue(session, exp)
        return
    exp["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        exp["analysis"] = analyse(exp, session.runs_dir)
    except Exception as e:  # noqa: BLE001 - keep the record even if the analysis fails
        exp["analysis"] = {"error": repr(e)}
    exp["state"] = "stopped" if result.outcome == "stopped" else "done"   # last: readers wait for this
    session.act("speed", value=exp["prev_speed"])
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / f"{exp['id']}.json").write_text(json.dumps(exp, indent=1, default=str))
    with _lock:
        _history.append(exp)
    session.log("experiment", f"experiment '{exp['key']}' {exp['state']}: saved to results/workbench/experiments/{exp['id']}.json")


# ------------------------------------------------------------------ analysis from recorded data


def run_metrics(rec: Recording) -> dict:
    d, res = rec.data, rec.result or {}
    m = res.get("metrics", {})
    ee = 1e3 * np.linalg.norm(d["ee"] - d["ee_ref"], axis=1)
    phase = d["phase"]
    carry = np.isin(phase, ["hold", "transport", "lower"]) if rec.meta["kind"] == "pick_place" else np.zeros(len(phase), bool)
    out = {"outcome": res.get("outcome"), "failure_stage": res.get("failure_stage"),
           "failure_reason": res.get("failure_reason"),
           "max EE error while carrying (mm)": float(ee[carry].max()) if carry.any() else None,
           "placement error (mm)": m.get("placement_error_mm"), "lift height (mm)": m.get("lift_height_mm"),
           "slip while carrying (mm)": m.get("carry_slip_mm"),
           "forbidden-contact steps": m.get("forbidden_contact_steps", m.get("robot_contact_steps"))}
    force = np.linalg.norm(d["force"], axis=1) > 0
    if force.any():
        t0, t1 = rec.t[force][0], rec.t[force][-1]
        win = (rec.t >= t0) & (rec.t <= t1 + 0.3)
        out["max EE error during push + 0.3 s (mm)"] = float(ee[win].max())
        late = (rec.t >= t1 - 0.3) & (rec.t <= t1)
        out["mean EE error in the last 0.3 s of the push (mm)"] = float(ee[late].mean())
    tau = d["tau"]
    if len(tau) > 20:
        # High-frequency torque content: applied torque minus its centred 50 ms moving average (5 samples of 10 ms),
        # RMS over joints and the whole run. Smooth planned-motion torques largely cancel; noise-driven jitter stays.
        k = np.ones(5) / 5
        smooth = np.column_stack([np.convolve(tau[:, j], k, mode="same") for j in range(tau.shape[1])])
        hf = (tau - smooth)[3:-3]
        out["high-frequency torque (RMS of tau - 50 ms moving average, Nm)"] = float(np.sqrt(np.mean(hf ** 2)))
    out["max EE error, whole run (mm)"] = float(ee.max())
    out["wall time (s)"] = res.get("wall_time_s")
    mpc = m.get("mpc") or {}
    if mpc:
        out["MPC fallback updates"] = mpc.get("fallback_updates", mpc.get("fallbacks"))
    return out


def analyse(exp: dict, runs_dir: Path = RUNS_DIR) -> dict:
    rows, recs = [], []
    for r in exp["runs"]:
        if not r.get("run_id"):
            rows.append({"label": r["label"], "metrics": {"outcome": "not run"}})
            continue
        rec = Recording.load(Path(runs_dir) / r["run_id"])
        recs.append(rec)
        rows.append({"label": r["label"], "run_id": r["run_id"], "metrics": run_metrics(rec)})
    lines = []
    for row in rows:
        mt = row["metrics"]
        if mt.get("outcome") not in ("success",):
            lines.append(f"{row['label']}: {str(mt.get('outcome')).upper()}"
                         + (f" at {mt.get('failure_stage')}: {mt.get('failure_reason')}" if mt.get("failure_reason") else ""))
    keys = [k for k in rows[0]["metrics"] if k not in ("outcome", "failure_stage", "failure_reason")] if rows else []
    for k in keys:
        vals = [row["metrics"].get(k) for row in rows]
        if all(isinstance(v, (int, float)) for v in vals) and len(vals) >= 2:
            a, b = vals[0], vals[-1]
            change = f" ({b / a:.2g}x)" if len(vals) == 2 and a not in (0, None) and abs(a) > 1e-12 else ""
            lines.append(f"{k}: " + " vs ".join(f"{row['label']} {v:.3g}" for row, v in zip(rows, vals)) + change)
    lines.append("One run per condition at one object pose and seed: an observation about these runs, not a general claim.")
    return {"rows": rows, "lines": lines}


# ------------------------------------------------------------------ panel for the UI


def panel(session=None) -> list[dict]:
    secs = [{"type": "intuition", "title": "Guided experiments",
             "text": ["Each experiment runs matched simulations one after another (same scene, seed and presets; only "
                      "the stated setting changes), saves every run, and then explains the recorded differences - "
                      "including failures. Runs execute at maximum speed; open any of them in replay afterwards."]}]
    exp = session.experiment if session is not None else None
    if exp is not None:
        done = sum(1 for r in exp["runs"] if r.get("run_id"))
        rows = [[r["label"], r.get("run_id") or ("running" if i == exp["index"] and exp["state"] == "running" else "queued"),
                 r.get("outcome") or "-"] for i, r in enumerate(exp["runs"])]
        secs.append({"type": "values", "title": f"{exp['title']} - {exp['state']} ({done}/{len(exp['runs'])} runs)",
                     "text": [f"Question: {exp['question']}", f"Prediction (before running): {exp['prediction']}",
                              f"Held fixed: {exp['fixed']}", f"Changed: {exp['varied']}"],
                     "header": ["condition", "run", "outcome"], "table": rows,
                     "buttons": [{"label": f"replay {r['label']}", "cmd": f"replay {r['run_id']}"} for r in exp["runs"] if r.get("run_id")]})
        an = exp.get("analysis")
        if an and "rows" in an:
            keys = list(an["rows"][0]["metrics"].keys())
            secs.append({"type": "consequence", "title": "Result (from the recorded runs)",
                         "header": ["metric", *[r["label"] for r in an["rows"]]],
                         "table": [[k, *[_cell(r["metrics"].get(k)) for r in an["rows"]]] for k in keys],
                         "text": an["lines"]})
        elif an:
            secs.append({"type": "warn", "text": f"analysis failed: {an.get('error')}"})
    for key, spec in EXPERIMENTS.items():
        secs.append({"type": "", "title": spec["title"], "text": [spec["question"], f"Changes: {spec['varied']}. Fixed: {spec['fixed']}."],
                     "buttons": [{"label": f"Run: compare {key}", "cmd": f"compare {key}"}]})
    return secs


def _cell(v):
    if isinstance(v, float):
        return float(np.round(v, 4))
    return "-" if v is None else v
