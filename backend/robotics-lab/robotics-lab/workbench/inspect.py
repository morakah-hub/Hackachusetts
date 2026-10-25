"""Inspectors, explanations, pre-flight checks and the learning sandbox (v0.7).

Every number shown comes either from the recording (measured / logged by the run) or from an explicitly
named calculation on a *private scratch model* rebuilt from the run's declarative request (never the live
plant). Each section says which. Explanations follow: physical intuition -> equation -> actual values ->
observed consequence.

Sections are plain dicts rendered by the UI: {"type": intuition|equation|values|consequence|warn|detail|"",
"title", "text", "eq", "table", "header", "plot", "details", "buttons"}.
"""

from __future__ import annotations

import threading

import mujoco
import numpy as np

from workbench.api import RequestError, RunRequest
from workbench.runs import Recording

PLOT_FIELDS = ("q", "q_ref", "qd", "qd_ref", "tau", "d_hat", "mpc_cmd", "feedthrough")
G = 9.81
UNAVAILABLE = "unavailable"


def _down(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def telemetry(rec: Recording | None, max_points: int = 1500) -> dict:
    """Time series for the plots, downsampled to at most max_points (uniform in sample index)."""
    if rec is None or not len(rec.t):
        return {"available": False}
    d = rec.data
    idx = _down(len(rec.t), max_points)
    ee_err = 1e3 * np.linalg.norm(d["ee"][idx] - d["ee_ref"][idx], axis=1)
    out = {"available": True, "run_id": rec.meta["run_id"], "t": d["t"][idx], "ee_err_mm": ee_err,
           "phase": d["phase"][idx].tolist(), "fallback": d["fallback"][idx],
           "force_n": np.linalg.norm(d["force"][idx], axis=1), "obj_z": d["obj_pos"][idx, 2],
           "grip_mm": 1e3 * d["grip_width"][idx], "forbidden": d["forbidden"][idx],
           "clearance_mm": d["clearance"][idx], "sample_dt": rec.meta["sample_dt"],
           "events": rec.events, "result": rec.result, "kind": rec.meta["kind"],
           "summary": rec.meta["resolved_summary"]}
    for f in PLOT_FIELDS:
        out[f] = d[f][idx]
    return out


# ------------------------------------------------------------------ scratch models (private, cached)

_cache: dict = {}
_cache_lock = threading.Lock()


class Scratch:
    """Private models rebuilt from a run's request: the plant scene (true object mass) and the controllers'
    nominal model (contact-free, nominal object mass) - both with their own MjData, used for calculations."""

    def __init__(self, request: RunRequest):
        from workbench.session import scene_engine
        self.request = request
        self.engine = scene_engine(request)       # plant geometry and masses (simulation truth)
        e = self.engine
        cfg = request.resolve()["config"]
        cm = cfg.get("controller_model") or {}
        self.nominal = e.contact_free_model(cm.get("nominal_masses")) if not cm.get("contacts", True) else e.model
        self.d_plant = mujoco.MjData(e.model)
        self.d_nom = mujoco.MjData(self.nominal)
        self.lock = threading.Lock()

    def set(self, data, qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0

    def fk_site(self, qpos) -> tuple[np.ndarray, np.ndarray]:
        m, d, e = self.engine.model, self.d_plant, self.engine
        self.set(d, qpos)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, e.ee_site_id)
        return d.site_xpos[e.ee_site_id].copy(), jacp[:, e.arm_dof_adr].copy()

    def nominal_terms(self, qpos, qvel_arm) -> dict:
        """M(q) (arm block), bias h(q, qd) and gravity g(q) of the controllers' nominal model."""
        m, d, e = self.nominal, self.d_nom, self.engine
        self.set(d, qpos)
        d.qvel[e.arm_dof_adr] = qvel_arm
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        h = d.qfrc_bias[e.arm_dof_adr].copy()
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        g = d.qfrc_bias[e.arm_dof_adr].copy()
        return {"M": M[np.ix_(e.arm_dof_adr, e.arm_dof_adr)], "h": h, "g": g,
                "passive": d.qfrc_passive[e.arm_dof_adr].copy()}

    def payload_torque(self, qpos) -> np.ndarray | None:
        """Joint torque of the object's weight if it hung rigidly from the arm: J_com(object)^T (0, 0, -m g),
        on the plant model (simulation truth, not available to the controller; quasi-static, no object inertia)."""
        m, d, e = self.engine.model, self.d_plant, self.engine
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        if b < 0:
            return None
        self.set(d, qpos)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, d.xipos[b], m.body("hand").id)  # point on the hand at the object's COM
        return jacp[:, e.arm_dof_adr].T @ np.array([0.0, 0.0, -m.body_mass[b] * G])


def scratch_for(rec: Recording) -> Scratch:
    key = rec.meta["run_id"]
    with _cache_lock:
        s = _cache.get(key)
        if s is None:
            if len(_cache) > 6:
                _cache.clear()
            s = _cache[key] = Scratch(RunRequest.from_dict(rec.meta["request"]))
    return s


# ------------------------------------------------------------------ helpers


def _need(rec):
    if rec is None or not len(rec.t):
        raise RequestError("no recorded data yet: run a scenario or open a saved run")


def _at(rec: Recording, t) -> int:
    return rec.index(float(t)) if t not in (None, "", "null", "undefined") else len(rec.t) - 1


def _window(rec, i, before=1.5, after=1.5):
    t = rec.t
    return np.flatnonzero((t >= t[i] - before) & (t <= t[i] + after))


def _v(x, digits=4):
    return None if x is None or not np.isfinite(x) else float(np.round(x, digits))


def _len(mm: float) -> str:
    return f"{mm:.3f} mm" if mm >= 0.1 else f"{1e3 * mm:.2f} um"


def _ang(rad: float) -> str:
    return f"{1e3 * rad:.3f} mrad" if rad >= 1e-4 else f"{1e6 * rad:.2f} urad"


def _header(rec, i) -> dict:
    d = rec.data
    return {"type": "", "text": [f"Recorded sample at t = {rec.t[i]:.3f} s (samples every {1e3 * rec.meta['sample_dt']:.0f} ms; "
                                 f"physics step {1e3 * rec.meta['physics_dt']:.0f} ms), phase '{d['phase'][i]}', run "
                                 f"{rec.meta['run_id']}."]}


# ------------------------------------------------------------------ inspectors


def inspect(rec: Recording | None, kind: str, t, params: dict) -> dict:
    _need(rec)
    joint = int(params.get("joint", 0) or 0)
    i = _at(rec, t)
    fn = {"motion": motion, "mpc": mpc, "dynamics": dynamics, "planning": planning}.get(kind)
    if fn is None:
        raise RequestError(f"unknown inspector '{kind}'")
    return {"sections": fn(rec, i, joint), "t": float(rec.t[i])}


def motion(rec: Recording, i: int, j: int) -> list[dict]:
    d = rec.data
    q, qr, qd, qdr = d["q"][i], d["q_ref"][i], d["qd"][i], d["qd_ref"][i]
    ee, eer = d["ee"][i], d["ee_ref"][i]
    sc = scratch_for(rec)
    with sc.lock:
        p0, J = sc.fk_site(d["qpos"][i])
        dq = 0.01
        qpos1 = d["qpos"][i].copy()
        qpos1[sc.engine.arm_qpos_adr[j]] += dq
        p1, _ = sc.fk_site(qpos1)
    pred = J[:, j] * dq
    actual = p1 - p0
    w = _window(rec, i)
    return [
        _header(rec, i),
        {"type": "intuition", "title": "Joint angles decide where the hand is",
         "text": ["Forward kinematics (FK): seven joint angles q fix every link's pose, so they fix the hand. "
                  "Inverse kinematics (IK) runs the other way: for a desired hand pose, find joint angles that "
                  "produce it (the planner's IK solved this for the targets of this run).",
                  "For a small joint change the hand moves almost linearly: the Jacobian J(q) maps joint "
                  "velocities to hand velocities."]},
        {"type": "equation", "eq": "p = FK(q) &nbsp;&nbsp;·&nbsp;&nbsp; Δp ≈ J(q) Δq &nbsp;&nbsp;·&nbsp;&nbsp; ṗ = J(q) q̇",
         "text": "p: end-effector (between the fingertips) position in the world frame (m); q: joint angles (rad); "
                 "J: 3 x 7 positional Jacobian (m/rad)."},
        {"type": "values", "title": "Measured vs reference at this moment (recorded)",
         "header": ["joint", "q meas (rad)", "q ref (rad)", "q̇ meas (rad/s)", "q̇ ref (rad/s)"],
         "table": [[f"{k + 1}{' ◀' if k == j else ''}", _v(q[k]), _v(qr[k]), _v(qd[k]), _v(qdr[k])] for k in range(7)]},
        {"type": "values", "title": "Hand position (FK of the recorded joints, plant model)",
         "header": ["", "x (m)", "y (m)", "z (m)"],
         "table": [["measured FK(q)", *[_v(v) for v in ee]], ["reference FK(q_ref)", *[_v(v) for v in eer]],
                   ["difference (mm)", *[_v(1e3 * v, 3) for v in ee - eer]]]},
        {"type": "consequence", "title": f"Local approximation check for joint {j + 1} (scratch calculation)",
         "text": [f"Δq = {dq} rad on joint {j + 1} only, from the recorded pose. Jacobian prediction J Δq vs. "
                  f"forward kinematics of the perturbed pose on a private copy of the model (the live robot is "
                  f"not moved). The difference is the second-order (curvature) term the linearization drops."],
         "header": ["", "Δx (mm)", "Δy (mm)", "Δz (mm)", "|Δp| (mm)"],
         "table": [["J(q) Δq", *[_v(1e3 * v, 3) for v in pred], _v(1e3 * np.linalg.norm(pred), 3)],
                   ["FK(q+Δq) − FK(q)", *[_v(1e3 * v, 3) for v in actual], _v(1e3 * np.linalg.norm(actual), 3)],
                   ["difference", *[_v(1e3 * v, 4) for v in actual - pred], _v(1e3 * np.linalg.norm(actual - pred), 4)]]},
        {"type": "", "title": f"Joint {j + 1} around this moment",
         "plot": {"title": f"joint {j + 1} (rad): measured (blue) vs reference (grey)", "cursor": float(rec.t[i]),
                  "series": [{"x": rec.t[w].tolist(), "y": d["q_ref"][w, j].tolist(), "color": "#c9d1d9", "dash": [4, 3], "label": "reference"},
                             {"x": rec.t[w].tolist(), "y": d["q"][w, j].tolist(), "color": "#4aa3ff", "label": "measured"}]}},
    ]


def objective_terms(snap: dict, w: dict) -> dict:
    """The torque MPC's documented cost evaluated at the accepted plan (from the snapshot), term by term,
    plus the constant that the solver's algebraic objective omits. J_solver = J_total - constant."""
    X = np.asarray(snap["states"])
    R = np.asarray(snap["references"])
    tau = np.asarray(snap["commands"])
    N = len(X)
    s = np.asarray(w["tau_scale"])
    qx = np.r_[w["position"], w["velocity"]]
    stage = np.ones(N)
    stage[-1] = w["terminal"]
    hold = np.asarray(snap["gravity"]) - np.asarray(snap["disturbance"])
    prev = np.asarray(snap["tau_prev"])
    track = float(np.sum(stage[:, None] * qx * (X - R) ** 2))
    effort = float(np.sum(np.asarray(w["effort"]) * ((tau - hold) / s) ** 2))
    dtau = np.diff(np.vstack([prev, tau]), axis=0)
    rate = float(np.sum(np.asarray(w["torque_rate"]) * (dtau / s) ** 2))
    S = np.asarray(snap["slack"]) if snap.get("slack") is not None else np.zeros_like(X)
    slack = float(w["slack_quadratic"] * np.sum(S ** 2) + w["slack_linear"] * np.sum(S))
    const = float(np.sum(stage[:, None] * qx * R ** 2) + N * np.sum(np.asarray(w["effort"]) * (hold / s) ** 2)
                  + np.sum(np.asarray(w["torque_rate"]) * (prev / s) ** 2))
    total = track + effort + rate + slack
    return {"tracking": track, "effort": effort, "rate": rate, "slack": slack, "total": total,
            "constant": const, "solver_objective": total - const}


def mpc(rec: Recording, i: int, j: int) -> list[dict]:
    d, meta = rec.data, rec.meta
    w = meta.get("mpc_weights")
    if w is None:
        return [_header(rec, i), {"type": "warn", "text": f"This run used {meta['controller_class']}, not the torque MPC: "
                                                         "there are no MPC updates to inspect."}]
    t = float(rec.t[i])
    snap = rec.snapshot_at(t)
    if snap is None:
        return [_header(rec, i), {"type": "warn", "text": "No MPC update recorded at or before this time "
                                                         "(snapshots disabled, or before the first update)."}]
    Ts, N = meta["control_period"], w["horizon"]
    age = t - snap["time"]
    secs = [_header(rec, i), {
        "type": "intuition", "title": "What the MPC does every 20 ms",
        "text": ["At each update the controller measures the arm state, predicts the next 0.5 s with a linear model "
                 "of MuJoCo's dynamics, and chooses 25 torque vectors that best follow the reference preview without "
                 "breaking limits. It applies only the first one, holds it for one interval, then re-plans.",
                 "It knows joint limits, speed limits and torque limits. It does NOT know about obstacles or the "
                 "object: collision avoidance comes from the planned reference it is asked to track."]}]
    status_rows = [["update time", snap["time"], "s"], ["selected moment", t, "s"],
                   ["snapshot age (selected − update)", age, "s"], ["control interval Ts", Ts, "s"],
                   ["horizon", f"{N} steps = {N * Ts:.2f} s", ""], ["solver status", snap["status"], ""],
                   ["iterations", snap["iterations"], ""], ["solve / whole update time", f"{snap['solve_ms']:.2f} / {snap['update_ms']:.2f}", "ms (wall clock)"],
                   ["primal / dual residual", f"{snap['prim_res']:.1e} / {snap['dual_res']:.1e}", ""],
                   ["max slack (soft constraints)", snap["max_slack"], ""],
                   ["fallback (CTC) after this update", "yes" if snap["fallback"] else "no", ""]]
    secs.append({"type": "values", "title": "Selected optimization update (recorded snapshot)", "table": status_rows})
    if not snap["has_prediction"]:
        secs.append({"type": "warn", "text": "This update did not produce an accepted plan (solver failure); the CTC "
                                             "fallback produced the torques until the next successful update."})
        return secs
    X = np.asarray(snap["states"])
    R = np.asarray(snap["references"])
    tau = np.asarray(snap["commands"])
    scale = np.asarray(w["tau_scale"])
    k_t = snap["time"] + Ts * np.arange(1, N + 1)
    obs_w = np.flatnonzero((rec.t >= snap["time"]) & (rec.t <= snap["time"] + N * Ts))
    past = np.flatnonzero((rec.t >= snap["time"] - 0.5) & (rec.t <= snap["time"]))
    secs.append({"type": "", "title": f"Joint {j + 1}: prediction made at t = {snap['time']:.3f} s",
                 "plot": {"title": f"joint {j + 1} (rad): past measured · predicted · reference preview · later measured",
                          "cursor": [{"x": snap["time"], "color": "#c678dd"}, {"x": t, "color": "#e69a2e"}],
                          "series": [
                              {"x": rec.t[past].tolist(), "y": d["q"][past, j].tolist(), "color": "#4aa3ff", "label": "measured (before)"},
                              {"x": [snap["time"]] + k_t.tolist(), "y": [snap["x0"][j]] + X[:, j].tolist(), "color": "#c678dd", "label": "MPC prediction", "points": True},
                              {"x": k_t.tolist(), "y": R[:, j].tolist(), "color": "#c9d1d9", "dash": [4, 3], "label": "reference preview"},
                              {"x": rec.t[obs_w].tolist(), "y": d["q"][obs_w, j].tolist(), "color": "#3ccf7a", "label": "measured (after)"}]},
                 "text": "Purple dots are the model's predicted states at the knots (every Ts); green is what the "
                         "simulated arm actually did afterwards (while later updates re-planned). The gap grows "
                         "along the horizon: one linear model is used for all 25 steps (v0.6 audit: 1.6-5.3 mrad "
                         "after 0.5 s in normal motion)."})
    secs.append({"type": "", "plot": {"title": f"joint {j + 1}: planned torque sequence (Nm, held for Ts each)",
                                      "series": [{"x": (snap["time"] + Ts * np.arange(N)).tolist(), "y": tau[:, j].tolist(),
                                                  "color": "#c678dd", "points": True, "label": "τ_k"}]}})
    terms = objective_terms(snap, w)
    secs.append({"type": "equation", "title": "The implemented objective",
                 "eq": "J = Σ<sub>k=1..N</sub> w<sub>k</sub>‖x<sub>k</sub> − r<sub>k</sub>‖²<sub>Q</sub> + Σ<sub>k=0..N−1</sub> w<sub>e</sub>‖(τ<sub>k</sub> − τ<sub>hold</sub>)/τ<sub>max</sub>‖² "
                       "+ w<sub>r</sub>‖(τ<sub>k</sub> − τ<sub>k−1</sub>)/τ<sub>max</sub>‖² + ρ<sub>q</sub>‖s‖² + ρ<sub>l</sub>Σs",
                 "text": [f"Q = diag(position {w['position'][0]:g} per rad², velocity {w['velocity'][0]:g} per (rad/s)²), "
                          f"terminal stage x{w['terminal']:g}; effort w_e = {w['effort'][0]:g}, rate w_r = {w['torque_rate'][0]:g} on "
                          f"normalized torque (τ/τ_max, τ_max = 87 Nm joints 1-4, 12 Nm joints 5-7); slack ρ_q = "
                          f"{w['slack_quadratic']:g}, ρ_l = {w['slack_linear']:g}.",
                          "τ_hold = gravity torque − d̂ (the torque that would hold the arm still against gravity and "
                          "the estimated disturbance); τ_{-1} is the torque actually applied before this update."]})
    secs.append({"type": "values", "title": "Objective terms at the accepted plan (computed from the snapshot)",
                 "header": ["term", "value"],
                 "table": [["tracking (Q, terminal weight)", terms["tracking"]], ["effort above hold torque", terms["effort"]],
                           ["torque change", terms["rate"]], ["slack penalties", terms["slack"]],
                           ["total documented cost J", terms["total"]],
                           ["constant omitted by the solver (r'Qr, hold and previous-torque terms)", terms["constant"]],
                           ["solver objective ½z'Pz + q'z = J − constant", terms["solver_objective"]]],
                 "text": "The solver minimizes the same function without its constant part, so its objective value "
                         "differs from J by exactly the constant shown (checked in tests/test_workbench_math.py)."})
    qb, vb = np.asarray(w["q_bounds"]), np.asarray(w["v_bounds"])
    tl = np.asarray(w["torque_limits"])
    q_margin = np.minimum(X[:, :7] - qb[:, 0], qb[:, 1] - X[:, :7]).min(axis=0)
    v_margin = np.minimum(X[:, 7:] - vb[:, 0], vb[:, 1] - X[:, 7:]).min(axis=0)
    t_margin = np.minimum(tau - tl[:, 0], tl[:, 1] - tau).min(axis=0)
    secs.append({"type": "values", "title": "Constraints and their smallest margins over the horizon",
                 "text": ["Hard: torque limits on every planned torque τ_0..τ_{N−1} (each held for Ts, so the limit holds "
                          "at every instant). Soft (slack allowed, heavily penalized): joint position within the limits "
                          "minus 0.02 rad and |q̇| ≤ 0.995 v_max, imposed only at the 25 knots, not between them, and "
                          "only on the linear model's prediction. No collision constraints."],
                 "header": ["joint", "position margin (rad)", "velocity margin (rad/s)", "torque margin (Nm)"],
                 "table": [[f"{k + 1}{' ◀' if k == j else ''}", _v(q_margin[k]), _v(v_margin[k]), _v(t_margin[k], 3)] for k in range(7)]})
    cmd = tau[0]
    feed = d["feedthrough"][i]
    rows = [["optimizer τ_0 (physical)", *[_v(v, 3) for v in cmd]], ["optimizer τ_0 / τ_max (normalized)", *[_v(v, 4) for v in cmd / scale]]]
    if np.all(np.isfinite(feed)):
        rows.append(["observer feedthrough −(d̂_now − d̂_update)", *[_v(v, 3) for v in feed]])
    rows += [["command sent to the motors (ctrl, after clipping)", *[_v(v, 3) for v in d["ctrl"][i]]],
             ["torque the actuators produced", *[_v(v, 3) for v in d["tau"][i]]]]
    secs.append({"type": "values", "title": "Command path (Nm unless stated)",
                 "header": ["", *[f"j{k + 1}" for k in range(7)]], "table": rows,
                 "text": f"The optimizer's τ_0 comes from the update at {snap['time']:.3f} s; the other rows are the "
                         f"recorded sample at {t:.3f} s (age {1e3 * age:.0f} ms). Between updates the torque is held, "
                         f"corrected every 2 ms by the change of the observer's estimate (feedthrough), then clipped to "
                         f"the actuator limits."})
    secs.append({"type": "detail", "title": "Expanded: full predicted states and torques",
                 "details": {"summary": "show the matrices of this update",
                             "text": "k  t(s)   " + "  ".join(f"q{k + 1:<7}" for k in range(7)) + "\n" + "\n".join(
                                 f"{k + 1:<2} {k_t[k]:.3f}  " + "  ".join(f"{v:+.5f}" for v in X[k, :7]) for k in range(N))
                             + "\n\nk  " + "  ".join(f"tau{k + 1:<6}" for k in range(7)) + "\n" + "\n".join(
                                 f"{k:<2} " + "  ".join(f"{v:+8.3f}" for v in tau[k]) for k in range(N))}})
    return secs


def dynamics(rec: Recording, i: int, j: int) -> list[dict]:
    d, meta = rec.data, rec.meta
    req = RunRequest.from_dict(meta["request"])
    sc = scratch_for(rec)
    with sc.lock:
        nom = sc.nominal_terms(d["qpos"][i], d["qd"][i])
        pay = sc.payload_torque(d["qpos"][i])
    i0, i1 = max(i - 1, 0), min(i + 1, len(rec.t) - 1)
    qdd = (d["qd"][i1] - d["qd"][i0]) / (rec.t[i1] - rec.t[i0]) if i1 > i0 else np.full(7, np.nan)
    tau = d["tau"][i]
    dhat = d["d_hat"][i]
    obs_on = np.all(np.isfinite(dhat))
    resid = nom["M"] @ qdd + nom["h"] - nom["passive"] - tau
    held = d["phase"][i] in ("lift", "hold", "transport", "lower")
    w = _window(rec, i, 3, 3)
    payload = req.payload_kg if req.spec.kind == "pick_place" else None
    secs = [_header(rec, i),
            {"type": "intuition", "title": "Torque = inertia × acceleration + gravity and motion terms + whatever the model misses",
             "text": ["The motors must accelerate the links (inertia), hold them up against gravity, and overcome "
                      "velocity-dependent (Coriolis/centrifugal) and damping forces. Anything the controller's model "
                      "does not contain - here the carried object, pushes, model error - shows up as an extra torque d.",
                      "The controllers' nominal model contains the bare arm only: the object's weight never reaches "
                      "them through the model (v0.6 model boundary). Only the disturbance observer can estimate it, "
                      "from how the measured motion differs from what the commanded torque should have produced."]},
            {"type": "equation", "eq": "M(q) q̈ + h(q, q̇) = τ + d &nbsp;&nbsp;·&nbsp;&nbsp; d̂ ← filtered[ ID(q, q̇, q̈<sub>meas</sub>) − τ<sub>applied</sub> ], T = 10 ms",
             "text": "M: joint-space inertia (kg·m²); h: gravity + Coriolis/centrifugal + passive damping (Nm); τ: applied "
                     "motor torque; d: unknown disturbance torque; d̂: the observer's estimate (uses the measured state "
                     "and the torque actually applied, never the simulator's forces or masses)."},
            {"type": "values", "title": f"Joint {j + 1} at this moment",
             "header": ["quantity", "value", "source"],
             "table": [["M_jj (diagonal inertia)", _v(nom["M"][j, j], 4), "nominal model, recorded q (calculation)"],
                       ["gravity torque g_j(q)", _v(nom["g"][j], 3), "nominal model (calculation)"],
                       ["bias h_j(q, q̇)", _v(nom["h"][j], 3), "nominal model (calculation)"],
                       ["q̈_j (finite difference of recorded q̇, ±10 ms)", _v(qdd[j], 3), "recorded data (approximate)"],
                       ["applied torque τ_j", _v(tau[j], 3), "recorded (actuator force)"],
                       ["observer estimate d̂_j", _v(dhat[j], 3) if obs_on else "observer off", "recorded"],
                       ["residual M q̈ + h − τ (approx. d)", _v(resid[j], 2), "calculation (10 ms finite differences: noisy)"],
                       ["object weight torque J^T m g (if rigidly held)", _v(pay[j], 3) if pay is not None and held else
                        ("not held in this phase" if pay is not None else "no object"), "plant model, simulation truth"]]},
            ]
    if payload is not None:
        secs.append({"type": "values", "text": [f"Plant truth: object mass {payload:.3f} kg (in the simulator only). "
                                                 f"Controller's model: contact-free, nominal object that cannot load the arm."]})
    series = [{"x": rec.t[w].tolist(), "y": d["tau"][w, j].tolist(), "color": "#4aa3ff", "label": "applied τ"}]
    if obs_on:
        series.append({"x": rec.t[w].tolist(), "y": d["d_hat"][w, j].tolist(), "color": "#f2b33d", "label": "d̂ (estimate)"})
    secs.append({"type": "consequence", "title": "Estimate vs tracking error",
                 "plot": {"title": f"joint {j + 1}: torque and disturbance estimate (Nm)", "cursor": float(rec.t[i]), "series": series}})
    err = 1e3 * np.linalg.norm(d["ee"][w] - d["ee_ref"][w], axis=1)
    secs.append({"type": "", "plot": {"title": "end-effector tracking error (mm)", "cursor": float(rec.t[i]), "ymin": 0,
                                      "series": [{"x": rec.t[w].tolist(), "y": err.tolist(), "color": "#f2b33d", "label": "|FK(q) − FK(q_ref)|"}]},
                 "text": "Without the observer an unmodelled payload acts like a constant push: the controller's stiffness "
                         "settles at a deflection (the v0.6 study measured up to 6 mm at 1.15 kg with CTC). With the "
                         "observer, d̂ converges within a few time constants and the error returns near zero; compare "
                         "with 'compare observer'."})
    return secs


def planning(rec: Recording, i: int, j: int) -> list[dict]:
    res = rec.result or {}
    m = res.get("metrics", {})
    kind = rec.meta["kind"]
    secs = [_header(rec, i),
            {"type": "intuition", "title": "A path is not yet a motion",
             "text": ["The planner returns a geometric path: a list of joint configurations whose straight joint-space "
                      "edges were collision-checked with 20 mm clearance (green curve in the viewport: the hand's path). "
                      "A trajectory adds timing: each edge is played with a smooth rest-to-rest profile stretched until "
                      "speed, acceleration, jerk and torque limits hold. Pick-and-place approach/lift lines are straight "
                      "hand lines from pose IK (blue)."]},
            {"type": "equation", "eq": "path: q<sub>0</sub>, q<sub>1</sub>, …, q<sub>n</sub> &nbsp;→&nbsp; trajectory: q(t) = q<sub>i</sub> + (q<sub>i+1</sub> − q<sub>i</sub>) s(t), s = 10τ³ − 15τ⁴ + 6τ⁵",
             "text": "Edges are checked at samples so that no robot point moves more than 10 mm between samples; if every "
                     "sample keeps ≥ 20 mm, every configuration on the edge keeps ≥ 15 mm - for MuJoCo's geometry, exact "
                     "distance queries, static obstacles and exact tracking of the edge. It is a conditional bound, not a "
                     "continuous guarantee for the executed motion."}]
    rows = []
    for p in rec.paths:
        rows.append([p.get("phase", ""), p["kind"], len(p["path"])])
    secs.append({"type": "values", "title": "Paths used in this run", "header": ["phase / goal", "kind", "vertices"], "table": rows or [["none recorded", "", ""]]})
    if kind == "pick_place":
        ph = m.get("phases") or []
        secs.append({"type": "values", "title": "Three separate outcomes (recorded)",
                     "table": [["task completion", res.get("outcome", "running")],
                               ["physics steps with forbidden contacts (checked every 2 ms)", m.get("forbidden_contact_steps", UNAVAILABLE)],
                               ["min robot–environment distance (measured every 10 ms)", f"{m['min_clearance_robot_env_mm']:.1f} mm" if m.get("min_clearance_robot_env_mm") is not None else UNAVAILABLE],
                               ["min carried-object–environment distance (10 ms; hold/transport)", f"{m['min_clearance_object_env_mm']:.1f} mm" if m.get("min_clearance_object_env_mm") is not None else UNAVAILABLE],
                               ["requested planning clearance", "20 mm at checked samples"],
                               ["measured ≥ 20 mm?", ("yes" if (m.get("min_clearance_robot_env_mm") or 0) >= 20 and (m.get("min_clearance_object_env_mm") or 99) >= 20 else "no")
                                if m.get("min_clearance_robot_env_mm") is not None else UNAVAILABLE]]})
        if ph:
            secs.append({"type": "values", "title": "Per phase", "header": ["phase", "t start (s)", "max EE err (mm)", "min robot dist (mm)", "forbidden steps"],
                         "table": [[p["phase"], p["t_start"], _v(1e3 * p["ee_err_max"], 2), p.get("min_clear_robot_mm"), p["forbidden_steps"]] for p in ph]})
        secs.append({"type": "warn", "text": "Distances are finite samples (every 10 ms) of the simulated motion: they are "
                                             "measurements, not a continuous collision guarantee. Tracking error subtracts "
                                             "from the planned clearance (v0.6: 14.8 mm measured without the observer at 1.15 kg)."})
    else:
        secs.append({"type": "values", "title": "Outcomes (recorded)",
                     "table": [["task completion", res.get("outcome", "running")],
                               ["physics steps with robot contacts", m.get("robot_contact_steps", UNAVAILABLE)],
                               ["executed clearance", "not measured in reaching runs (unavailable)"]]})
    return secs


# ------------------------------------------------------------------ explain this moment


def explain(rec: Recording | None, t) -> dict:
    _need(rec)
    i = _at(rec, t)
    d, meta = rec.data, rec.meta
    req = RunRequest.from_dict(meta["request"])
    ti = float(rec.t[i])
    phase = d["phase"][i]
    goals = {"approach": "move above the object (planned, collision-free path)", "descend": "lower the open gripper straight down around the object",
             "close": "close the fingers on the object", "lift": "lift the object 100 mm straight up", "hold": "hold it still for 1 s to verify the grasp",
             "transport": "carry it to above the target (planned with the object attached)", "lower": "lower it straight down onto the table",
             "release": "open the fingers", "retreat": "move the hand straight up", "verify": "let the object settle and measure its final pose",
             "move": "follow the planned path to the next target", "hold_reach": "hold the target"}
    lines = [f"t = {ti:.3f} s (recorded sample; the selected time was {float(t) if t not in (None, '') else ti:.3f} s). "
             f"Phase '{phase}': {goals.get(phase, goals.get(phase + '_reach', '-'))}."]
    qe = np.abs(d["q"][i] - d["q_ref"][i])
    jmax = int(np.argmax(qe))
    ee = 1e3 * np.linalg.norm(d["ee"][i] - d["ee_ref"][i])
    lines.append(f"Measured vs reference: hand {_len(ee)} apart (FK of measured vs reference joints); largest joint "
                 f"error {_ang(qe[jmax])} on joint {jmax + 1}.")
    ctrl = meta["controller_class"]
    if d["fallback"][i]:
        lines.append(f"Active controller: CTC fallback (the {ctrl}'s last update failed).")
    else:
        lines.append(f"Active controller: {ctrl}" + (" (torque MPC; its own plan, no fallback)" if "TorqueMPC" in ctrl else "") + ".")
    snap = rec.snapshot_at(ti)
    if snap is not None and snap.get("has_prediction"):
        lines.append(f"Latest MPC update at {snap['time']:.3f} s ({1e3 * (ti - snap['time']):.0f} ms before): {snap['status']}, "
                     f"{snap['iterations']} iterations, solve {snap['solve_ms']:.1f} ms; planned τ_0 on joint {jmax + 1} "
                     f"{np.asarray(snap['commands'])[0, jmax]:.2f} Nm.")
    elif "TorqueMPC" in ctrl:
        lines.append("MPC snapshot: unavailable at this time.")
    dh = d["d_hat"][i]
    if np.all(np.isfinite(dh)):
        k = int(np.argmax(np.abs(dh)))
        lines.append(f"Observer: largest estimated disturbance {dh[k]:.2f} Nm on joint {k + 1} (applied torque there "
                     f"{d['tau'][i][k]:.2f} Nm).")
    else:
        lines.append("Observer: off in this run (no disturbance estimate).")
    tl = np.asarray(meta["torque_limits"])
    sat = np.abs(d["tau"][i]) >= 0.999 * tl
    lines.append("Torque limits: " + ("SATURATED on joints " + ", ".join(str(k + 1) for k in np.flatnonzero(sat)) if sat.any()
                                      else f"no joint at its limit (largest use {np.max(np.abs(d['tau'][i]) / tl) * 100:.0f}%)."))
    jl = np.asarray(meta["joint_limits"])
    jm = np.minimum(d["q"][i] - jl[:, 0], jl[:, 1] - d["q"][i])
    lines.append(f"Joint-limit margin: smallest {jm.min():.3f} rad (joint {int(np.argmin(jm)) + 1}).")
    if np.isfinite(d["clearance"][i]):
        c = d["clearance"][i]
        lines.append("Clearance: smallest robot–environment distance measured so far in this phase "
                     + (f"{c:.1f} mm" if c < 99.9 else ">= 100 mm (measurement cap)") + " (sampled every 10 ms).")
    else:
        lines.append("Clearance: not measured in this run type (unavailable).")
    if req.spec.kind == "pick_place":
        lines.append(f"Object: centre at z = {d['obj_pos'][i][2]:.4f} m, gripper opening {1e3 * d['grip_width'][i]:.1f} mm.")
    if np.any(d["force"][i]):
        lines.append(f"External force pulse active: {np.linalg.norm(d['force'][i]):.1f} N on "
                     f"{req.force_pulse['body']} (applied to the plant only).")
    if d["forbidden"][i]:
        lines.append("A FORBIDDEN CONTACT occurred within this 10 ms sample.")
    recent = [e for e in rec.events if ti - 1.0 <= e["t"] <= ti + 1e-9 and e["kind"] not in ("phase",)]
    for e in recent[-3:]:
        lines.append(f"Event at {e['t']:.3f} s: {e['text']}")
    if rec.result and abs(ti - float(rec.t[-1])) < 0.02:
        r = rec.result
        lines.append(f"Run outcome: {r['outcome']}" + (f" at {r['failure_stage']}: {r['failure_reason']}" if r.get("failure_reason") else "") + ".")
    return {"t": ti, "lines": lines, "sections": [
        {"type": "", "title": f"Explain this moment (t = {ti:.3f} s)", "text": lines},
        {"type": "warn", "text": "Every number above is a recorded value or a named calculation from recorded values. "
                                 "Data not recorded for this run is reported as unavailable."}]}


# ------------------------------------------------------------------ learning sandbox (one joint)


def sandbox(params: dict) -> dict:
    from workbench.sandbox import simulate
    r = simulate(**{k: float(v) for k, v in params.items()})
    t = r["t"]
    return {"sections": [
        {"type": "intuition", "title": "Accelerate, then brake",
         "text": ["To move a wheel of inertia I to a new angle quickly, push hard first, then push the other way to stop "
                  "on target. With a torque limit the push is capped, so the move takes longer; the horizon decides how "
                  "far ahead the controller sees the need to brake."]},
        {"type": "equation", "eq": "θ̈ = τ / I &nbsp;·&nbsp; x = [θ, ω], x<sub>k+1</sub> = A x<sub>k</sub> + B τ<sub>k</sub> (exact for a held torque) "
                                    "&nbsp;·&nbsp; J = Σ q<sub>θ</sub>(θ<sub>k</sub> − θ*)² + q<sub>ω</sub>ω<sub>k</sub>² + w<sub>e</sub>(τ<sub>k</sub>/τ<sub>max</sub>)²",
         "text": f"Solved each 0.1 s by the project's TorqueMPCController (same code as on the arm) with A = {np.round(r['A'], 4).tolist()}, "
                 f"B = {np.round(r['B'], 5).tolist()}. Effort is weighted on τ/τ_max, as in the arm controller."},
        {"type": "values", "title": "First optimization (t = 0)",
         "table": [["first planned torque τ_0", r["tau0"], "Nm"], ["unconstrained first torque (same cost, no limit)", r["tau0_unconstrained"], "Nm"],
                   ["time to reach within 1% of the target", r["settle_time"], "s"], ["overshoot", r["overshoot"], "rad"],
                   ["torque at the limit (fraction of steps)", r["saturated_fraction"], ""]]},
        {"type": "consequence", "plot": {"title": "angle θ (rad) and target", "series": [
            {"x": t, "y": r["theta"], "color": "#4aa3ff", "label": "θ"}, {"x": [t[0], t[-1]], "y": [r["target"]] * 2, "color": "#c9d1d9", "dash": [4, 3], "label": "θ*"}]}},
        {"type": "consequence", "plot": {"title": "torque τ (Nm): acceleration then braking", "series": [
            {"x": t[:-1], "y": r["tau"], "color": "#c678dd", "label": "τ (held 0.1 s)", "points": True}]}},
        {"type": "", "plot": {"title": "first plan: predicted θ over the horizon (rad)", "series": [
            {"x": r["plan_t"], "y": r["plan_theta"], "color": "#c678dd", "points": True, "label": "prediction"}]}},
    ]}


# ------------------------------------------------------------------ workspace scan (bounded, private models)


def reach_scan(request: RunRequest, step: float = 0.03) -> dict:
    """Classify candidate object positions on a grid over the supported table region for the prescribed top-down
    grasp: 'found' (pose IK converged for grasp and pre-grasp and both pass the collision checks of the descend
    phase), 'blocked' (IK converged but a collision check failed), 'not_found' (the IK search from two seeds did
    not converge - not a proof of physical unreachability). Uses private scratch models only."""
    from planning.collision import CollisionChecker
    from simulation.kinematics import ArmKinematics
    from tasks.pick_place import top_down_rotation
    from workbench.api import TABLE_XY
    from workbench.session import scene_engine
    if request.spec.kind != "pick_place":
        raise RequestError("the reachability scan is for the pick-and-place scenarios")
    e = scene_engine(request)
    kin = ArmKinematics(e)
    chk = CollisionChecker.from_engine(e)
    m = e.model
    obj_adr = m.jnt_qposadr[m.joint("object_free").id]
    gripper = chk.body_geoms("hand", "left_finger", "right_finger")
    obj = chk.body_geoms("object")
    R = top_down_rotation(0.0)
    z = 0.40 + 0.03
    xs = np.arange(TABLE_XY["x"][0], TABLE_XY["x"][1] + 1e-9, step)
    ys = np.arange(TABLE_XY["y"][0], TABLE_XY["y"][1] + 1e-9, step)
    points, counts = [], {"found": 0, "blocked": 0, "not_found": 0}
    seed_prev = kin.q_rest
    for x in xs:
        for y in ys:
            qpos = e.data.qpos.copy()
            qpos[obj_adr:obj_adr + 3] = [x, y, z]
            chk.set_fixed_qpos(qpos)
            chk.clear_allowed()
            sols = []
            for p in ([x, y, z + 0.115], [x, y, z + 0.015]):
                r = None
                for seed in (seed_prev, kin.q_rest):
                    r = kin.solve_pose_ik(np.array(p), R, seed, max_iter=300)
                    if r.success:
                        break
                sols.append(r)
            if not all(r.success for r in sols):
                cls, why = "not_found", "IK search did not converge"
            else:
                seed_prev = sols[0].q
                pre_ok = chk.check(sols[0].q)
                chk.allow(gripper, obj, 0.005)
                grasp_ok = chk.check(sols[1].q)
                cls = "found" if pre_ok.valid and grasp_ok.valid else "blocked"
                why = "" if cls == "found" else (pre_ok.reason or grasp_ok.reason)
            counts[cls] += 1
            points.append({"x": float(x), "y": float(y), "class": cls, "reason": why})
    return {"step": step, "points": points, "counts": counts,
            "note": "Top-down grasp only; 'not_found' means the IK search from two seeds failed, not that the pose "
                    "is physically unreachable. Planning and the full task can still fail at 'found' positions."}


# ------------------------------------------------------------------ pre-flight check of a request


def preflight(request: RunRequest) -> list[dict]:
    """Checks before running (on private scratch models): validation, IK reachability of the key poses,
    initial collisions. IK failure means 'the search did not find a solution', not physical unreachability."""
    from simulation.kinematics import ArmKinematics
    from tasks.pick_place import top_down_rotation
    from workbench.session import scene_engine
    out = []
    try:
        request.validate()
        out.append({"check": "settings valid and supported", "ok": True, "detail": "; ".join(request.summary()[:2])})
    except RequestError as exc:
        return [{"check": "settings valid and supported", "ok": False, "detail": str(exc)}]
    e = scene_engine(request)
    kin = ArmKinematics(e)
    from planning.collision import CollisionChecker
    chk = CollisionChecker.from_engine(e)
    start = np.asarray(request.resolve()["config"].get("pick_place", {}).get("start", e.robot.home_qpos), float) \
        if request.spec.kind == "pick_place" else np.asarray(request.resolve()["config"]["task"]["start"], float)
    rep = chk.check(start)
    out.append({"check": "start configuration collision-free (20 mm)", "ok": rep.valid, "detail": rep.reason or "ok"})
    if request.spec.kind == "pick_place":
        R = top_down_rotation(0.0)
        z = 0.40 + 0.03
        poses = {"pre-grasp (100 mm above)": [*request.object_xy, z + 0.115], "grasp": [*request.object_xy, z + 0.015],
                 "pre-place": [*request.place_xy, z + 0.117], "place": [*request.place_xy, z + 0.017]}
        for name, p in poses.items():
            best = None
            for seed in (kin.q_rest, start):
                r = kin.solve_pose_ik(np.array(p), R, seed, max_iter=400)
                if r.success:
                    best = r
                    break
            out.append({"check": f"pose IK for {name} at {np.round(p, 3).tolist()}", "ok": best is not None,
                        "detail": "solution found" if best else "search did not find a solution (not a proof of unreachability)"})
    else:
        for k, p in enumerate(request.resolved_targets()):
            r = kin.solve_ik(np.array(p), kin.q_rest)
            out.append({"check": f"position IK for target {k + 1} {p}", "ok": r.success,
                        "detail": "solution found (collision checks and planning happen at run time)" if r.success else
                        "search did not find a solution (not a proof of unreachability)"})
    return out
