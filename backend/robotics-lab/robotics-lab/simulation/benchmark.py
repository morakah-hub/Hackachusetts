"""Headless controller benchmark: identical scenarios, robot and initial state for every controller.

Run via benchmark.py at the repo root. All metrics are computed from the
simulated robot (every physics step), not from the controllers' predictions.
"""

import copy
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import osqp
import yaml

from controllers.dynamics import ArmInverseDynamics
from controllers.mpc import JointMPCController
from controllers.torque_mpc import TorqueMPCController
from simulation.engine import PROJECT_ROOT
from simulation.kinematics import ArmKinematics, rotation_angle
from simulation.runner import SimulationRunner, load_config
from tasks.reach import CartesianReachTask

BENCHMARK_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def scenario_config(base: dict, bench: dict, controller: str, scenario: str) -> dict:
    """Full runner config for one (controller, scenario) pair."""
    sc = bench["scenarios"][scenario]
    cfg = copy.deepcopy(base)
    cfg["environment"] = bench.get("environment", cfg.get("environment", "table"))
    cfg["gripper"] = bench.get("gripper", cfg.get("gripper", 255))
    if "targets" in sc:
        cfg["task"] = {
            "type": "cartesian_reach",
            "move_time": sc["move_time"],
            "hold_time": sc["hold_time"],
            "targets": sc["targets"],
            "ik": sc.get("ik", {}),
            "trajectory": sc.get("trajectory"),
        }
    else:
        cfg["task"] = {
            "type": "joint_target",
            "move_time": sc["move_time"],
            "hold_time": sc["hold_time"],
            "loop": True,
            "waypoints": sc["waypoints"],
            "trajectory": sc.get("trajectory"),
        }
    cfg["controller"] = deep_merge(base["controller"], bench["controllers"][controller])
    if "controller_model" in sc:
        cfg["controller_model"] = sc["controller_model"]
    if "sensor_noise" in sc:
        cfg["sensor_noise"] = sc["sensor_noise"]
    return cfg


@dataclass
class Disturbance:
    body: str
    force: np.ndarray
    start: float
    end: float

    def __call__(self, runner: SimulationRunner) -> None:
        t = runner.engine.time
        if self.start <= t < self.end:
            runner.engine.apply_external_force(self.body, self.force)
        else:
            runner.engine.clear_external_forces()


def settling_times(t: np.ndarray, err: np.ndarray, events: list[tuple[float, float]], tol: float) -> list[float]:
    """For each (start, end) window: time after start until the max joint error
    stays below tol for the rest of the window (nan if it never does)."""
    out = []
    for start, end in events:
        mask = (t >= start) & (t < end)
        if not mask.any():
            continue
        tw, ew = t[mask], err[mask]
        above = np.nonzero(ew >= tol)[0]
        if len(above) == 0:
            out.append(0.0)
        elif above[-1] == len(ew) - 1:
            out.append(float("nan"))
        else:
            out.append(float(tw[above[-1] + 1] - start))
    return out


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    if not mask.any():
        return 0
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return int(np.max(np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]))


def run_case(cfg: dict, scenario: dict, tol: float) -> tuple[dict, dict]:
    runner = SimulationRunner(cfg)
    engine, task = runner.engine, runner.task
    duration = scenario.get("duration", task.duration)
    dist = None
    if "disturbance" in scenario:
        d = scenario["disturbance"]
        dist = Disturbance(d["body"], np.asarray(d["force"], float), d["start"], d["end"])

    contact_steps = 0

    def before_step(r: SimulationRunner) -> None:
        nonlocal contact_steps
        if dist is not None:
            dist(r)
        if r.engine.data.ncon and r.engine.robot_contacts():
            contact_steps += 1

    wall = time.perf_counter()
    log = runner.run_headless(duration, log_every=1, before_step=before_step).arrays()
    wall = time.perf_counter() - wall

    dt = engine.dt
    t, q, q_ref, qd, force = log["time"], log["q"], log["q_ref"], log["qd"], log["force"]
    qd_ref, qdd_ref = log["qd_ref"], log["qdd_ref"]
    err = np.abs(q - q_ref)
    err_max = err.max(axis=1)
    lo, hi = engine.joint_limits[:, 0], engine.joint_limits[:, 1]
    vmax = np.asarray(engine.robot.velocity_limits)
    flim = engine.model.actuator_forcerange[engine.arm_actuator_ids, 1]
    finite = bool(np.all(np.isfinite(engine.data.qpos)) and np.all(np.isfinite(q)) and np.all(np.isfinite(qd)))

    # End-effector pose: FK(q) vs FK(q_ref), every 5th physics step. Both via
    # FK because MjData.site_xpos read after mj_step still holds the kinematics
    # from the start of that step (one step behind qpos).
    kin = ArmKinematics(engine)
    idx = np.arange(0, len(t), 5)
    poses = [kin.fk_pose(q[i]) for i in idx]
    ref_poses = [kin.fk_pose(q_ref[i]) for i in idx]
    ee_err = np.linalg.norm(np.array([p[0] - r[0] for p, r in zip(poses, ref_poses)]), axis=1)
    # Orientation. Tracking error: angle between the actual hand orientation and the one the
    # joint reference implies; only defined when the task specifies joint configurations
    # (joint_target). A position-only reach task has no orientation reference, so there we
    # report how much the orientation changes instead (never an invented reference).
    ori_err = np.degrees([rotation_angle(r[1], p[1]) for p, r in zip(poses, ref_poses)])
    ori_change = np.degrees([rotation_angle(poses[0][1], p[1]) for p in poses])
    is_reach = isinstance(task, CartesianReachTask)

    # Velocity limits on the simulated robot, every physics step.
    excess = np.maximum(np.abs(qd) - vmax, 0.0)
    violating = np.any(excess > 0.0, axis=1)
    # Reference feasibility: speed and acceleration against the datasheet limits, torque from
    # the plant's inverse dynamics against the actuator limits (every 5th step).
    amax = np.asarray(engine.robot.acceleration_limits)
    dyn = ArmInverseDynamics(engine.model, engine.data, engine.arm_qpos_adr, engine.arm_dof_adr)
    tau_ref = np.array([dyn.torque(q_ref[i], qd_ref[i], qdd_ref[i]) for i in idx])
    ref_ratios = {
        "reference_velocity_ratio": float(np.max(np.abs(qd_ref) / vmax)),
        "reference_acceleration_ratio": float(np.max(np.abs(qdd_ref) / amax)),
        "reference_torque_ratio": float(np.max(np.abs(tau_ref) / flim)),
    }

    events = [task.segment_bounds(k)[1:] for k in range(task.num_segments)
              if task.segment_bounds(k)[2] <= duration + 1e-9]
    # Orientation drift while holding still at a waypoint (orientation at the hold start as reference).
    ts = t[idx]
    drift = [0.0]
    for start, end in events:
        w = np.nonzero((ts >= start) & (ts < end))[0]
        if len(w) > 1:
            drift.append(max(np.degrees(rotation_angle(poses[w[0]][1], poses[i][1])) for i in w))
    if dist is not None:
        events = [(dist.end, duration)]
    settle = settling_times(t, err_max, events, tol)
    settled = [s for s in settle if np.isfinite(s)]

    m = {
        "actuation": engine.robot.actuation,
        "sim_time_s": float(engine.time),
        "rms_error_rad": float(np.sqrt(np.mean(err**2))),
        "max_error_rad": float(err.max()),
        "final_error_rad": float(err_max[-1]),
        "ee_rms_error_mm": float(1e3 * np.sqrt(np.mean(ee_err**2))),
        "ee_max_error_mm": float(1e3 * ee_err.max()),
        "ee_orientation_rms_error_deg": float("nan") if is_reach else float(np.sqrt(np.mean(ori_err**2))),
        "ee_orientation_max_error_deg": float("nan") if is_reach else float(ori_err.max()),
        "ee_orientation_change_max_deg": float(ori_change.max()),
        "ee_orientation_hold_drift_max_deg": float(max(drift)),
        "settling_time_mean_s": float(np.mean(settled)) if settled else float("nan"),
        "settling_time_max_s": float(np.max(settled)) if settled else float("nan"),
        "unsettled_events": int(len(settle) - len(settled)),
        "rms_torque_Nm": float(np.sqrt(np.mean(force**2))),
        "torque_squared_integral_Nm2s": float(np.sum(force**2) * dt),
        "joint_limit_violation_steps": int(np.sum(np.any((q < lo) | (q > hi), axis=1))),
        "min_joint_limit_margin_rad": float(np.min(np.minimum(q - lo, hi - q))),
        "velocity_limit_violation_steps": int(np.sum(violating)),
        "max_velocity_ratio": float(np.max(np.abs(qd) / vmax)),
        "velocity_violation_max_excess": float(excess.max()),
        "velocity_violation_longest_s": float(longest_run(violating) * dt),
        "velocity_violation_integral_rad": float(excess.sum() * dt),
        **ref_ratios,
        "reference_feasible": bool(max(ref_ratios.values()) <= 1.0 + 1e-9),
        "torque_saturation_fraction": float(np.mean(np.any(np.abs(force) >= flim - 1e-6, axis=1))),
        "robot_contact_steps": int(contact_steps),
        "finite_state": finite,
        "wall_time_s": float(wall),
        "realtime_factor": float(engine.time / wall),
    }
    if isinstance(task, CartesianReachTask):
        finals = []
        for p in task.reachable:
            i = np.searchsorted(t, task.hold_end_time(p)) - 1
            finals.append(np.linalg.norm(log["ee"][i] - p.position))
        m |= {
            "reach_targets": len(task.plans),
            "reach_rejected": len(task.rejected),
            "reach_final_ee_error_max_mm": float(1e3 * max(finals)) if finals else float("nan"),
        }
    ctrl = runner.controller
    if isinstance(ctrl, (JointMPCController, TorqueMPCController)):
        s = ctrl.stats.summary(ctrl.control_period)
        m |= {
            "control_rate_hz": 1.0 / ctrl.control_period,
            "mpc_updates": s["updates"],
            "mpc_fallbacks": s["fallbacks"],
            "mpc_solve_time_mean_ms": s["solve_time_mean_ms"],
            "mpc_solve_time_p95_ms": s["solve_time_p95_ms"],
            "mpc_solve_time_p99_ms": s["solve_time_p99_ms"],
            "mpc_update_time_mean_ms": s["update_time_mean_ms"],
            "mpc_update_time_p95_ms": s["update_time_p95_ms"],
            "mpc_update_time_p99_ms": s["update_time_p99_ms"],
            "mpc_update_time_max_ms": s["update_time_max_ms"],
            "deadline_overruns": s["deadline_overruns"],
            "mpc_mean_iterations": s["mean_iterations"],
            "mpc_inaccurate": s["inaccurate"],
            "mpc_max_slack": s["max_slack"],
            "mpc_max_predicted_velocity_ratio": s["max_predicted_velocity_ratio"],
            "mpc_max_predicted_position_excess_rad": s["max_predicted_position_excess"],
        }
    else:
        m |= {"control_rate_hz": 1.0 / dt, "deadline_overruns": 0}
    series = {"t": t, "err_max": err_max, "q": q, "q_ref": q_ref, "force": force,
              "v_ratio": np.max(np.abs(qd) / vmax, axis=1), "t_sub": ts,
              "ori_err": None if is_reach else ori_err, "ori_change": ori_change}
    return m, series


def run_benchmark(
    controllers: list[str] | None = None,
    scenarios: list[str] | None = None,
    out_dir: Path | None = None,
    plots: bool = True,
    bench_path: Path = BENCHMARK_CONFIG,
    base_config: dict | None = None,
    verbose: bool = True,
) -> dict:
    with open(bench_path, encoding="utf-8") as f:
        bench = yaml.safe_load(f)
    base = base_config or load_config()
    controllers = controllers or list(bench["controllers"])
    scenarios = scenarios or list(bench["scenarios"])
    tol = bench.get("metrics", {}).get("settle_tolerance", 0.002)

    results, all_series = [], {}
    for sc_name in scenarios:
        for c_name in controllers:
            cfg = scenario_config(base, bench, c_name, sc_name)
            metrics, series = run_case(cfg, bench["scenarios"][sc_name], tol)
            results.append({"scenario": sc_name, "controller": c_name, **metrics})
            all_series[(sc_name, c_name)] = series
            if verbose:
                print(
                    f"  {sc_name:12s} {c_name:15s} rms {metrics['rms_error_rad']:.5f} max {metrics['max_error_rad']:.4f} rad"
                    f"  wall {metrics['wall_time_s']:.1f}s",
                    flush=True,
                )

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "osqp": osqp.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "physics_dt": base.get("physics_dt", 0.002),
        "benchmark_config": bench,
        "results": results,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)
        keys = list(dict.fromkeys(k for r in results for k in r))
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        if plots:
            report["plots"] = save_plots(out_dir, all_series, results, scenarios, controllers)
    return report


def save_plots(out_dir: Path, series: dict, results: list[dict], scenarios: list[str], controllers: list[str]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return []

    palette = ["#2a6fdb", "#e0663d", "#2e9e6a", "#8a5cc2", "#b8860b"]
    color = {c: palette[i % len(palette)] for i, c in enumerate(controllers)}
    files = []

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.6 * len(scenarios)), squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios):
        for c in controllers:
            s = series[(sc, c)]
            ax.plot(s["t"], 1e3 * s["err_max"], lw=1.0, color=color[c], label=c)
        ax.set_title(sc, loc="left", fontsize=10)
        ax.set_ylabel("max |q - q_ref|\n(mrad)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=len(controllers), fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_error.png", dpi=120)
    plt.close(fig)
    files.append("tracking_error.png")

    metrics = [("rms_error_rad", "RMS joint error (rad)"), ("max_error_rad", "max joint error (rad)"),
               ("ee_max_error_mm", "max end-effector error (mm)"), ("max_velocity_ratio", "peak speed / limit"),
               ("rms_torque_Nm", "RMS torque (Nm)"), ("torque_saturation_fraction", "torque-saturated steps")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    axes = axes.ravel()
    x = np.arange(len(scenarios))
    w = 0.8 / len(controllers)
    for ax, (key, label) in zip(axes, metrics):
        for i, c in enumerate(controllers):
            vals = [next(r[key] for r in results if r["scenario"] == sc and r["controller"] == c) for sc in scenarios]
            ax.bar(x + (i - (len(controllers) - 1) / 2) * w, vals, w, color=color[c], label=c)
        ax.set_xticks(x, scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if key in ("rms_error_rad", "max_error_rad", "ee_max_error_mm"):
            ax.set_yscale("log")
        if key == "max_velocity_ratio":
            ax.axhline(1.0, color="0.3", lw=0.8, ls="--")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=120)
    plt.close(fig)
    files.append("summary.png")

    # Peak joint speed relative to its limit over time (constraint behaviour).
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.4 * len(scenarios)), squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios):
        for c in controllers:
            s = series[(sc, c)]
            ax.plot(s["t"], s["v_ratio"], lw=1.0, color=color[c], label=c)
        ax.axhline(1.0, color="0.3", lw=0.8, ls="--")
        ax.set_title(sc, loc="left", fontsize=10)
        ax.set_ylabel("max |qd| / v_max")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=len(controllers), fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_ratio.png", dpi=120)
    plt.close(fig)
    files.append("velocity_ratio.png")

    # End-effector orientation: tracking error (joint tasks) or change from the start (reach tasks).
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 2.4 * len(scenarios)), squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios):
        reach = series[(sc, controllers[0])]["ori_err"] is None
        for c in controllers:
            s = series[(sc, c)]
            y = s["ori_change"] if reach else s["ori_err"]
            ax.plot(s["t_sub"], y, lw=1.0, color=color[c], label=c)
        ax.set_title(f"{sc}: {'orientation change from start (no orientation reference)' if reach else 'orientation tracking error'}",
                     loc="left", fontsize=10)
        ax.set_ylabel("deg")
        if not reach:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=len(controllers), fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "orientation.png", dpi=120)
    plt.close(fig)
    files.append("orientation.png")
    return files
