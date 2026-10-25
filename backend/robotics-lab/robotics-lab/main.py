"""Robotics Lab entry point.

    python main.py                       # interactive MuJoCo viewer (controller from config)
    python main.py --controller mpc      # viewer, model predictive control
    python main.py --headless --duration 12 --controller pid
    python benchmark.py                  # headless PID vs MPC comparison
    python main.py --config configs/planning_demo.yaml                 # v0.5 obstacle-avoidance demo
    python main.py --config configs/planning_demo.yaml --no-planning   # same goal without planning
"""

import argparse

import numpy as np

from controllers.mpc import JointMPCController
from tasks.reach import CartesianReachTask
from controllers.torque_mpc import TorqueMPCController
from simulation.runner import SimulationRunner, load_config
from tasks.planned_joint import PlanningFailed
from tasks.trajectory import TrajectoryInfeasible


def main() -> None:
    parser = argparse.ArgumentParser(description="Robotics Lab: Franka Panda in MuJoCo")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--environment", default=None, help="Override environment: table | empty | obstacles")
    parser.add_argument("--headless", action="store_true", help="Run without the viewer and print a summary")
    parser.add_argument("--duration", type=float, default=None, help="Simulated seconds to run")
    parser.add_argument(
        "--controller",
        choices=["pid", "mpc", "ctc", "torque_mpc"],
        default=None,
        help="Override controller: pid/mpc use position servos, ctc/torque_mpc use torque motors",
    )
    parser.add_argument("--control-rate", type=float, default=None, help="MPC update rate in Hz (e.g. 50 or 100)")
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        action="append",
        metavar=("X", "Y", "Z"),
        help="Reach this end-effector position in metres (repeatable; replaces the configured task)",
    )
    plan = parser.add_mutually_exclusive_group()
    plan.add_argument("--planning", dest="planning", action="store_true", default=None,
                      help="Enable collision-aware motion planning (task.planning.enabled)")
    plan.add_argument("--no-planning", dest="planning", action="store_false",
                      help="Disable motion planning: straight joint moves, blocked targets are rejected")
    parser.add_argument("--plan-seed", type=int, default=None, help="Random seed of the motion planner")
    parser.add_argument("--unchecked-direct", action="store_true",
                        help="With --no-planning: skip the straight-path collision check (UNSAFE comparison mode; "
                             "the arm may drive into obstacles)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.environment:
        config["environment"] = args.environment
    if args.controller:
        config["controller"]["type"] = {"pid": "joint_pid", "mpc": "joint_mpc", "ctc": "torque_ctc",
                                        "torque_mpc": "torque_mpc"}[args.controller]
    if args.target:
        reach = config["task"] if config["task"].get("type") == "cartesian_reach" else {
            "move_time": 3.0, "hold_time": 1.0, "trajectory": config["task"].get("trajectory")}
        config["task"] = {
            **reach,
            "type": "cartesian_reach",
            "targets": [{"name": f"target{i + 1}", "position": t} for i, t in enumerate(args.target)],
        }
    if args.planning is not None or args.plan_seed is not None:
        planning = config["task"].setdefault("planning", {}) or {}
        config["task"]["planning"] = planning
        if args.planning is not None:
            planning["enabled"] = args.planning
        if args.plan_seed is not None:
            planning["seed"] = args.plan_seed
    if args.unchecked_direct:
        if (config["task"].get("planning") or {}).get("enabled"):
            raise SystemExit("--unchecked-direct only applies with planning disabled (add --no-planning)")
        config["task"].setdefault("ik", {})["check_path"] = False
        if (config["task"].get("trajectory") or {}).get("torque_scale") is not None:
            # The v0.4 torque check uses inverse dynamics, which includes contact forces: it would
            # reject a reference that penetrates an obstacle. Drop it so the collision can be observed.
            config["task"]["trajectory"]["torque_scale"] = None
        print("WARNING: straight-path collision check and torque check disabled (--unchecked-direct); "
              "the arm may hit obstacles.")
    if args.control_rate:
        section = "torque_mpc" if config["controller"]["type"] == "torque_mpc" else "mpc"
        config["controller"].setdefault(section, {})["control_rate_hz"] = args.control_rate
    try:
        runner = SimulationRunner(config)
    except TrajectoryInfeasible as exc:
        raise SystemExit(f"Infeasible trajectory: {exc}")
    except PlanningFailed as exc:
        raise SystemExit(f"Motion planning failed: {exc}")
    engine = runner.engine
    print(
        f"Loaded '{config['environment']}' environment: nq={engine.model.nq}, "
        f"nu={engine.model.nu}, dt={engine.dt}s, waypoints={runner.waypoint_names}"
    )
    print(f"Controller: {type(runner.controller).__name__} ({engine.robot.actuation} actuators)")
    if isinstance(runner.task, CartesianReachTask):
        for p in runner.task.plans:
            pos = np.round(p.position, 3).tolist()
            status = "reachable" if p.reachable else f"REJECTED ({p.reason})"
            print(f"  target {p.name} {pos}: {status}")
            if p.path is not None:
                print("    " + plan_summary(p.stats, len(p.path)))
    elif runner.planner is not None:
        for k, p in enumerate(runner.task.plans):
            print(f"  move {k}: " + plan_summary(p.stats, len(p.path)))
    for s in runner.task.timing_report():
        if s["limited_by"] != "requested":
            print(f"  move {s['from']} -> {s['to']}: {s['requested_s']:.2f} s requested, {s['move_s']:.2f} s used "
                  f"({s['limited_by']} limit of joint {s['limiting_joint']})")

    if not args.headless:
        print("Opening MuJoCo viewer. Close the window to exit.")
        runner.run_viewer(args.duration)
        return

    duration = args.duration or runner.task.duration
    contact_steps, contact_pairs = 0, set()

    def count_contacts(r) -> None:  # contacts of the simulated robot, every physics step
        nonlocal contact_steps
        found = r.engine.robot_contacts() if r.engine.data.ncon else []
        if found:
            contact_steps += 1
            contact_pairs.update(found)

    log = runner.run_headless(duration, before_step=count_contacts).arrays()
    err = np.abs(log["q"] - log["q_ref"]).max(axis=1)
    print(f"Simulated {engine.time:.2f}s ({len(log['time'])} samples)")
    print(f"Max |q - q_ref| tracking error: {err.max():.4f} rad")
    print(f"Final arm qpos: {np.round(engine.arm_qpos(), 4).tolist()}")
    print(f"Final end-effector position: {np.round(engine.ee_position(), 4).tolist()}")
    print(f"Robot contacts at end: {engine.robot_contacts() or 'none'}")
    print(f"Physics steps with robot contacts during the run: {contact_steps}"
          + (f" {sorted(contact_pairs)}" if contact_pairs else ""))
    if isinstance(runner.task, CartesianReachTask):
        for p in runner.task.reachable:
            ee = log["ee"][np.searchsorted(log["time"], runner.task.hold_end_time(p)) - 1]
            print(f"  {p.name}: end-effector error at end of hold {1e3 * np.linalg.norm(ee - p.position):.2f} mm")
    if isinstance(runner.controller, (JointMPCController, TorqueMPCController)):
        stats = runner.controller.stats.summary(runner.controller.control_period)
        print("MPC: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))


def plan_summary(st: dict, vertices: int) -> str:
    direct = st.get("direct_edge_valid")
    route = "direct edge free" if direct else "direct edge BLOCKED -> RRT-Connect"
    return (f"planned ({route}): {1e3 * st.get('total_time_s', st.get('planning_time_s', 0.0)):.0f} ms, "
            f"{st.get('nodes', 0)} nodes, {st.get('collision_checks', 0)} collision checks, "
            f"path {st.get('raw_length_rad', 0.0):.2f} -> {st.get('smoothed_length_rad', 0.0):.2f} rad, "
            f"{vertices} vertices, seed {st.get('seed')}")


if __name__ == "__main__":
    main()
