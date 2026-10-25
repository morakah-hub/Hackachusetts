r"""Known-world pick-and-place episodes (v0.6).

    .\.venv\Scripts\python.exe pick_place.py                                   # all development cases, headless
    .\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --controller torque_mpc
    .\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --observer off --mass 1.0
    .\.venv\Scripts\python.exe pick_place.py --case dev_free_1 --fault weak_grip      # negative control

Each episode prints its outcome, the failure stage/reason if any, and the measured metrics; --output
writes all results as JSON. See simulation/manipulation.py for the phases and checks.
"""

import argparse
import json
import sys
from pathlib import Path

from simulation.manipulation import run_case
from simulation.runner import load_config

CONFIG = Path(__file__).resolve().parent / "configs" / "pick_place.yaml"
CONTROLLERS = {"ctc": "torque_ctc", "torque_ctc": "torque_ctc", "torque_mpc": "torque_mpc", "mpc": "torque_mpc"}
FAULTS = {  # negative controls: each must make the verifier reject the episode
    "skip_close": {"skip_close": True},                       # gripper never closes
    # Menagerie gripper gains (2.5 N squeeze) with a 1.0 kg object: closes on it, cannot lift it.
    "weak_grip": {"robot": {"gripper_stiffness": 100.0, "gripper_damping": 10.0}, "mass": 1.0},
    "place_offset": {"place_offset": [0.03, 0.0]},            # object put down 30 mm from the target
    "release_during_transport": {"release_during_transport": True},
}


def configure(cfg: dict, controller: str | None, observer: str | None) -> dict:
    if controller:
        cfg["controller"]["type"] = CONTROLLERS[controller]
    if observer:
        on = observer == "on"
        cfg["controller"]["ctc"].setdefault("disturbance_observer", {})["enabled"] = on
        cfg["controller"]["torque_mpc"].setdefault("disturbance_observer", {})["enabled"] = on
    kind = cfg["controller"]["type"]
    sec = cfg["controller"]["ctc" if kind == "torque_ctc" else "torque_mpc"]
    obs = (sec.get("disturbance_observer") or {}).get("enabled", False)
    cfg["label"] = f"{'ctc' if kind == 'torque_ctc' else 'mpc'}_{'obs' if obs else 'noobs'}"
    return cfg


def summary_line(r: dict) -> str:
    out = f"{r['case']:<18} {r['controller']:<10} m={r['mass_kg']:.2f} kg  "
    out += "SUCCESS" if r["success"] else f"FAIL at {r['failure_stage']}: {r['failure_reason']}"
    extra = []
    for key, fmt in (("lift_height_mm", "lift {:.0f} mm"), ("hold_slip_mm", "hold slip {:.2f} mm"),
                     ("carry_slip_mm", "carry slip {:.2f} mm"), ("placement_error_mm", "place err {:.2f} mm"),
                     ("final_tilt_deg", "tilt {:.2f} deg")):
        if r.get(key) is not None:
            extra.append(fmt.format(r[key]))
    extra.append(f"ee err max {r['ee_err_max_mm']:.2f} mm")
    extra.append(f"forbidden steps {r['forbidden_contact_steps']}")
    if r.get("min_clearance_robot_env_mm") is not None:
        extra.append(f"clear robot {r['min_clearance_robot_env_mm']:.1f} mm")
    if r.get("min_clearance_object_env_mm") is not None:
        extra.append(f"clear object {r['min_clearance_object_env_mm']:.1f} mm")
    if "mpc" in r:
        extra.append(f"MPC fallback steps {r['mpc']['fallback_steps']}/{r['steps']}, statuses {r['mpc']['statuses']}")
    extra.append(f"sim {r['sim_time_s']:.1f} s, wall {r['wall_time_s']:.1f} s")
    return out + "\n    " + ", ".join(extra)


def main() -> None:
    ap = argparse.ArgumentParser(description="Known-world pick-and-place episodes (v0.6)")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--case", action="append", help="case name (repeatable); default: all cases in the config")
    ap.add_argument("--controller", choices=sorted(CONTROLLERS))
    ap.add_argument("--observer", choices=["on", "off"])
    ap.add_argument("--mass", type=float, help="override the object mass (kg)")
    ap.add_argument("--fault", choices=sorted(FAULTS), help="negative control")
    ap.add_argument("--output", help="write results as JSON")
    args = ap.parse_args()

    cfg = configure(load_config(args.config), args.controller, args.observer)
    cases = cfg.get("cases") or []
    if args.case:
        cases = [c for c in cases if c["name"] in args.case]
        if not cases:
            sys.exit(f"no case named {args.case} in {args.config}")
    results = []
    for case in cases:
        case = dict(case)
        if args.mass is not None:
            case["mass"] = args.mass
        if args.fault:
            fault = dict(FAULTS[args.fault])
            if "mass" in fault:
                case["mass"] = fault.pop("mass")
            case["fault"] = fault
        r = run_case(cfg, case)
        results.append(r)
        print(summary_line(r), flush=True)
    ok = sum(r["success"] for r in results)
    print(f"{ok}/{len(results)} episodes succeeded")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=1, default=float))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
