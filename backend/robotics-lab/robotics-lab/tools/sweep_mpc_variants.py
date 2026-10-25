"""Run benchmark scenarios for controller config variants.

    python tools/sweep_mpc_variants.py <scenario>[,...] '<name>=<json>' ...

The JSON is merged into config["controller"] of the benchmark controller named by
its optional "_controller" key (default "mpc"), e.g.
    'n10={"mpc": {"horizon": 10}}'   or   't50={"_controller": "torque_mpc", "torque_mpc": {"control_rate_hz": 50}}'
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.benchmark import BENCHMARK_CONFIG, deep_merge, run_case, scenario_config  # noqa: E402
from simulation.runner import load_config  # noqa: E402

bench = yaml.safe_load(open(BENCHMARK_CONFIG))
base = load_config()
scenarios = sys.argv[1].split(",")
for arg in sys.argv[2:]:
    name, override = arg.split("=", 1)
    for sc in scenarios:
        ctrl = json.loads(override)
        controller = ctrl.pop("_controller", "mpc")
        cfg = scenario_config(base, bench, controller, sc)
        cfg["controller"] = deep_merge(cfg["controller"], ctrl)
        m, _ = run_case(cfg, bench["scenarios"][sc], 0.002)
        print(f"{sc:11s} {name:22s} rms {1e3 * m['rms_error_rad']:7.3f} max {1e3 * m['max_error_rad']:7.2f} mrad "
              f"sat {m['torque_saturation_fraction']:.2f} v-viol {m['velocity_limit_violation_steps']:4d} "
              f"fb {m.get('mpc_fallbacks', 0):4d}/{m.get('mpc_updates', 0)} solve {m.get('mpc_solve_time_mean_ms', 0):5.1f}/"
              f"{m.get('mpc_solve_time_p95_ms', 0):5.1f} ms overrun {m['deadline_overruns']}", flush=True)
