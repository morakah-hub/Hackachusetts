"""Median MPC update/solve time over repeated demo runs (no profiler attached).

    python tools/time_mpc.py [repeats]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.runner import SimulationRunner, load_config  # noqa: E402

repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for kind in ("joint_mpc", "torque_mpc"):
    upd, solve, p95, err = [], [], [], []
    for _ in range(repeats):
        cfg = load_config()
        cfg["controller"] = {**cfg["controller"], "type": kind}
        r = SimulationRunner(cfg)
        log = r.run_headless(r.task.duration, log_every=1).arrays()
        s = r.controller.stats.summary(r.controller.control_period)
        upd.append(s["update_time_mean_ms"])
        p95.append(s["update_time_p95_ms"])
        solve.append(s["solve_time_mean_ms"])
        err.append(np.abs(log["q"] - log["q_ref"]).max())
    print(f"{kind:10s} update mean {np.median(upd):.2f} ms (runs {np.round(upd, 2).tolist()}), p95 {np.median(p95):.2f} ms, "
          f"OSQP {np.median(solve):.2f} ms, max error {err[0]:.3e} rad")
