r"""Reach-derived work area and table size.

The table is as large as the arm's top-down grasp reach. For every azimuth around the base, probe height
(short to tall objects) and gripper yaw, the probe walks outwards and inwards along the ray and records
where the pre-grasp and grasp poses stop being reachable and collision-free (GraspProbe: pose IK plus the
planner's clearance rules). The usable ring is the intersection of all those bands, shrunk by a margin, over
the contiguous range of azimuths around the front whose band is wide enough. The table radius is the outer
radius of the ring plus a rim.

    ..\..\.venv\Scripts\python.exe -m studio.workspace                   # measure and print
    ..\..\.venv\Scripts\python.exe -m studio.workspace --write-default   # also refresh the packaged default

Results are cached by a key over every input (settings, grasp rules, MuJoCo version, algorithm version).
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from robots.panda import Panda
from simulation.engine import SimulationEngine
from studio.reach import GraspProbe
from studio.robot import grasp_depth, wrap_angle
from studio.settings import PACKAGE_DIR, StudioSettings

ALGORITHM_VERSION = 1
DEFAULT_FILE = PACKAGE_DIR / "data" / "workspace_default.json"
CACHE_NAME = "workspace.json"
PROBE_TABLE_RADIUS = 1.2  # m: large table for the measurement, so the table edge never limits the result


def workspace_key(settings: StudioSettings) -> str:
    pp = settings.pick_place
    s = settings.studio
    measured = {k: v for k, v in s["workspace"].items() if k != "max_azimuth_deg"}  # applied after measuring
    inputs = {
        "version": ALGORITHM_VERSION, "mujoco": mujoco.__version__, "environment": settings.environment,
        "table_height": settings.table_height, "workspace": measured, "grasp_depth": s["grasp_depth"],
        "home": s["robot"]["home"], "approach": pp["grasp"]["approach_height"], "collision": pp["collision"],
        "planning": {k: pp["planning"].get(k) for k in ("clearance", "joint_limit_margin", "exact_box_box")},
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _ray(probe: GraspProbe, theta: float, z_grasp: float, yaw: float, step: float,
         r_lo: float = 0.12, r_hi: float = 1.05) -> tuple[float, float] | None:
    """Reachable radius band along one azimuth for one gripper yaw representation (None: nowhere)."""
    c, s = np.cos(theta), np.sin(theta)

    def ok(r, warm):
        return probe.top_down((r * c, r * s), z_grasp, yaw, warm)[0]

    q0, r0 = None, None
    for r in (0.50, 0.45, 0.56, 0.40, 0.62, 0.35, 0.68):
        q0 = ok(r, None)
        if q0 is not None:
            r0 = r
            break
    if q0 is None:
        return None

    def walk(direction: float) -> float:
        r_ok, q = r0, q0
        while True:
            r = r_ok + direction * step
            if not r_lo <= r <= r_hi:
                return r_ok
            q1 = ok(r, q)
            if q1 is None:
                # Refine the boundary between r_ok and r (two bisection steps).
                lo, hi = r_ok, r
                for _ in range(2):
                    mid = 0.5 * (lo + hi)
                    qm = ok(mid, q)
                    if qm is None:
                        hi = mid
                    else:
                        lo, q = mid, qm
                return lo
            r_ok, q = r, q1

    return walk(-1.0), walk(+1.0)


def measure(settings: StudioSettings, log=print) -> dict:
    t0 = time.perf_counter()
    ws = settings.studio["workspace"]
    gd = settings.studio["grasp_depth"]
    table_z = settings.table_height
    engine = SimulationEngine(settings.environment, Panda(actuation="torque"),
                              {"geoms": {"table_top": {"size": [PROBE_TABLE_RADIUS, 0.02, 0.0]}}})
    probe = GraspProbe(settings, engine)
    step_deg = float(ws["azimuth_step_deg"])
    azimuths = np.arange(-180.0, 180.0, step_deg)
    profile = []
    for k, th_deg in enumerate(azimuths):
        th = np.radians(th_deg)
        band = [0.0, np.inf]
        for h in ws["probe_heights_m"]:
            z = table_z + float(h) - grasp_depth(float(h), gd)
            for yaw_deg in ws["yaws_deg"]:
                gamma = wrap_angle(th + np.radians(float(yaw_deg)))  # finger axis relative to the radial direction
                spans = [s for s in (_ray(probe, th, z, gamma, 2 * ws["radial_step_m"]),
                                     _ray(probe, th, z, wrap_angle(gamma + np.pi), 2 * ws["radial_step_m"]))
                         if s is not None]
                if not spans:
                    band = None
                    break
                # Either representation of the same grasp may be used: union of overlapping spans.
                spans.sort()
                lo, hi = spans[0]
                for a, b in spans[1:]:
                    if a <= hi + 1e-9:
                        hi = max(hi, b)
                    elif b - a > hi - lo:
                        lo, hi = a, b
                band[0], band[1] = max(band[0], lo), min(band[1], hi)
            if band is None:
                break
        profile.append({"theta_deg": float(th_deg),
                        "r_min": None if band is None else round(float(band[0]), 4),
                        "r_max": None if band is None else round(float(band[1]), 4)})
        log(f"  azimuth {th_deg:+6.1f} deg: " + ("unreachable" if band is None else
                                                  f"r in [{band[0]:.3f}, {band[1]:.3f}] m") +
            f"  ({k + 1}/{len(azimuths)}, {time.perf_counter() - t0:.0f} s)")

    zone = derive_zone(profile, ws)
    radius = zone["r_out"] + float(ws["table_rim_m"])
    return {"key": workspace_key(settings), "version": ALGORITHM_VERSION,
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_s": round(time.perf_counter() - t0, 1), "mujoco": mujoco.__version__,
            "table_height": table_z, "table_radius": round(radius, 4), "zone": zone, "profile": profile,
            "settings": {k: v for k, v in ws.items() if k != "max_azimuth_deg"}}


def derive_zone(profile: list[dict], ws: dict) -> dict:
    """Contiguous, symmetric azimuth range around the front with a wide enough band, and the common ring."""
    by_theta = {round(p["theta_deg"], 3): p for p in profile}
    step = float(ws["azimuth_step_deg"])
    width = float(ws["min_ring_width_m"])

    def good(t):
        p = by_theta.get(round(((t + 180.0) % 360.0) - 180.0, 3))
        return p is not None and p["r_min"] is not None and p["r_max"] - p["r_min"] >= width

    if not good(0.0):
        raise RuntimeError("the arm cannot grasp in front of itself; check the studio configuration")
    limit = 0.0
    while limit + step <= 180.0 and good(limit + step) and good(-(limit + step)):
        limit += step
    if limit >= 180.0 - step and good(180.0):
        limit = 180.0  # every sampled azimuth is usable: full ring
    used = [p for p in profile if p["r_min"] is not None and abs(p["theta_deg"]) <= limit + 1e-9]
    margin = float(ws["margin_m"])
    r_in = max(p["r_min"] for p in used) + margin
    r_out = min(p["r_max"] for p in used) - margin
    return {"r_in": round(r_in, 4), "r_out": round(r_out, 4), "theta_max_deg": float(limit)}


def load_or_measure(settings: StudioSettings, log=print, force: bool = False) -> dict:
    key = workspace_key(settings)
    cache = settings.data_dir / CACHE_NAME
    if not force:
        for path in (cache, DEFAULT_FILE):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("key") == key:
                return data
    log("Measuring the arm's reach (one time, about a minute) ...")
    data = measure(settings, log)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    log(f"Reach measured in {data['wall_s']:.0f} s: ring {data['zone']['r_in']:.3f}-{data['zone']['r_out']:.3f} m, "
        f"+-{data['zone']['theta_max_deg']:.0f} deg; table radius {data['table_radius']:.3f} m")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the studio work area")
    ap.add_argument("--write-default", action="store_true", help="also write studio/data/workspace_default.json")
    args = ap.parse_args()
    settings = StudioSettings.load()
    data = load_or_measure(settings, force=True)
    print(json.dumps(data["zone"]), "table radius", data["table_radius"])
    if args.write_default:
        DEFAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"wrote {DEFAULT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
