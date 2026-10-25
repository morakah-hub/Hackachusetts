"""Audit MuJoCo's distance queries for the robot-environment geom pairs the planner relies on (v0.5.2).

For each pair type (Panda geom x environment geom, real sizes and meshes) an isolated two-geom
model is built: the environment geom fixed in the world with margin = clearance, the robot geom
on a free body, and the planner checker's solver flags (midphase off, same CCD settings). The robot
geom is placed at random poses near the environment geom. Three answers are compared per pose:

  contact   mj_collision with the margin: "too close" iff a contact is generated (the v0.5 checker);
  distance  mj_geomDistance (the v0.5.2 checker for box-box pairs, and ExactClearance);
  truth     an independent, certified value.

The truth does not trust MuJoCo:
  - overlap is decided by an LP (scipy linprog: is some convex combination of the robot geom's
    vertices inside the obstacle?), for cylinders with inscribed / circumscribed 64-gon prisms;
  - a separation distance is certified by a duality gap: mj_geomDistance's witness points must
    lie in their geoms (upper bound |a - b|), and the obstacle's support function along the
    witness normal gives the lower bound  min_B n.y - max_A n.x  (valid for any n). The distance
    counts as certified when upper - lower < 1 um; poses that cannot be certified are reported;
  - plane: the exact distance is the lowest vertex height (no witness needed);
  - when mj_geomDistance finds nothing within its search radius, a separating plane (direction
    from the robot geom's vertex centroid to the closest obstacle point) must prove the pose is
    farther than 40 mm; otherwise the pose is reported as uncertified.

Run from the repo root:
    .\\.venv\\Scripts\\python.exe tools\\audit_clearance.py               # all pairs, 4000 poses each
    .\\.venv\\Scripts\\python.exe tools\\audit_clearance.py --poses 500   # quick
Writes results/audit_clearance/{audit.csv, summary.md}.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planning.collision import CollisionChecker  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402

CLEARANCE = 0.02
NEAR = 0.04          # poses with true distance above this are not counted
CERT_TOL = 1e-6      # m: duality gap below which a distance is certified
PRISM_SIDES = 64
DISTMAX = 0.2        # m: mj_geomDistance search radius in the audit

# (label, scene, robot geom selector, environment geom name). Selectors: (body, type, index among that body's
# collidable geoms of that type). left_finger box 0 is the fingertip pad (17 x 8 x 17 mm).
PAIRS = [
    ("fingertip pad (box) - wall (box)", "planning_demo", ("left_finger", "box", 0), "wall"),
    ("fingertip pad (box) - table top (box)", "planning_demo", ("left_finger", "box", 0), "table_top"),
    ("finger tooth (box) - wall (box)", "planning_demo", ("left_finger", "box", 1), "wall"),
    ("fingertip pad (box) - obstacle_1 (box)", "obstacles", ("left_finger", "box", 0), "obstacle_1"),
    ("finger (mesh) - wall (box)", "planning_demo", ("left_finger", "mesh", 0), "wall"),
    ("hand (mesh) - wall (box)", "planning_demo", ("hand", "mesh", 0), "wall"),
    ("link7 (mesh) - table top (box)", "planning_demo", ("link7", "mesh", 0), "table_top"),
    ("fingertip pad (box) - obstacle_2 (cylinder)", "obstacles", ("left_finger", "box", 0), "obstacle_2"),
    ("hand (mesh) - obstacle_2 (cylinder)", "obstacles", ("hand", "mesh", 0), "obstacle_2"),
    ("fingertip pad (box) - floor (plane)", "planning_demo", ("left_finger", "box", 0), "floor"),
    ("hand (mesh) - floor (plane)", "planning_demo", ("hand", "mesh", 0), "floor"),
]

TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "mesh": mujoco.mjtGeom.mjGEOM_MESH}


def select_geom(m: mujoco.MjModel, body: str, gtype: str, index: int) -> int:
    b = m.body(body).id
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == TYPES[gtype]
          and (m.geom_contype[g] or m.geom_conaffinity[g])]
    return gs[index]


def pair_model(src: mujoco.MjModel, g_rob: int, g_env: int) -> mujoco.MjModel:
    """Two-geom model: copy of the environment geom (world, margin = clearance) and of the robot geom (free body)."""
    spec = mujoco.MjSpec()
    spec.option.disableflags = int(src.opt.disableflags)
    spec.option.enableflags = int(src.opt.enableflags)
    spec.option.ccd_iterations = int(src.opt.ccd_iterations)
    spec.option.ccd_tolerance = float(src.opt.ccd_tolerance)
    e = spec.worldbody.add_geom(type=src.geom_type[g_env], size=src.geom_size[g_env].copy(), margin=CLEARANCE)
    e.name = "env"
    body = spec.worldbody.add_body(pos=[0, 0, 1])
    body.add_freejoint()
    kw = {"type": src.geom_type[g_rob], "size": src.geom_size[g_rob].copy()}
    if src.geom_type[g_rob] == mujoco.mjtGeom.mjGEOM_MESH:
        k = src.geom_dataid[g_rob]
        verts = src.mesh_vert[src.mesh_vertadr[k]: src.mesh_vertadr[k] + src.mesh_vertnum[k]]
        spec.add_mesh(name="robot_mesh", uservert=verts.flatten().tolist())
        kw["meshname"] = "robot_mesh"
    r = body.add_geom(**kw)
    r.name = "robot"
    m = spec.compile()
    m.opt.disableflags = src.opt.disableflags
    return m


def robot_vertices(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> np.ndarray:
    R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        local = s * m.geom_size[g]
    else:
        k = m.geom_dataid[g]
        local = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    return c + local @ R.T


def env_halfspaces(m, d, g, sides=PRISM_SIDES, outer=False):
    """Obstacle as {x: A x <= b} (box exactly; cylinder as an inscribed or circumscribed prism)."""
    R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        N = np.vstack([R.T, -R.T])
        return N, N @ c + np.r_[h, h]
    r, hz = h[0], h[1]
    ang = 2 * np.pi * (np.arange(sides) + 0.5) / sides
    rad = r if outer else r * np.cos(np.pi / sides)
    loc = np.c_[np.cos(ang), np.sin(ang), np.zeros(sides)]
    loc = np.vstack([loc, [0, 0, 1], [0, 0, -1]])
    N = loc @ R.T
    return N, N @ c + np.r_[np.full(sides, rad), hz, hz]


def overlaps(V: np.ndarray, A: np.ndarray, b: np.ndarray) -> bool:
    """LP feasibility: some convex combination of V lies in {A x <= b}."""
    n = len(V)
    res = linprog(np.zeros(n), A_ub=A @ V.T, b_ub=b, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, None)] * n,
                  method="highs")
    return res.status == 0


def env_support_min(m, d, g, n):
    """min over the obstacle of n.y (box, cylinder)."""
    R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
    nl = R.T @ n
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(n @ c - np.abs(nl) @ h)
    return float(n @ c - h[0] * np.linalg.norm(nl[:2]) - h[1] * abs(nl[2]))


def env_contains(m, d, g, p, tol=1e-7):
    R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
    loc = R.T @ (p - c)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        return bool(np.all(np.abs(loc) <= h + tol))
    return bool(np.linalg.norm(loc[:2]) <= h[0] + tol and abs(loc[2]) <= h[1] + tol)


def robot_contains(V: np.ndarray, p: np.ndarray, tol=1e-7) -> bool:
    """p in conv(V) up to tol (LP with an L-inf slack)."""
    n = len(V)
    # min s  s.t. |V^T l - p| <= s, sum l = 1, l >= 0
    c = np.r_[np.zeros(n), 1.0]
    A = np.vstack([np.c_[V.T, -np.ones(3)], np.c_[-V.T, -np.ones(3)]])
    res = linprog(c, A_ub=A, b_ub=np.r_[p, -p], A_eq=np.r_[np.ones(n), 0.0][None], b_eq=[1.0],
                  bounds=[(0, None)] * n + [(0, None)], method="highs")
    return res.status == 0 and res.fun <= tol


def env_closest(m, d, g, p):
    """Closest point of the obstacle (box, cylinder) to p."""
    R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
    loc = R.T @ (p - c)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        loc = np.clip(loc, -h, h)
    else:
        r = np.linalg.norm(loc[:2])
        if r > h[0]:
            loc[:2] *= h[0] / r
        loc[2] = np.clip(loc[2], -h[1], h[1])
    return c + R @ loc


def true_distance(m, d, g_env, g_rob, V, witness, distmax):
    """(kind, value): kind 'overlap' | 'separated' | 'far' | 'uncertified' | 'ambiguous'.
    For 'separated' the value is the certified distance and the third item the duality gap."""
    if m.geom_type[g_env] == mujoco.mjtGeom.mjGEOM_PLANE:
        n, c = d.geom_xmat[g_env].reshape(3, 3)[:, 2], d.geom_xpos[g_env]
        z = float(np.min((V - c) @ n))
        return ("overlap", z) if z < 0 else ("separated", z, 0.0)
    inner = overlaps(V, *env_halfspaces(m, d, g_env, outer=False))
    if inner:
        return "overlap", 0.0
    if m.geom_type[g_env] == mujoco.mjtGeom.mjGEOM_CYLINDER and overlaps(V, *env_halfspaces(m, d, g_env, outer=True)):
        return "ambiguous", 0.0  # within 3 um of the cylinder surface: below what the prism resolves
    dist, a, b = witness  # a on the environment geom (geom1), b on the robot geom (geom2)
    if dist >= distmax:  # MuJoCo: nothing within distmax. Prove the pose is far with a separating plane.
        p = V.mean(axis=0)
        y = env_closest(m, d, g_env, p)
        n = (y - p) / np.linalg.norm(y - p)
        lower = env_support_min(m, d, g_env, n) - float(np.max(V @ n))
        return ("far", lower) if lower > NEAR else ("uncertified", float("nan"))
    if dist <= 0 or not (env_contains(m, d, g_env, a) and robot_contains(V, b)):
        return "uncertified", float("nan")
    n = (a - b) / np.linalg.norm(a - b)  # from robot to environment
    lower = env_support_min(m, d, g_env, n) - float(np.max(V @ n))
    upper = float(np.linalg.norm(a - b))
    return ("separated", upper, upper - lower) if upper - lower < CERT_TOL else ("uncertified", float("nan"))


def sample_pose(rng, m, d, g_env, rb):
    if m.geom_type[g_env] == mujoco.mjtGeom.mjGEOM_PLANE:
        pos = np.r_[rng.uniform(-0.2, 0.2, 2), rng.uniform(-rb, rb + NEAR)]
    else:
        R, c, h = d.geom_xmat[g_env].reshape(3, 3), d.geom_xpos[g_env], m.geom_size[g_env]
        half = h if m.geom_type[g_env] == mujoco.mjtGeom.mjGEOM_BOX else np.r_[h[0], h[0], h[1]]
        hi = half + rb + NEAR
        lo = -hi
        big = half > 0.2  # long sides (table top, wall length): sample only the last 0.2 m so ends/corners count
        lo[big] = half[big] - 0.2
        pos = c + R @ rng.uniform(lo, hi)
    q = rng.normal(size=4)
    return pos, q / np.linalg.norm(q)


def audit_pair(label, scene, sel, env_name, poses, seed):
    src = CollisionChecker.from_engine(SimulationEngine(scene), clearance=CLEARANCE).model
    g_rob_src, g_env_src = select_geom(src, *sel), src.geom(env_name).id
    m = pair_model(src, g_rob_src, g_env_src)
    d = mujoco.MjData(m)
    g_env, g_rob = 0, 1
    mujoco.mj_kinematics(m, d)
    rb = m.geom_rbound[g_rob]
    rng = np.random.default_rng(seed)
    st = {"pair": label, "geom_types": f"{mujoco.mjtGeom(m.geom_type[g_rob]).name[7:].lower()}-"
                                      f"{mujoco.mjtGeom(m.geom_type[g_env]).name[7:].lower()}",
          "poses": 0, "overlap": 0, "separated_near": 0, "inside_clearance": 0, "uncertified": 0, "ambiguous": 0,
          "contact_missed_overlap": 0, "distance_missed_overlap": 0,
          "contact_false_safe": 0, "contact_false_alarm": 0, "contact_dist_err_min_mm": 0.0, "contact_dist_err_max_mm": 0.0,
          "distance_false_safe": 0, "distance_false_alarm": 0,
          "distance_err_max_mm": 0.0,  # |mj_geomDistance - certified upper bound| + duality gap: a bound on the error
          "closest_contact_miss_mm": float("nan")}
    ft = np.zeros(6)
    t0 = time.perf_counter()
    while st["poses"] < poses:
        pos, quat = sample_pose(rng, m, d, g_env, rb)
        d.qpos[:3], d.qpos[3:7] = pos, quat
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        gd = mujoco.mj_geomDistance(m, d, g_env, g_rob, DISTMAX, ft)
        V = robot_vertices(m, d, g_rob)
        kind, truth, *gap = true_distance(m, d, g_env, g_rob, V, (gd, ft[:3].copy(), ft[3:].copy()), DISTMAX)
        if kind == "far" or (kind == "separated" and truth > NEAR):
            continue
        st["poses"] += 1
        contact = d.ncon > 0
        cdist = min((c.dist for c in d.contact[: d.ncon]), default=None)
        if kind == "ambiguous":
            st["ambiguous"] += 1
            continue
        if kind == "uncertified":
            st["uncertified"] += 1
            continue
        if kind == "overlap":
            st["overlap"] += 1
            st["contact_missed_overlap"] += not contact
            st["distance_missed_overlap"] += not gd < 0
            continue
        st["separated_near"] += 1
        inside = truth < CLEARANCE
        st["inside_clearance"] += inside
        if inside and not contact:
            st["contact_false_safe"] += 1
            st["closest_contact_miss_mm"] = np.nanmin([st["closest_contact_miss_mm"], 1e3 * truth])
        st["contact_false_alarm"] += (not inside) and contact
        if cdist is not None:
            e = 1e3 * (cdist - truth)
            st["contact_dist_err_min_mm"] = min(st["contact_dist_err_min_mm"], e)
            st["contact_dist_err_max_mm"] = max(st["contact_dist_err_max_mm"], e)
        st["distance_false_safe"] += inside and not gd < CLEARANCE
        st["distance_false_alarm"] += (not inside) and gd < CLEARANCE
        st["distance_err_max_mm"] = max(st["distance_err_max_mm"], 1e3 * (abs(gd - truth) + gap[0]))
    st["seconds"] = round(time.perf_counter() - t0, 1)
    return st


def markdown(rows):
    head = ("| pair | types | poses | overlap | sep. < 40 mm | of which < 20 mm | contact: missed overlap | "
            "contact: false safe (closest mm) | contact: false alarm | contact.dist err (mm) | "
            "geomDistance: missed overlap | geomDistance: false safe / alarm | geomDistance error bound (mm) | uncertified |")
    lines = [head, "|" + "---|" * 14]
    for r in rows:
        cm = "" if np.isnan(r["closest_contact_miss_mm"]) else f" ({r['closest_contact_miss_mm']:.2f})"
        lines.append(
            f"| {r['pair']} | {r['geom_types']} | {r['poses']} | {r['overlap']} | {r['separated_near']} | {r['inside_clearance']} | "
            f"{r['contact_missed_overlap']} | {r['contact_false_safe']}{cm} | {r['contact_false_alarm']} | "
            f"{r['contact_dist_err_min_mm']:+.2f} .. {r['contact_dist_err_max_mm']:+.2f} | {r['distance_missed_overlap']} | "
            f"{r['distance_false_safe']} / {r['distance_false_alarm']} | {r['distance_err_max_mm']:.1e} | "
            f"{r['uncertified'] + r['ambiguous']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--poses", type=int, default=4000, help="counted poses per pair (true distance <= 40 mm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/audit_clearance"))
    args = ap.parse_args()
    print(f"MuJoCo {mujoco.__version__}, clearance {1e3 * CLEARANCE:.0f} mm, {args.poses} poses per pair")
    rows = []
    for k, (label, scene, sel, env) in enumerate(PAIRS):
        r = audit_pair(label, scene, sel, env, args.poses, args.seed + k)
        rows.append(r)
        print(f"  {label:45s} contact false-safe {r['contact_false_safe']:4d}  missed overlap {r['contact_missed_overlap']}  "
              f"| geomDistance false-safe {r['distance_false_safe']} missed overlap {r['distance_missed_overlap']} "
              f"max err {r['distance_err_max_mm']:.1e} mm | uncertified {r['uncertified']} ({r['seconds']} s)", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.output / "summary.md").write_text(markdown(rows), encoding="utf-8")
    print(f"wrote {args.output / 'audit.csv'} and summary.md")


if __name__ == "__main__":
    main()
