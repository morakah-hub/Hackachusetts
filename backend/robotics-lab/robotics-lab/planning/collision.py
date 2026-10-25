"""Configuration and edge collision checking for the arm, on a private model copy.

A configuration q (7 arm joint angles) is *valid* if

  1. it lies inside the joint limits shrunk by `limit_margin`, and
  2. MuJoCo's collision detection, run on private scratch data at q, reports
     no contact involving a robot body, where robot-environment pairs count
     as contacts already when they are closer than `clearance` (m).

The clearance is implemented with MuJoCo's own contact margin: the checker
compiles nothing new, it deep-copies the plant model and sets `geom_margin =
clearance` on every collidable non-robot geom. A pair's margin is the larger of
the two geoms' margins, so robot-environment pairs report contacts with
dist < clearance (and `contact.dist` is the signed distance), while robot-robot
pairs keep margin 0 (self-collision = penetration). Everything else is MuJoCo's
intended filtering, unchanged: contype/conaffinity bits, parent-child filtering,
and explicit <exclude> pairs (Menagerie link0-link1, and the robot base-table
exclude added by simulation/engine.py). The plant model is never modified.

Exception (v0.5.2): box-box robot-environment pairs (the Panda's fingertip pads
against box obstacles, the table top, walls). MuJoCo 3.16's box-box routine does
not generate margin contacts reliably: it misses some pairs inside the margin
(one 1.25 mm apart under a 20 mm margin) and misreports `contact.dist` by
several mm, while it always detects penetration. With `exact_box_box` (default)
the checker ignores MuJoCo's contacts for these pairs and asks
`mj_geomDistance` instead, which matched certified distances on every audited
pose (tools/audit_clearance.py, docs/research/v0.5.2-results.md). Only pairs
that a conservative bound cannot rule out are queried (robot box center vs the
obstacle's world AABB grown by clearance + the robot box's bounding radius);
the others are provably at least the clearance apart. All other pair types (meshes vs boxes, cylinders,
planes; boxes vs cylinders, planes) passed the same audit with MuJoCo's own
margin contacts and stay with mj_collision. `exact_box_box=False` restores the
v0.5 / v0.5.1 behavior (for comparisons only).

Edges (straight joint-space segments q0 -> q1) are checked at interpolated
configurations. The number of samples comes from a configuration-independent
bound on how far any point of the robot's collision geometry can move:

    |dp| <= sum_j R_j |dq_j|,

where R_j bounds the distance from joint j's anchor to any collision geom of
the bodies it moves (chain of body offsets + geom offset + bounding radius;
triangle inequality, valid for every posture). Samples are spaced so that no
point moves more than `cartesian_step` between consecutive samples (and no
joint more than `max_joint_step`). Consequence, for robot-environment pairs:
if every sample keeps distance >= clearance, then every configuration on the
edge keeps distance >= clearance - cartesian_step / 2 (each point is within
half a step, along its path, of a sampled position; distance is 1-Lipschitz).
With the defaults (clearance 20 mm, step 10 mm) that is >= 15 mm. This holds
for MuJoCo's collision geometry (convex hulls of the meshes) and requires the
per-sample answer "distance < clearance" to be exact. With `exact_box_box` it
is, up to the audited accuracy of MuJoCo's queries (pair types audited on random
poses, not proven); without it, box-box pairs break it (v0.5.1 finding). It is
not a statement about the real robot. Self-collision
pairs have margin 0, so for them the check is discrete only: a brief
self-intersection between two samples could be missed (it would have to be
shallower than the motion between samples).

Manipulation (v0.6), all on the private copy and all opt-in (defaults give the v0.5.2 checker):

* `set_fixed_qpos(qpos)` replaces the non-arm joint positions used for every check (finger
  opening, the free object's known pose).
* `allow(geoms_a, geoms_b, min_distance)` relaxes the rule for one set of pairs: a robot
  contact between a geom of `geoms_a` and one of `geoms_b` is ignored when its distance is
  >= min_distance (-inf: any contact allowed, e.g. intended finger-object contact while
  grasping; 0: touching allowed, penetration not). Every other pair keeps the clearance.
  `clear_allowed()` restores the default.
* `attach(body, parent)` rigidly attaches a free body (the grasped object) to a robot body at
  their current relative pose: at every checked configuration its pose follows the parent,
  its geoms count as robot geoms (so object-environment pairs need the clearance, box-box
  pairs among them go through mj_geomDistance), and the lever arms grow by its extent so the
  edge-sampling bound covers it. `detach()` undoes this. This is a rigid-attachment
  approximation used only here, for predicting the carried object's geometry; the plant
  holds the object through simulated contact and friction.

Nothing here reads or writes the live MjData except one read of the non-arm
joint positions (fingers) at construction.
"""

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

_BOX = mujoco.mjtGeom.mjGEOM_BOX


@dataclass
class CollisionReport:
    valid: bool
    reason: str = ""
    contacts: list[tuple[str, str, float]] = field(default_factory=list)  # (body1, body2, signed distance m)


class CollisionChecker:
    def __init__(
        self,
        model: mujoco.MjModel,
        base_qpos: np.ndarray,
        qpos_adr: np.ndarray,
        dof_adr: np.ndarray,
        robot_bodies: list[int],
        joint_limits: np.ndarray,
        clearance: float = 0.02,
        cartesian_step: float = 0.01,
        max_joint_step: float = 0.05,
        limit_margin: float = 0.05,
        exact_box_box: bool = True,
    ):
        if clearance < 0 or cartesian_step <= 0 or max_joint_step <= 0:
            raise ValueError("clearance must be >= 0, cartesian_step and max_joint_step > 0")
        self.model = copy.deepcopy(model)
        self.data = mujoco.MjData(self.model)
        self.base_qpos = np.array(base_qpos, dtype=float)
        self.qpos_adr = np.asarray(qpos_adr)
        self.robot_bodies = set(int(b) for b in robot_bodies)
        self.clearance = float(clearance)
        self.cartesian_step = float(cartesian_step)
        self.max_joint_step = float(max_joint_step)
        self.limit_margin = float(limit_margin)
        limits = np.asarray(joint_limits, dtype=float)
        self.q_min, self.q_max = limits[:, 0] + limit_margin, limits[:, 1] - limit_margin

        m = self.model
        self._is_robot_geom = np.array([m.geom_bodyid[g] in self.robot_bodies for g in range(m.ngeom)])
        collidable = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        env = collidable & ~self._is_robot_geom
        m.geom_margin[env] = np.maximum(m.geom_margin[env], self.clearance)
        # MuJoCo's midphase (per-body BVH culling, used for multi-geom bodies such as the
        # fingers) does not account for the other geom's margin: in MuJoCo 3.16 it dropped a
        # finger-wall pair 8.5 mm apart under a 20 mm margin. Without it every pair that passes
        # the (margin-aware) broadphase reaches the narrowphase. See docs/research/v0.5-results.md.
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self.dof_adr = np.asarray(dof_adr)
        self.lever_arms = self._lever_arms(self.dof_adr)
        self._base_lever_arms = self.lever_arms.copy()
        self.config_checks = 0  # configurations evaluated by MuJoCo (all calls)
        self.exact_queries = 0  # mj_geomDistance calls for box-box pairs (all calls)
        self._pair_min = np.full((m.ngeom, m.ngeom), np.nan)  # allowed pairs: ignored if dist >= value
        self._attached = None  # (body, parent, qpos adr, relative pos, relative quat)
        self._base_robot_geom = self._is_robot_geom.copy()

        self.exact_box_box = bool(exact_box_box)
        self._setup_exact_pairs()

    def _setup_exact_pairs(self) -> None:
        m = self.model
        pairs = self._box_box_pairs() if self.exact_box_box else []
        rob, envb = sorted({a for a, _ in pairs}), sorted({b for _, b in pairs})
        self._ex_rob, self._ex_env = np.array(rob, dtype=int), np.array(envb, dtype=int)
        self._ex_pair = np.zeros((len(rob), len(envb)), dtype=bool)  # [robot box, env box] pairs to check
        for a, b in pairs:
            self._ex_pair[rob.index(a), envb.index(b)] = True
        self._ex_rbound = m.geom_rbound[self._ex_rob][:, None]
        self._ex_half = m.geom_size[self._ex_env]
        bodies = m.geom_bodyid[self._ex_env]
        self._ex_static = bool(np.all(m.body_weldid[bodies] == 0) and np.all(m.body_mocapid[bodies] < 0))
        self._ex_aabb = None  # obstacle boxes' world AABBs (center, half-size), cached when static
        self._ex_threshold = (None, None)  # (cap, query radius per pair; -inf = not a pair)
        self._ex_skip = np.zeros((m.ngeom, m.ngeom), dtype=bool)  # MuJoCo's contacts for these pairs are ignored
        for a, b in pairs:
            self._ex_skip[a, b] = self._ex_skip[b, a] = True

    @classmethod
    def from_engine(cls, engine, **kwargs) -> "CollisionChecker":
        """Checker for the engine's plant geometry. Fingers (and any other non-arm
        joints) stay at their current positions, e.g. the open gripper."""
        return cls(engine.model, engine.data.qpos, engine.arm_qpos_adr, engine.arm_dof_adr,
                   engine._robot_body_ids(), engine.joint_limits, **kwargs)

    def _box_box_pairs(self) -> list[tuple[int, int]]:
        """(robot box geom, environment box geom) pairs that MuJoCo's filters let collide."""
        m = self.model
        live = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        boxes = live & (m.geom_type == _BOX)
        excluded = {(int(s) >> 16, int(s) & 0xFFFF) for s in m.exclude_signature[: m.nexclude]}
        pairs = []
        for a in np.nonzero(boxes & self._is_robot_geom)[0]:
            for b in np.nonzero(boxes & ~self._is_robot_geom)[0]:
                if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
                    continue
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                if (ba, bb) in excluded or (bb, ba) in excluded:
                    continue
                pairs.append((int(a), int(b)))
        return pairs

    # ------------------------------------------------------------------ geometry bound

    def _chain_bound(self, dof: int, body: int) -> float | None:
        """Upper bound on the distance from joint `dof`'s anchor to `body`'s origin, valid for every
        posture (None if the joint does not move the body)."""
        m = self.model
        jnt = m.dof_jntid[dof]
        jbody = m.jnt_bodyid[jnt]
        chain, b = 0.0, body
        while b > 0:  # walk up to the joint's body, summing link offsets
            if b == jbody:
                return float(np.linalg.norm(m.jnt_pos[jnt]) + chain)
            chain += np.linalg.norm(m.body_pos[b])
            for k in range(m.body_jntnum[b]):  # slide joints shift the body further
                jk = m.body_jntadr[b] + k
                if m.jnt_type[jk] == mujoco.mjtJoint.mjJNT_SLIDE:
                    chain += np.abs(m.jnt_range[jk]).max()
                else:
                    chain += 2.0 * np.linalg.norm(m.jnt_pos[jk])  # hinge off the body origin
            b = m.body_parentid[b]
        return None

    def _lever_arms(self, dof_adr: np.ndarray) -> np.ndarray:
        """R_j: upper bound on the distance from joint j's anchor to any robot collision geom it moves."""
        m = self.model
        geoms = [g for g in range(m.ngeom) if self._is_robot_geom[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
        out = np.zeros(len(dof_adr))
        for j, dof in enumerate(dof_adr):
            best = 0.0
            for g in geoms:
                chain = self._chain_bound(dof, m.geom_bodyid[g])
                if chain is not None:
                    best = max(best, chain + np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g])
            out[j] = best
        return out

    # ------------------------------------------------------------------ manipulation (v0.6)

    def set_fixed_qpos(self, qpos: np.ndarray) -> None:
        """Non-arm joint positions used at every check (fingers, free objects); arm entries are ignored."""
        self.base_qpos = np.array(qpos, dtype=float)

    def allow(self, geoms_a, geoms_b, min_distance: float = -np.inf) -> None:
        """Ignore contacts between geoms_a and geoms_b (either order) at distance >= min_distance."""
        for a in np.atleast_1d(geoms_a):
            for b in np.atleast_1d(geoms_b):
                self._pair_min[a, b] = self._pair_min[b, a] = min_distance

    def clear_allowed(self) -> None:
        self._pair_min[:] = np.nan

    def body_geoms(self, *bodies: str) -> np.ndarray:
        """Collidable geom ids of the named bodies."""
        m = self.model
        ids = {m.body(b).id for b in bodies}
        return np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in ids
                         and (m.geom_contype[g] or m.geom_conaffinity[g])], dtype=int)

    def attach(self, body: str, parent: str) -> None:
        """Rigidly attach free body `body` to robot body `parent` at their relative pose in base_qpos."""
        m, d = self.model, self.data
        b, p = m.body(body).id, m.body(parent).id
        jnt = m.body_jntadr[b]
        if m.body_jntnum[b] != 1 or m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"attach: body '{body}' must have exactly one free joint")
        if p not in self.robot_bodies:
            raise ValueError(f"attach: parent '{parent}' is not a robot body")
        d.qpos[:] = self.base_qpos
        mujoco.mj_kinematics(m, d)
        rel_pos = d.xmat[p].reshape(3, 3).T @ (d.xpos[b] - d.xpos[p])
        neg, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(neg, d.xquat[p])
        mujoco.mju_mulQuat(rel_quat, neg, d.xquat[b])
        self._attached = (b, p, int(m.jnt_qposadr[jnt]), rel_pos, rel_quat)
        geoms = self.body_geoms(body)
        self._is_robot_geom = self._base_robot_geom.copy()
        self._is_robot_geom[geoms] = True
        extent = max(np.linalg.norm(rel_pos) + np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g] for g in geoms)
        self.lever_arms = self._base_lever_arms.copy()
        for j, dof in enumerate(self.dof_adr):
            chain = self._chain_bound(dof, p)
            if chain is not None:
                self.lever_arms[j] = max(self.lever_arms[j], chain + extent)
        self._setup_exact_pairs()

    def detach(self) -> None:
        self._attached = None
        self._is_robot_geom = self._base_robot_geom.copy()
        self.lever_arms = self._base_lever_arms.copy()
        self._setup_exact_pairs()

    def _place_attached(self) -> None:
        _, p, adr, rel_pos, rel_quat = self._attached
        d = self.data
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, d.xquat[p], rel_quat)
        d.qpos[adr:adr + 3] = d.xpos[p] + d.xmat[p].reshape(3, 3) @ rel_pos
        d.qpos[adr + 3:adr + 7] = quat
        mujoco.mj_kinematics(self.model, d)

    def motion_bound(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Upper bound (m) on the distance any robot collision point travels along q0 -> q1."""
        return float(self.lever_arms @ np.abs(np.asarray(q1) - np.asarray(q0)))

    def edge_samples(self, q0: np.ndarray, q1: np.ndarray) -> int:
        """Number of intervals the edge q0 -> q1 is split into for checking."""
        dq = np.abs(np.asarray(q1) - np.asarray(q0))
        n = max(self.motion_bound(q0, q1) / self.cartesian_step, float(dq.max()) / self.max_joint_step)
        return max(1, math.ceil(n - 1e-12))

    # ------------------------------------------------------------------ configurations

    def within_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.q_min - 1e-12) and np.all(q <= self.q_max + 1e-12))

    def _collide(self, q: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = self.base_qpos
        d.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)
        if self._attached is not None:
            self._place_attached()
        mujoco.mj_collision(self.model, d)
        self.config_checks += 1

    def _exact_distances(self, cap: float):
        """(robot geom, env geom, distance) for box-box robot-environment pairs closer than cap,
        after _collide. A pair is queried only if the robot box's center lies inside the obstacle
        box's world AABB grown by cap + the robot box's bounding radius. Otherwise the center is
        at least that far from the AABB, hence from the obstacle, along some axis, and no point
        of the robot box can be within cap: those pairs are provably >= cap apart."""
        if not self._ex_pair.size:
            return
        m, d = self.model, self.data
        aabb = self._ex_aabb
        if aabb is None:
            rot = d.geom_xmat[self._ex_env].reshape(-1, 3, 3)
            aabb = (d.geom_xpos[self._ex_env].copy(), np.einsum("kij,kj->ki", np.abs(rot), self._ex_half))
            if self._ex_static:
                self._ex_aabb = aabb
        center, half = aabb
        last_cap, threshold = self._ex_threshold
        if last_cap != cap:
            threshold = np.where(self._ex_pair, cap + self._ex_rbound, -np.inf)
            self._ex_threshold = (cap, threshold)
        gap = (np.abs(d.geom_xpos[self._ex_rob][:, None, :] - center) - half).max(axis=2)
        n_env = len(self._ex_env)
        for k in np.flatnonzero(gap < threshold):
            a, b = int(self._ex_rob[k // n_env]), int(self._ex_env[k % n_env])
            self.exact_queries += 1
            dist = mujoco.mj_geomDistance(m, d, a, b, cap, None)
            if dist < cap and not dist >= self._pair_min[a, b]:  # allowed pairs (v0.6)
                yield a, b, float(dist)

    def _contact_arrays(self):
        """geom1, geom2, dist of MuJoCo's contacts, without the pairs handled by _exact_distances."""
        con = self.data.contact
        g1, g2, dist = con.geom1, con.geom2, con.dist
        keep = ~self._ex_skip[g1, g2] & ~(dist >= self._pair_min[g1, g2])  # allowed pairs (v0.6)
        return g1[keep], g2[keep], dist[keep]

    def _robot_contacts(self):
        m = self.model
        if self.data.ncon:
            g1, g2, dist = self._contact_arrays()
            for k in np.nonzero(self._is_robot_geom[g1] | self._is_robot_geom[g2])[0]:
                yield m.geom_bodyid[g1[k]], m.geom_bodyid[g2[k]], float(dist[k])
        for a, b, dist in self._exact_distances(self.clearance):
            yield m.geom_bodyid[a], m.geom_bodyid[b], dist

    def in_collision(self, q: np.ndarray) -> bool:
        """True if q penetrates or comes closer than the clearance (ignores joint limits)."""
        self._collide(q)
        return next(self._robot_contacts(), None) is not None

    def is_valid(self, q: np.ndarray) -> bool:
        return self.within_limits(q) and not self.in_collision(q)

    def check(self, q: np.ndarray) -> CollisionReport:
        """Validity of q with a human-readable reason and the offending contacts."""
        q = np.asarray(q, dtype=float)
        if q.shape != self.q_min.shape:
            return CollisionReport(False, f"expected {len(self.q_min)} joint angles, got shape {q.shape}")
        bad = np.nonzero((q < self.q_min - 1e-12) | (q > self.q_max + 1e-12))[0]
        if bad.size:
            j = int(bad[0])
            return CollisionReport(False, f"joint {j + 1} = {q[j]:.4f} outside [{self.q_min[j]:.4f}, {self.q_max[j]:.4f}] "
                                          f"(limits minus {self.limit_margin} rad margin)")
        self._collide(q)
        m = self.model
        contacts = [(m.body(b1).name, m.body(b2).name, dist) for b1, b2, dist in self._robot_contacts()]
        if not contacts:
            return CollisionReport(True)
        b1, b2, dist = min(contacts, key=lambda c: c[2])
        what = "collision" if dist <= 0 else f"within clearance ({1e3 * dist:.1f} mm < {1e3 * self.clearance:.0f} mm)"
        return CollisionReport(False, f"{what}: {b1}-{b2}", contacts)

    def min_distance(self, q: np.ndarray) -> float:
        """Smallest robot-environment distance (m) at q, capped at the clearance (= "at least
        this much"); negative = penetration depth. Self-contacts are ignored here."""
        self._collide(q)
        m = self.model
        best = self.clearance
        if self.data.ncon:
            g1, g2, dist = self._contact_arrays()
            env = self._is_robot_geom[g1] != self._is_robot_geom[g2]
            if env.any():
                best = min(best, float(dist[env].min()))
        for _, _, dist in self._exact_distances(self.clearance):
            best = min(best, dist)
        return best

    # ------------------------------------------------------------------ edges

    def edge_valid(self, q0: np.ndarray, q1: np.ndarray, check_endpoints: bool = False) -> bool:
        """True if the straight joint-space edge q0 -> q1 is valid at every sample.
        Interior samples are visited coarse-to-fine (midpoint first), so blocked
        edges are usually rejected after a few checks."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if check_endpoints and not (self.is_valid(q0) and self.is_valid(q1)):
            return False
        if not (self.within_limits(q0) and self.within_limits(q1)):  # the box is convex: endpoints suffice
            return False
        n = self.edge_samples(q0, q1)
        dq = q1 - q0
        for i in _coarse_to_fine(n):
            if self.in_collision(q0 + dq * (i / n)):
                return False
        return True

    def first_invalid(self, q0: np.ndarray, q1: np.ndarray) -> float | None:
        """Fraction s in (0, 1] of the first invalid sample walking from q0 (None if the edge is valid)."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = self.edge_samples(q0, q1)
        for i in range(1, n + 1):
            q = q0 + (q1 - q0) * (i / n)
            if not self.is_valid(q):
                return i / n
        return None

    def path_valid(self, path: list[np.ndarray]) -> tuple[bool, str]:
        """Check every vertex and every edge of a joint-space path."""
        for k, q in enumerate(path):
            rep = self.check(q)
            if not rep.valid:
                return False, f"vertex {k}: {rep.reason}"
        for k in range(len(path) - 1):
            if not self.edge_valid(path[k], path[k + 1]):
                s = self.first_invalid(path[k], path[k + 1])
                where = "" if s is None else f" at s = {s:.3f}"
                return False, f"edge {k} -> {k + 1} blocked{where}"
        return True, ""


def _coarse_to_fine(n: int):
    """Interior indices 1..n-1 in bisection order (n/2, n/4, 3n/4, ...), each once."""
    if n < 2:
        return
    seen = set()
    step = n
    while step >= 1:
        for i in range(step // 2 if step > 1 else 1, n, max(step, 1)):
            if 0 < i < n and i not in seen:
                seen.add(i)
                yield i
        if step == 1:
            break
        step //= 2
    for i in range(1, n):  # anything the halving missed (n not a power of two)
        if i not in seen:
            yield i
