"""Several inverse-kinematics solutions for one end-effector position.

The Panda has 7 joints and the IK fixes only the 3-D position, so a target has
a 4-D family of joint solutions (elbow up/down, wrist flips, ...). Some are in
collision or too close to an obstacle while others are free. Seeds: the
configuration the arm comes from, the rest posture, `local_seeds` Gaussian
perturbations of the current configuration (sigma `local_sigma` rad; they
favour postures similar to the current one, which tend to be connected to it),
then uniformly random configurations within the joint limits (seeded
generator). For perturbed and random seeds the IK null-space term pulls
towards the seed itself, so they converge to different postures instead of
all collapsing onto the rest-biased one.

A solution is a candidate only if IK converged (error < tol) *and* the
collision checker accepts it (joint limits with margin, no contact, clearance).
Small IK error alone never qualifies a configuration. Candidates closer to the
starting configuration (weighted joint distance) come first; near-duplicates
(within `dedupe` rad in every joint) are merged.

With `rotation` (v0.6) the IK fixes the full pose (ArmKinematics.solve_pose_ik): a 1-D
family of solutions remains (the elbow's self-motion), so seeds still matter.
"""

from dataclasses import dataclass, field

import numpy as np

from planning.collision import CollisionChecker
from simulation.kinematics import ArmKinematics


@dataclass
class IKCandidates:
    candidates: list[np.ndarray]                 # valid joint solutions, best first
    errors: list[float]                          # IK position error of each candidate (m)
    seeds_tried: int
    converged: int                               # distinct converged solutions (valid or not)
    rejected: list[str] = field(default_factory=list)  # reasons for converged-but-invalid solutions
    best_error: float = np.inf                   # smallest IK error over all seeds (m)

    @property
    def reachable(self) -> bool:
        return self.converged > 0


def ik_candidates(
    kin: ArmKinematics,
    checker: CollisionChecker,
    target: np.ndarray,
    q_from: np.ndarray,
    random_seeds: int = 12,
    tol: float = 1e-4,
    seed: int = 0,
    weights: np.ndarray | None = None,
    dedupe: float = 0.05,
    max_candidates: int | None = None,
    local_seeds: int = 0,
    local_sigma: float = 0.5,
    rotation: np.ndarray | None = None,
    rot_tol: float = 1e-3,
) -> IKCandidates:
    target = np.asarray(target, float)
    q_from = np.asarray(q_from, float)
    rng = np.random.default_rng(seed)
    seeds = [(q_from, None), (kin.q_rest, None)]
    for _ in range(local_seeds):
        s = np.clip(q_from + rng.normal(0.0, local_sigma, len(q_from)), kin.q_min, kin.q_max)
        seeds.append((s, s))
    for _ in range(random_seeds):
        s = rng.uniform(kin.q_min, kin.q_max)
        seeds.append((s, s))
    w = np.ones(len(q_from)) if weights is None else np.asarray(weights, float)

    solutions, best = [], np.inf
    for s, rest in seeds:
        if rotation is None:
            res = kin.solve_ik(target, s, tol, rest=rest)
        else:
            res = kin.solve_pose_ik(target, rotation, s, tol, rot_tol, rest=rest)
        best = min(best, res.error)
        if res.success and not any(np.all(np.abs(res.q - q) < dedupe) for q, _ in solutions):
            solutions.append((res.q.copy(), res.error))

    valid, rejected = [], []
    for q, err in solutions:
        rep = checker.check(q)
        if rep.valid:
            valid.append((q, err))
        else:
            rejected.append(rep.reason)
    valid.sort(key=lambda c: float(np.linalg.norm((c[0] - q_from) * w)))
    if max_candidates is not None:
        valid = valid[:max_candidates]
    return IKCandidates([q for q, _ in valid], [e for _, e in valid], len(seeds), len(solutions), rejected, best)
