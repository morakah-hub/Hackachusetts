"""Audit: why does the servo MPC hit its iteration budget in the wrist benchmark scenario?"""

import sys
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import controllers.mpc as M  # noqa: E402
from simulation.benchmark import BENCHMARK_CONFIG, scenario_config  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

cap = {}
orig = M.JointMPCController._qp_data


def spy(self, *a):
    out = orig(self, *a)
    cap["args"], cap["out"] = a, out
    return out


M.JointMPCController._qp_data = spy
bench = yaml.safe_load(open(BENCHMARK_CONFIG))
r = SimulationRunner(scenario_config(load_config(), bench, "mpc", "wrist"))
mpc = r.controller
while True:
    r.step()
    if mpc.stats.updates and mpc.stats.updates[-1].fallback:
        break
upd = mpc.stats.updates[-1]
print(f"first fallback t={upd.time:.2f}s status={upd.status}")
lin, x0, u_prev, tau_g, refs = cap["args"]
Ax, h, lo, hi = cap["out"]
n, N, nx = mpc.dof, mpc.N, mpc.nx
print("measured qd / v_max:", np.round(x0[n:] / mpc.v_bounds[:, 1], 2))
print("tracking error at x0 (rad):", np.round(x0[:n] - refs[0, :n], 3))
rho = mpc.kd / mpc.kp
sens = np.diag(np.eye(n) - lin.B[:n] - rho * lin.B[n:])
print("d(sustained torque row)/du0 per joint (1 = fully controllable):", np.round(sens, 3))


def solve(label, P, q, A, l, u, **st):
    p = osqp.OSQP()
    p.setup(P, q, A, l, u, **(mpc.settings | st))
    res = p.solve(raise_error=False)
    print(f"  {label:38s} {res.info.status:28s} iter {res.info.iter:6d}")
    return res


A = sp.csc_matrix((Ax, mpc._csc_indices, mpc._csc_indptr), shape=(mpc.nc, mpc.nz))
res = solve("as built, max_iter 20000", mpc._P, 2 * h, A, lo, hi, max_iter=20000)
z = res.x
S = z[N * (nx + mpc.nu):].reshape(N, -1)
print("  slack used: states max %.3g, torque rows max %.3g" % (S[:, :nx].max(), S[:, nx:].max()))
Az = A @ z
y0 = mpc._row_y
m = N * mpc.ny
ylo, yhi = Az[y0:y0 + m] - lo[y0:y0 + m], hi[y0 + m:y0 + 2 * m] - Az[y0 + m:y0 + 2 * m]
act = (np.minimum(np.abs(ylo), np.abs(yhi)) < 1e-6).reshape(N, -1)
print("  active soft rows per stage [pos, vel, torque]:",
      [(int(a[:n].sum()), int(a[n:nx].sum()), int(a[nx:].sum())) for a in act][:6], "...")
slew = slice(mpc._row_slew, mpc._row_slew + N * n)
act_slew = np.isclose(Az[slew], lo[slew], atol=1e-7) | np.isclose(Az[slew], hi[slew], atol=1e-7)
print("  active slew rows:", int(act_slew.sum()), "of", N * n)

Ax2 = Ax.copy()
Ax2[np.abs(Ax2) < 1e-6] = 0.0
solve("tiny A/B entries zeroed", mpc._P, 2 * h, sp.csc_matrix((Ax2, mpc._csc_indices, mpc._csc_indptr), shape=A.shape), lo, hi, max_iter=20000)
solve("scaling=50", mpc._P, 2 * h, A, lo, hi, max_iter=20000, scaling=50)
solve("rho=1e-3", mpc._P, 2 * h, A, lo, hi, max_iter=20000, rho=1e-3)
solve("adaptive_rho_interval=25", mpc._P, 2 * h, A, lo, hi, max_iter=20000, adaptive_rho_interval=25)
solve("adaptive_rho_tolerance=2", mpc._P, 2 * h, A, lo, hi, max_iter=20000, adaptive_rho_tolerance=2)
solve("eps 1e-3", mpc._P, 2 * h, A, lo, hi, max_iter=20000, eps_abs=1e-3, eps_rel=1e-3)

print("\nAblation on the same captured state (QP rebuilt with modified terms):")
from controllers.pid import JointPIDController  # noqa: E402


def rebuilt(label, weights=None, **kw):
    w = weights or {}
    base = dict(position=mpc.w.position, velocity=mpc.w.velocity, terminal=mpc.w.terminal, effort=mpc.w.effort,
                command_rate=mpc.w.command_rate, slack_quadratic=mpc.w.slack_quadratic, slack_linear=mpc.w.slack_linear)
    c = M.JointMPCController(mpc.model, JointPIDController(7), horizon=mpc.N, weights=M.MPCWeights(**(base | w)),
                             velocity_limits=mpc.v_bounds[:, 1], joint_limits=r.engine.joint_limits, **kw)
    Ax_, h_, lo_, hi_ = orig(c, lin, x0, u_prev, tau_g, refs)
    A_ = sp.csc_matrix((Ax_, c._csc_indices, c._csc_indptr), shape=(c.nc, c.nz))
    solve(label, c._P, 2 * h_, A_, lo_, hi_, max_iter=20000)


rebuilt("baseline (rebuilt)")
rebuilt("effort weight 0", {"effort": 0.0})
rebuilt("effort weight 0.01", {"effort": 0.01})
rebuilt("torque limit off (fraction 1e3)", torque_limit_fraction=1e3)
rebuilt("effort 0 + torque limit off", {"effort": 0.0}, torque_limit_fraction=1e3)
rebuilt("command_rate 1e4", {"command_rate": 1e4})
rebuilt("position 1e3", {"position": 1e3})

print("\nCondensed formulation of the identical QP (states eliminated in numpy):")
nu = mpc.nu
nz = mpc.nz
nX, nU = N * nx, N * nu
nS = nz - nX - nU
# X = G @ [U; S] + g   (states as affine function of inputs; slack unaffected)
Gx = np.zeros((nX, nU))
gx = np.zeros(nX)
Ak = np.eye(nx)
c_eff = lo[mpc._row_dyn + nx: mpc._row_dyn + 2 * nx] if N > 1 else None
x = lo[mpc._row_dyn: mpc._row_dyn + nx].copy()     # A x0 + c (+ d_hat)
c = lin.c + mpc._d_hat
prev = np.zeros((nx, nU))
for k in range(N):
    blk = prev @ np.eye(nU) if k == 0 else lin.A @ prev
    blk = blk.copy()
    blk[:, k * nu:(k + 1) * nu] += lin.B
    Gx[k * nx:(k + 1) * nx] = blk
    gx[k * nx:(k + 1) * nx] = x
    prev = blk
    x = lin.A @ x + c
T = np.zeros((nz, nU + nS))
T[:nX, :nU] = Gx
T[nX:nX + nU, :nU] = np.eye(nU)
T[nX + nU:, nU:] = np.eye(nS)
t = np.zeros(nz)
t[:nX] = gx
Pf = (mpc._P + sp.triu(mpc._P, 1).T).toarray()
Pc = T.T @ Pf @ T
qc = T.T @ (Pf @ t + 2 * h)
Arows = A.toarray()[mpc._row_u:]          # all non-dynamics rows
Ac = Arows @ T
off = Arows @ t
lc, uc = lo[mpc._row_u:] - off, hi[mpc._row_u:] - off
print(f"  condensed size: {Pc.shape[0]} vars, {Ac.shape[0]} rows; cond(P) = {np.linalg.cond(Pc + 0 * np.eye(len(Pc))):.2e}")
for label, st in [("condensed, default settings", {}), ("condensed, max_iter 500", {"max_iter": 500})]:
    p = osqp.OSQP()
    p.setup(sp.triu(sp.csc_matrix(Pc), format="csc"), qc, sp.csc_matrix(Ac), lc, uc, **(mpc.settings | {"max_iter": 20000} | st))
    rc = p.solve(raise_error=False)
    print(f"  {label:38s} {rc.info.status:28s} iter {rc.info.iter:6d}  run {1e3 * rc.info.run_time:.1f} ms")
    zc = T @ rc.x + t
    print(f"    |u0 condensed - u0 sparse| = {np.abs(zc[nX:nX + nu] - z[nX:nX + nu]).max():.2e} rad")
