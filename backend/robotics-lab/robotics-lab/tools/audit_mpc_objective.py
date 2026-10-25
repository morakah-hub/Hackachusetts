"""Audit: the OSQP objective and constraints of JointMPCController equal the documented cost/model.

Captures a QP mid-motion and compares 1/2 z'Pz + q'z against a direct, term-by-term
evaluation of the cost for random z (differences cancel the constant term).
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import controllers.mpc as M  # noqa: E402
from simulation.runner import SimulationRunner, load_config  # noqa: E402

cap = {}
orig = M.JointMPCController._qp_data


def spy(self, lin, x0, u_prev, tau_g, refs):
    out = orig(self, lin, x0, u_prev, tau_g, refs)
    cap.update(lin=lin, x0=x0, u_prev=u_prev, tau_g=tau_g, refs=refs, out=out)
    return out


M.JointMPCController._qp_data = spy
cfg = load_config()
cfg["controller"]["type"] = "joint_mpc"
r = SimulationRunner(cfg)
while r.engine.time < 1.5:
    r.step()
mpc = r.controller
Ax, h, lo, hi = cap["out"]
N, nx, nu, n = mpc.N, mpc.nx, mpc.nu, mpc.dof
rho = mpc.kd / mpc.kp
P = mpc._P + sp.triu(mpc._P, 1).T  # OSQP stores the upper triangle


def direct_cost(z):
    X = z[: N * nx].reshape(N, nx)
    U = z[N * nx : N * (nx + nu)].reshape(N, nu)
    S = z[N * (nx + nu) :].reshape(N, -1) if mpc.soft else np.zeros((N, 1))
    J = 0.0
    for k in range(N):
        w = mpc.w.terminal if k == N - 1 else 1.0
        e = X[k] - cap["refs"][k]
        J += w * e @ (mpc._Qx * e)
        tau_off = U[k] - X[k, :n] - rho * X[k, n:] - cap["tau_g"] / mpc.kp
        J += tau_off @ (mpc._te2 * tau_off)
        du = U[k] - (cap["u_prev"] if k == 0 else U[k - 1])
        J += du @ (mpc._wdu * du)
    J += mpc.w.slack_quadratic * np.sum(S**2) + mpc.w.slack_linear * np.sum(S)
    return J


def qp_cost(z):
    return 0.5 * z @ (P @ z) + 2 * h @ z


rng = np.random.default_rng(0)
errs = []
for _ in range(20):
    z1, z2 = rng.normal(size=mpc.nz), rng.normal(size=mpc.nz)
    d_direct = direct_cost(z1) - direct_cost(z2)
    d_qp = qp_cost(z1) - qp_cost(z2)
    errs.append(abs(d_direct - d_qp) / max(1.0, abs(d_direct)))
print(f"objective: max relative mismatch over 20 random pairs = {max(errs):.2e}")

# Constraints: dynamics rows reproduce x[k+1] = A x[k] + B u[k] + c.
A = sp.csc_matrix((Ax, mpc._csc_indices, mpc._csc_indptr), shape=(mpc.nc, mpc.nz))
lin = cap["lin"]
U = rng.normal(size=(N, nu)) * 0.01 + cap["u_prev"]
X = lin.predict(cap["x0"], U)
z = np.zeros(mpc.nz)
z[: N * nx] = X.ravel()
z[N * nx : N * (nx + nu)] = U.ravel()
rows = A @ z
dyn = slice(mpc._row_dyn, mpc._row_dyn + N * nx)
print(f"dynamics rows: max |Az - rhs| for model-consistent z = {np.abs(rows[dyn] - lo[dyn]).max():.2e}")
# Torque rows equal scaled sustained torque.
y0 = mpc._row_y
tau_rows = rows[y0 + nx : y0 + mpc.ny]
tau_direct = (mpc.kp * (U[0] - X[0, :n]) - mpc.kd * X[0, n:]) / np.abs(mpc.force_limits).max(axis=1)
print(f"torque rows: max |row - tau/tau_max| = {np.abs(tau_rows - tau_direct).max():.2e}")
# Conditioning
Pd = P.toarray()
ev = np.linalg.eigvalsh(Pd)
print(f"P eigenvalues: min {ev.min():.3e}, max {ev.max():.3e}, cond (pos part) {ev.max() / ev[ev > 1e-9].min():.2e}")
print(f"constraint matrix |entries|: min {np.abs(A.data[A.data != 0]).min():.2e}, max {np.abs(A.data).max():.2e}")
