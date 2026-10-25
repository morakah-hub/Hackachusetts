"""Computed figures for the technical guide (docs/guide_src/figures/fig_*.svg).

Educational figures only: small, fast computations (no benchmark runs). Figures that use project
code import it read-only (tasks/trajectory.py for the timing laws). Toy examples (2-link arm,
2-D RRT, 1-DOF MPC, toy PID) are illustrations, labelled as such in the guide; they are not
results of the Robotics Lab software.

    .\\.venv\\Scripts\\python.exe docs\\guide_src\\make_figures.py
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import osqp  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from tasks.trajectory import DoubleS, quintic  # noqa: E402

OUT = HERE / "figures"
plt.rcParams.update({"svg.fonttype": "none", "font.family": "Segoe UI", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 100})
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#1f5f8b", "#d9822b", "#2e8b57", "#c0392b", "#7f8c8d", "#7a4fa3"


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ C-space of a 2-link arm


def fig_cspace():
    L1, L2 = 1.0, 0.8
    obst = (1.05, 0.65, 0.22)  # circle obstacle (x, y, r) in the workspace

    def collides(t1, t2):
        p0 = np.zeros(2)
        p1 = np.array([L1 * math.cos(t1), L1 * math.sin(t1)])
        p2 = p1 + np.array([L2 * math.cos(t1 + t2), L2 * math.sin(t1 + t2)])
        c = np.array(obst[:2])
        for a, b in ((p0, p1), (p1, p2)):
            ab = b - a
            s = np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0, 1)
            if np.linalg.norm(a + s * ab - c) < obst[2]:
                return True
        return False

    n = 220
    th = np.linspace(-math.pi, math.pi, n)
    grid = np.array([[collides(a, b) for a in th] for b in th])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax1.set_aspect("equal")
    ax1.add_patch(plt.Circle(obst[:2], obst[2], color=RED, alpha=0.6))
    for (t1, t2), col, lab in (((0.2, 0.3), GREEN, "q_A"), ((1.4, -0.6), BLUE, "q_B")):
        p1 = (L1 * math.cos(t1), L1 * math.sin(t1))
        p2 = (p1[0] + L2 * math.cos(t1 + t2), p1[1] + L2 * math.sin(t1 + t2))
        ax1.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "-o", color=col, lw=2.5, ms=4)
        ax1.text(p2[0] + 0.05, p2[1], lab, color=col)
    ax1.set_xlim(-0.4, 2.0); ax1.set_ylim(-0.6, 1.9)
    ax1.set_title("Workspace (x, y)"); ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax2.imshow(grid, origin="lower", extent=[-math.pi, math.pi, -math.pi, math.pi], cmap="Reds", alpha=0.65, aspect="auto")
    ax2.plot([0.2], [0.3], "o", color=GREEN); ax2.text(0.28, 0.35, "q_A", color=GREEN)
    ax2.plot([1.4], [-0.6], "o", color=BLUE); ax2.text(1.48, -0.55, "q_B", color=BLUE)
    ax2.plot([0.2, 1.4], [0.3, -0.6], "--", color=GREY)
    ax2.set_title("Configuration space (θ1, θ2)"); ax2.set_xlabel("θ1 (rad)"); ax2.set_ylabel("θ2 (rad)")
    save(fig, "fig_cspace")


# ------------------------------------------------------------------ 2-D RRT-Connect illustration


def fig_rrt():
    rng = np.random.default_rng(4)
    boxes = [(0.42, 0.0, 0.58, 0.62), (0.15, 0.75, 0.75, 0.85)]

    def free(p):
        return all(not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1) for x0, y0, x1, y1 in boxes) and 0 <= p[0] <= 1 and 0 <= p[1] <= 1

    def edge_free(a, b):
        return all(free(a + (b - a) * s) for s in np.linspace(0, 1, 25))

    start, goal, step = np.array([0.15, 0.2]), np.array([0.85, 0.25]), 0.06
    trees = [{"q": [start], "p": [-1]}, {"q": [goal], "p": [-1]}]

    def extend(t, target):
        q = np.array(t["q"]); i = int(np.argmin(np.linalg.norm(q - target, axis=1)))
        d = np.linalg.norm(target - q[i])
        new = target.copy() if d <= step else q[i] + (target - q[i]) * step / d
        if not free(new) or not edge_free(q[i], new):
            return "trapped", None
        t["q"].append(new); t["p"].append(i)
        return ("reached" if d <= step else "advanced"), len(t["q"]) - 1

    a, b, path = 0, 1, None
    for it in range(3000):
        r = rng.uniform(0, 1, 2)
        st, j = extend(trees[a], r)
        if st != "trapped":
            target = trees[a]["q"][j]
            while True:
                st2, k = extend(trees[b], target)
                if st2 != "advanced":
                    break
            if st2 == "reached":
                def branch(t, i):
                    out = []
                    while i >= 0:
                        out.append(t["q"][i]); i = t["p"][i]
                    return out
                pa, pb = branch(trees[a], j), branch(trees[b], k)
                path = (pa[::-1] + pb[1:]) if a == 0 else (pb[::-1] + pa[1:])
                break
        a, b = b, a
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.set_aspect("equal"); ax.grid(False)
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=RED, alpha=0.55))
    for t, col in ((trees[0], GREEN), (trees[1], BLUE)):
        for q, p in zip(t["q"], t["p"]):
            if p >= 0:
                ax.plot([q[0], t["q"][p][0]], [q[1], t["q"][p][1]], color=col, lw=0.8, alpha=0.8)
    if path:
        P = np.array(path)
        ax.plot(P[:, 0], P[:, 1], color=ORANGE, lw=2.4, label="raw path")
        # greedy prune (the first pass of planning/shortcut.py)
        out, i = [path[0]], 0
        while i < len(path) - 1:
            for jj in range(len(path) - 1, i, -1):
                if jj == i + 1 or edge_free(path[i], path[jj]):
                    out.append(path[jj]); i = jj; break
        S = np.array(out)
        ax.plot(S[:, 0], S[:, 1], "-o", color="k", lw=1.6, ms=3, label="after pruning")
    ax.plot(*start, "s", color=GREEN, ms=8); ax.plot(*goal, "s", color=BLUE, ms=8)
    ax.text(start[0], start[1] - 0.06, "start", ha="center", color=GREEN)
    ax.text(goal[0], goal[1] - 0.06, "goal", ha="center", color=BLUE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"RRT-Connect in 2-D (illustration): {len(trees[0]['q']) + len(trees[1]['q'])} nodes")
    save(fig, "fig_rrt")


# ------------------------------------------------------------------ timing laws from tasks/trajectory.py


def fig_timing():
    tau = np.linspace(0, 1, 400)
    q = np.array([quintic(t) for t in tau])
    ds = DoubleS(v=1.4, a=4.0, j=40.0)
    t = np.linspace(0, ds.duration, 400)
    d = np.array([ds(x) for x in t])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for col, lab, k in ((BLUE, "s", 0), (ORANGE, "ds/dτ", 1), (GREEN, "d²s/dτ²", 2)):
        axs[0].plot(tau, q[:, k], color=col, label=lab)
    axs[0].set_title("Minimum-jerk quintic (tasks/trajectory.py quintic)"); axs[0].set_xlabel("normalised time τ = t / T")
    axs[0].legend(fontsize=8)
    for col, lab, k in ((BLUE, "s", 0), (ORANGE, "s'", 1), (GREEN, "s''", 2)):
        axs[1].plot(t, d[:, k], color=col, label=lab)
    axs[1].set_title("Double-S (DoubleS(v=1.4, a=4, j=40))"); axs[1].set_xlabel("time (s)"); axs[1].legend(fontsize=8)
    save(fig, "fig_timing")


# ------------------------------------------------------------------ worked 1-DOF MPC example


def mpc_example(print_numbers=True):
    I, Ts = 0.5, 0.1
    A = np.array([[1.0, Ts], [0.0, 1.0]])
    B = np.array([[Ts**2 / (2 * I)], [Ts / I]])
    Q = np.diag([10.0, 1.0]); R = 0.1; N = 20; umax = 1.0
    r = np.array([1.0, 0.0])

    def solve(x0, constrained=True):
        nx, nu = 2, 1
        nz = N * nx + N * nu
        P = sp.block_diag([sp.kron(sp.eye(N), Q), R * sp.eye(N)], format="csc")
        q = np.r_[np.tile(-Q @ r, N), np.zeros(N)]
        Aeq = sp.lil_matrix((N * nx, nz)); beq = np.zeros(N * nx)
        for k in range(N):
            Aeq[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = np.eye(nx)
            if k > 0:
                Aeq[k * nx:(k + 1) * nx, (k - 1) * nx:k * nx] = -A
            Aeq[k * nx:(k + 1) * nx, N * nx + k] = -B[:, 0]
        beq[:nx] = A @ x0
        Ain = sp.hstack([sp.csc_matrix((N, N * nx)), sp.eye(N)])
        Afull = sp.vstack([Aeq, Ain]).tocsc()
        lim = umax if constrained else 1e9
        l = np.r_[beq, -lim * np.ones(N)]; u = np.r_[beq, lim * np.ones(N)]
        prob = osqp.OSQP()
        prob.setup(2 * P, 2 * q, Afull, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8, polishing=True)
        z = prob.solve().x
        return z[:N * nx].reshape(N, nx), z[N * nx:]

    x = np.zeros(2)
    X0, U0 = solve(x)
    Xu, Uu = solve(x, constrained=False)
    hist_x, hist_u = [x.copy()], []
    for k in range(40):
        _, U = solve(x)
        u = U[0]
        hist_u.append(u)
        x = A @ x + B[:, 0] * u
        hist_x.append(x.copy())
    hist_x = np.array(hist_x)
    if print_numbers:
        print("MPC example: A", A.tolist(), "B", B[:, 0].tolist())
        print("first plan torques (constrained):", np.round(U0[:6], 4).tolist())
        print("first plan torques (unconstrained):", np.round(Uu[:6], 3).tolist())
        print("predicted theta (constrained):", np.round(X0[:6, 0], 4).tolist())
        print("closed loop theta at k=10,20,30:", np.round(hist_x[[10, 20, 30], 0], 4).tolist())
        print("closed loop first 5 applied torques", np.round(hist_u[:5], 4).tolist())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    tt = np.arange(N + 1) * Ts
    axs[0].plot(tt, np.r_[0, X0[:, 0]], "-o", ms=3, color=BLUE, label="θ predicted, |τ| ≤ 1 Nm")
    axs[0].plot(tt, np.r_[0, Xu[:, 0]], "--", color=GREY, label="θ predicted, unconstrained")
    axs[0].step(tt[:-1], U0, where="post", color=ORANGE, label="τ plan, |τ| ≤ 1 Nm")
    axs[0].axhline(1.0, color=GREEN, lw=0.8, ls=":")
    axs[0].set_title("First MPC solve (k = 0): plan over N = 20"); axs[0].set_xlabel("prediction time (s)")
    axs[0].legend(fontsize=7)
    tc = np.arange(len(hist_x)) * Ts
    axs[1].plot(tc, hist_x[:, 0], color=BLUE, label="θ closed loop")
    axs[1].step(tc[:-1], hist_u, where="post", color=ORANGE, label="τ applied")
    axs[1].axhline(1.0, color=GREEN, lw=0.8, ls=":", label="reference")
    axs[1].set_title("Closed loop: re-solve every 0.1 s, apply first τ"); axs[1].set_xlabel("time (s)")
    axs[1].legend(fontsize=7)
    save(fig, "fig_mpc_example")


# ------------------------------------------------------------------ receding horizon sketch


def fig_receding():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.grid(False)
    t = np.linspace(0, 3, 300)
    ref = np.where(t < 1.0, 0.0, 1.0 - np.exp(-(t - 1.0) * 3))
    ax.plot(t, ref, color=GREEN, lw=1.5, label="reference r(t)")
    past = t <= 1.2
    ax.plot(t[past], 0.6 * np.clip(t[past] - 0.8, 0, None) ** 1.3, color=BLUE, lw=2.2, label="measured (past)")
    tk = 1.2
    tp = np.linspace(tk, tk + 1.0, 60)
    x0 = 0.6 * (tk - 0.8) ** 1.3
    for off, alpha, lab in ((0.0, 1.0, "plan at t_k (horizon N·Ts)"), (0.2, 0.35, "plan at t_k+1")):
        tp2 = tp + off
        y = x0 + (1 - np.exp(-(tp2 - tk) * 2.5)) * (0.95 - x0) + off * 0.25
        ax.plot(tp2, y, "--", color=ORANGE, alpha=alpha, lw=1.8, label=lab)
    ax.axvline(tk, color=GREY, lw=0.8); ax.text(tk + 0.02, -0.15, "t_k (now)", color=GREY)
    ax.annotate("", xy=(tk + 1.0, 1.15), xytext=(tk, 1.15), arrowprops=dict(arrowstyle="<->", color=GREY))
    ax.text(tk + 0.35, 1.19, "prediction horizon", color=GREY)
    ax.set_ylim(-0.25, 1.35); ax.set_xlim(0, 3); ax.set_yticks([]); ax.set_xlabel("time")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig_receding")


# ------------------------------------------------------------------ toy feedback comparison


def fig_toy_pid():
    # one joint: I th'' + b th' = tau - m g l sin(th), target 1 rad; explicit Euler, dt 1 ms
    I, b, mgl, dt = 0.5, 0.2, 2.0, 0.001
    def run(kp, kd, ki, grav_ff=False):
        th, om, integ, out = 0.0, 0.0, 0.0, []
        for k in range(4000):
            e = 1.0 - th
            integ += e * dt
            tau = kp * e - kd * om + ki * integ + (mgl * math.sin(th) if grav_ff else 0.0)
            om += dt * (tau - b * om - mgl * math.sin(th)) / I
            th += dt * om
            out.append(th)
        return np.array(out)
    t = np.arange(4000) * dt
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.plot(t, run(20, 0, 0), color=GREY, label="P (kp = 20): oscillates, steady error")
    ax.plot(t, run(20, 4, 0), color=BLUE, label="PD (kd = 4): damped, steady error from gravity")
    ax.plot(t, run(20, 4, 30), color=ORANGE, label="PID (ki = 30): error removed slowly")
    ax.plot(t, run(20, 4, 0, True), color=GREEN, label="PD + gravity compensation")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("time (s)"); ax.set_ylabel("joint angle (rad)"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Toy single joint under gravity (illustration, not the Panda)")
    save(fig, "fig_toy_pid")


# ------------------------------------------------------------------ disk usage


def fig_disk():
    items = [(".venv (third-party packages)", 409), ("robots/ (Menagerie meshes)", 34.6), ("results/ (git-ignored outputs)", 27.1),
             (".git (history)", 13.8), ("docs/ (reports, data, figures)", 5.8), ("project code, configs, tests (incl. __pycache__)", 2.0)]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    names = [n for n, _ in items][::-1]; vals = [v for _, v in items][::-1]
    ax.barh(names, vals, color=[GREEN, BLUE, GREY, ORANGE, PURPLE, RED][::-1])
    ax.set_xscale("log"); ax.set_xlabel("MB (log scale), measured 19 Oct 2026 before this guide was added")
    for i, v in enumerate(vals):
        ax.text(v * 1.08, i, f"{v:g} MB", va="center", fontsize=8)
    ax.set_xlim(1, 1500)
    save(fig, "fig_disk")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_cspace()
    fig_rrt()
    fig_timing()
    mpc_example()
    fig_receding()
    fig_toy_pid()
    fig_disk()
