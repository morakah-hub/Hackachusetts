"""Figures for docs/research/v0.4-results.md from saved benchmark results.

    python tools/plot_v04.py [final_dir] [robustness_dir]
    (defaults: results/v0.4_final results/v0.4_robustness)

Writes docs/research/figures/v0.4-*.png. Colours follow the controller, never its
rank (fixed categorical order), and every series also has its own marker, so
identity never rests on colour alone.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "research" / "figures"
# Fixed categorical slots (reference palette order) and markers per controller.
STYLE = {
    "torque_ctc": ("#2a78d6", "o", "CTC (v0.3)"),
    "torque_ctc_i": ("#eda100", "v", "CTC + integral"),
    "torque_ctc_dob": ("#1baf7a", "s", "CTC + observer"),
    "torque_mpc_v03": ("#eb6834", "^", "torque MPC (v0.3)"),
    "torque_mpc": ("#e87ba4", "D", "torque MPC (v0.4)"),
}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def load(d):
    return {(r["scenario"], r["controller"]): r for r in json.load(open(Path(d) / "results.json"))["results"]}


def mismatch(rob):
    scales = [("mass_m25", 0.75), ("mass_m15", 0.85), ("mass_m10", 0.90), ("mass_m05", 0.95), ("nominal", 1.0),
              ("mass_p05", 1.05), ("mass_p10", 1.10), ("mass_p15", 1.15), ("mass_p25", 1.25)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for c, (col, mk, label) in STYLE.items():
        ys = [1e3 * rob[(s, c)]["max_error_rad"] for s, _ in scales]
        ax.plot([x for _, x in scales], ys, color=col, marker=mk, ms=6, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("controller model mass / true mass")
    ax.set_ylabel("max joint error (mrad)")
    ax.set_title("Model mismatch: demo motion, plant always nominal", loc="left", color=INK)
    ax.grid(alpha=0.25, color=MUTED, lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "v0.4-mismatch-sweep.png", dpi=130)
    plt.close(fig)


def disturbances(rob):
    scen = [("push_py", "+y hand"), ("push_ny", "-y hand"), ("push_px", "+x hand"), ("push_nz", "-z hand"),
            ("push_link4", "+y elbow"), ("push_moving", "+y moving"), ("push_mismatch", "+y, mass -15%"),
            ("noise_push", "+y, noisy sensors")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(scen))
    w = 0.8 / len(STYLE)
    for i, (c, (col, mk, label)) in enumerate(STYLE.items()):
        ys = [1e3 * rob[(s, c)]["max_error_rad"] for s, _ in scen]
        ax.bar(x + (i - (len(STYLE) - 1) / 2) * w, ys, w * 0.9, color=col, label=label)
    ax.set_yscale("log")
    ax.set_xticks(x, [n for _, n in scen], rotation=20, ha="right")
    ax.set_ylabel("max joint error (mrad)")
    ax.set_title("External forces (30 N on the hand, 40 N on the elbow, 1 s)", loc="left", color=INK)
    ax.grid(axis="y", alpha=0.25, color=MUTED, lw=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(FIG / "v0.4-disturbance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fast_wrist(fin):
    scen = ["fast", "fast_feasible", "wrist", "wrist_feasible", "wrist_feasible_dsc"]
    ctrls = ["torque_ctc", "torque_ctc_dob", "torque_mpc_v03", "torque_mpc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(scen))
    w = 0.8 / len(ctrls)
    for i, c in enumerate(ctrls):
        col, mk, label = STYLE[c]
        off = (i - (len(ctrls) - 1) / 2) * w
        axes[0].bar(x + off, [1e3 * fin[(s, c)]["max_error_rad"] for s in scen], w * 0.9, color=col, label=label)
        axes[1].bar(x + off, [fin[(s, c)]["max_velocity_ratio"] for s in scen], w * 0.9, color=col, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max joint error (mrad)")
    axes[0].set_title("Tracking: infeasible vs feasible references", loc="left", color=INK)
    axes[1].axhline(1.0, color=MUTED, lw=1, ls="--")
    axes[1].set_ylabel("peak |qd| / v_max")
    axes[1].set_title("Speed limit (dashed = datasheet limit)", loc="left", color=INK)
    for ax in axes:
        ax.set_xticks(x, scen, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25, color=MUTED, lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "v0.4-fast-wrist.png", dpi=130)
    plt.close(fig)


def main():
    final = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "v0.4_final"
    rob = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "v0.4_robustness"
    FIG.mkdir(parents=True, exist_ok=True)
    if (rob / "results.json").exists():
        r = load(rob)
        mismatch(r)
        disturbances(r)
    if (final / "results.json").exists():
        fast_wrist(load(final))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
