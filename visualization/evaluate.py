# visualization/evaluate.py
# Model evaluation & policy quality visualization (publication quality)
# Nature-figure style: restrained palette, English labels, SVG/PDF/PNG export

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STATE_DIM, MIN_PITCH, MAX_PITCH, REWARD_SCALE, GAMMA
from data.dataset import load_dataset
from models.actor import Actor
from models.cql import Critic

# Nature-figure rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "legend.fontsize": 6.5,
    "axes.unicode_minus": False,
})

# Restrained palette
C_DATA   = "#2B5C8F"
C_ACTOR  = "#C44E52"
C_BIAS   = "#D9754A"
C_STATE  = "#3A7CA5"
C_QRAND  = "#7F7F7F"
C_ZERO   = "#AAAAAA"
BG       = "#FFFFFF"
GRID     = "#E8E8E8"


def save_pub(fig, filename, dpi=600):
    """Publication export: SVG + PDF + PNG"""
    fig.savefig(f"{filename}.svg", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=dpi)


def load_model_and_data():
    """Load model and data"""
    s, a, r, ns, d, norm, *_ = load_dataset("PI_OfflineData.csv")

    actor = Actor(STATE_DIM)
    actor.load_state_dict(torch.load("output/actor_best.pth", map_location="cpu"))
    actor.eval()

    critic = Critic(STATE_DIM)
    try:
        critic.load_state_dict(torch.load("output/critic_best.pth", map_location="cpu"))
        critic.eval()
        has_critic = True
    except FileNotFoundError:
        has_critic = False

    return s, a, r, ns, d, norm, actor, critic, has_critic


# ============================================================
# Figure 4: Action comparison (hero for policy fidelity)
# ============================================================
def plot_action_comparison(s, a_true, actor, save_dir="visualization"):
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy().flatten()
    a_true = a_true.flatten()
    errors = a_pred - a_true
    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.mean(errors ** 2))

    fig = plt.figure(figsize=(7.2, 5.6), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1],
                          wspace=0.35, hspace=0.40,
                          left=0.10, right=0.96, top=0.91, bottom=0.10)
    fig.suptitle(
        "Policy Action Fidelity - Actor vs PI Controller",
        fontsize=10, fontweight="bold", x=0.0, y=0.98, ha="left"
    )

    # (a) Scatter: Actor vs PI
    ax = fig.add_subplot(gs[0, 0])
    N_s = min(3000, len(a_true))
    idx = np.random.RandomState(42).choice(len(a_true), N_s, replace=False)
    ax.scatter(a_true[idx], a_pred[idx], s=0.8, alpha=0.25,
               color=C_DATA, edgecolors="none", rasterized=True)
    lims = [a_true.min(), a_true.max()]
    ax.plot(lims, lims, "-", color=C_ACTOR, lw=0.8, alpha=0.6)
    ax.set_xlabel("PI Demand (rad)")
    ax.set_ylabel("Actor Output (rad)")
    ax.set_title("a  Predicted vs True Action")
    ax.text(0.04, 0.96,
            f"MAE = {mae:.4f} rad\nRMSE = {rmse:.4f} rad\nn = {N_s:,}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (b) Error distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(errors, bins=80, color=C_BIAS, alpha=0.7, edgecolor="white", lw=0.2)
    ax.axvline(x=0, color=C_ZERO, linestyle="--", lw=1.0)
    ax.set_xlabel("Prediction Error (rad)")
    ax.set_ylabel("Count")
    ax.set_title("b  Error Distribution")
    ax.text(0.95, 0.92,
            f"mean = {errors.mean():.5f}\nstd = {errors.std():.5f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (c) Time-series overlay (first 500 samples)
    ax = fig.add_subplot(gs[1, 0])
    n_ts = min(500, len(a_true))
    ax.plot(np.arange(n_ts), a_true[:n_ts], color=C_DATA, lw=0.9,
            label="PI (dataset)")
    ax.plot(np.arange(n_ts), a_pred[:n_ts], color=C_ACTOR, lw=0.7, alpha=0.85,
            label="Actor")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Pitch Demand (rad)")
    ax.set_title("c  Time-series Overlay (first %d samples)" % n_ts)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (d) Action distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(a_true, bins=50, alpha=0.45, color=C_DATA, density=True,
            label="PI (dataset)")
    ax.hist(a_pred, bins=50, alpha=0.45, color=C_ACTOR, density=True,
            label="Actor")
    ax.set_xlabel("Pitch Demand (rad)")
    ax.set_ylabel("Density")
    ax.set_title("d  Action Distribution")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    save_pub(fig, f"{save_dir}/action_comparison")
    plt.close(fig)
    print(f"[OK] action_comparison -> {save_dir}/action_comparison.{{svg,pdf,png}}")


# ============================================================
# Figure 5: Q-value based policy quality
# ============================================================
def plot_policy_quality(s, a, actor, critic, has_critic, save_dir="visualization"):
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy().flatten()
    a_true = a.flatten()
    diff = a_pred - a_true

    fig = plt.figure(figsize=(7.2, 4.5), facecolor=BG)
    gs = fig.add_gridspec(2, 3 if has_critic else 2,
                          width_ratios=([1, 1, 0.9] if has_critic else [1, 1]),
                          wspace=0.38, hspace=0.45,
                          left=0.09, right=0.96, top=0.89, bottom=0.12)

    # (a) Action deviation histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(diff, bins=80, color=C_BIAS, alpha=0.7, edgecolor="white", lw=0.2)
    ax.axvline(x=0, color=C_ZERO, linestyle="--", lw=1.0)
    ax.set_xlabel("Delta a = a_actor - a_PI (rad)")
    ax.set_ylabel("Count")
    ax.set_title("a  Action Deviation")
    ax.text(0.95, 0.92,
            f"mean = {diff.mean():.4f}\nstd = {diff.std():.4f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (b) Per-sample deviation
    ax = fig.add_subplot(gs[0, 1])
    n_show = min(1000, len(diff))
    ax.plot(np.arange(n_show), diff[:n_show], color=C_ACTOR, lw=0.5, alpha=0.8)
    ax.axhline(y=0, color=C_ZERO, linestyle="--", lw=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Delta a (rad)")
    ax.set_title("b  Per-sample Deviation")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    if has_critic:
        # (c) Q-value ranking
        ax = fig.add_subplot(gs[0, 2])
        s_t = torch.tensor(s[:2000]).float()
        a_t = torch.tensor(a[:2000]).float()
        a_p = torch.tensor(a_pred[:2000]).float().reshape(-1, 1)
        a_r = torch.rand(2000, 1) * MAX_PITCH
        with torch.no_grad():
            q1_pi, q2_pi = critic(s_t, a_t)
            q_pi = (q1_pi + q2_pi) / 2
            q1_ac, q2_ac = critic(s_t, a_p)
            q_ac = (q1_ac + q2_ac) / 2
            q1_rd, q2_rd = critic(s_t, a_r)
            q_rd = (q1_rd + q2_rd) / 2
        labels = ["PI\n(dataset)", "Actor\npi(s)", "Random\nU(0, pi/2)"]
        values = [q_pi.mean().item(), q_ac.mean().item(), q_rd.mean().item()]
        colors = [C_DATA, C_ACTOR, C_QRAND]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", lw=0.3)
        ax.axhline(y=0, color=C_ZERO, linestyle="--", lw=0.6)
        ax.set_ylabel("Mean Q-value")
        ax.set_title("c  Q-value Ranking")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.1f}", ha="center", fontsize=6, fontweight="bold")
        ax.grid(True, alpha=0.25, lw=0.3, color=GRID, axis="y")

    # (d) Action vs GenSpeed
    row1 = 1
    ax = fig.add_subplot(gs[row1, 0])
    s0 = s[:3000, 0]
    ax.scatter(s0, a_true[:3000], s=0.6, alpha=0.25, color=C_DATA,
               edgecolors="none", rasterized=True, label="PI")
    ax.scatter(s0, a_pred[:3000], s=0.6, alpha=0.25, color=C_ACTOR,
               edgecolors="none", rasterized=True, label="Actor")
    ax.set_xlabel("GenSpeed (norm.)")
    ax.set_ylabel("Pitch (rad)")
    ax.set_title("d  Action vs GenSpeed")
    ax.legend(fontsize=6, loc="upper right", markerscale=3)
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (e) Action vs WindSpeed
    ax = fig.add_subplot(gs[row1, 1])
    s2 = s[:3000, 2]
    ax.scatter(s2, a_true[:3000], s=0.6, alpha=0.25, color=C_DATA,
               edgecolors="none", rasterized=True, label="PI")
    ax.scatter(s2, a_pred[:3000], s=0.6, alpha=0.25, color=C_ACTOR,
               edgecolors="none", rasterized=True, label="Actor")
    ax.set_xlabel("WindSpeed (norm.)")
    ax.set_ylabel("Pitch (rad)")
    ax.set_title("e  Action vs WindSpeed")
    ax.legend(fontsize=6, loc="upper right", markerscale=3)
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    save_pub(fig, f"{save_dir}/policy_quality")
    plt.close(fig)
    print(f"[OK] policy_quality -> {save_dir}/policy_quality.{{svg,pdf,png}}")


# ============================================================
# Figure 6: State-action coverage
# ============================================================
def plot_state_coverage(s, a, save_dir="visualization"):
    state_names = [
        "GenSpeed",
        "Speed Error",
        "WindSpeed",
        "PitchAngle",
        "TowerAcc",
    ]

    fig = plt.figure(figsize=(7.2, 5.0), facecolor=BG)
    gs = fig.add_gridspec(2, 3, wspace=0.38, hspace=0.42,
                          left=0.09, right=0.96, top=0.91, bottom=0.11)
    fig.suptitle(
        "State-Action Coverage Analysis",
        fontsize=10, fontweight="bold", x=0.0, y=0.97, ha="left"
    )

    for i in range(5):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.scatter(s[:3000, i], a[:3000].flatten(), s=0.5, alpha=0.20,
                   color=C_STATE, edgecolors="none", rasterized=True)
        ax.set_xlabel(state_names[i], fontsize=6.5)
        ax.set_ylabel("Pitch (rad)", fontsize=6.5)
        ax.set_title(f"{chr(97+i)}  Dim {i}: {state_names[i]}", fontsize=7.5)
        ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # (f) Action histogram
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(a.flatten(), bins=50, color=C_STATE, alpha=0.7, edgecolor="white", lw=0.2)
    ax.set_xlabel("Pitch Demand (rad)")
    ax.set_ylabel("Count")
    ax.set_title("f  Action Distribution")
    ax.text(0.95, 0.92,
            f"range = [{a.min():.3f}, {a.max():.3f}]\nmean = {a.mean():.3f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    save_pub(fig, f"{save_dir}/state_action_coverage")
    plt.close(fig)
    print(f"[OK] state_action_coverage -> {save_dir}/state_action_coverage.{{svg,pdf,png}}")


# ============================================================
def print_evaluation_metrics(s, a, r, actor):
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy().flatten()
    a_true = a.flatten()
    mae = np.abs(a_pred - a_true).mean()
    rmse = np.sqrt(np.mean((a_pred - a_true) ** 2))
    max_err = np.abs(a_pred - a_true).max()

    print("\n" + "=" * 62)
    print("  Evaluation Metrics")
    print("=" * 62)
    print(f"  {'MAE':<40s}: {mae:.6f} rad")
    print(f"  {'RMSE':<40s}: {rmse:.6f} rad")
    print(f"  {'Max Error':<40s}: {max_err:.6f} rad")
    print(f"  {'Dataset Mean':<40s}: {a_true.mean():.6f} rad")
    print(f"  {'Actor Mean':<40s}: {a_pred.mean():.6f} rad")
    print(f"  {'Dataset Std':<40s}: {a_true.std():.6f} rad")
    print(f"  {'Actor Std':<40s}: {a_pred.std():.6f} rad")
    print(f"  {'Dataset Range':<40s}: [{a_true.min():.4f}, {a_true.max():.4f}]")
    print(f"  {'Actor Range':<40s}: [{a_pred.min():.4f}, {a_pred.max():.4f}]")
    print(f"  {'Pitch Limit':<40s}: [{MIN_PITCH}, {MAX_PITCH}]")
    print("=" * 62)

    if mae < 0.01:
        verdict = "EXCELLENT - RL policy highly consistent with PI controller"
    elif mae < 0.05:
        verdict = "GOOD - RL policy mostly consistent with PI controller"
    elif mae < 0.1:
        verdict = "FAIR - Noticeable deviation detected, check training"
    else:
        verdict = "POOR - Large deviation, training may need adjustment"
    print(f"  Verdict: {verdict}")
    print("=" * 62 + "\n")


# ============================================================
if __name__ == "__main__":
    print("Loading model & data...")
    s, a, r, ns, d, norm, actor, critic, has_critic = load_model_and_data()
    print(f"  Loaded: {len(s):,} samples  |  "
          f"Critic: {'available' if has_critic else 'unavailable'}\n")

    print_evaluation_metrics(s, a, r, actor)
    plot_action_comparison(s, a, actor)
    plot_policy_quality(s, a, actor, critic, has_critic)
    plot_state_coverage(s, a)

    print("Done - SVG + PDF + PNG exported.\n")
