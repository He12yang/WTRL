# visualization/plot_training.py
# Training curve visualization (publication quality)
# Nature-figure style: restrained palette, English labels, SVG/PDF/PNG export

import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
C_ACTOR   = "#C44E52"
C_CRITIC  = "#3A7CA5"
C_BELLMAN = "#D9754A"
C_CQL     = "#8B6BAE"
C_QDATA   = "#2B5C8F"
C_QRAND   = "#7F7F7F"
C_RETURN  = "#5B8C5A"
C_ALPHA   = "#D4A04A"
BG        = "#FFFFFF"
GRID      = "#E8E8E8"


def save_pub(fig, filename, dpi=600):
    """Publication export: SVG + PDF + PNG"""
    fig.savefig(f"{filename}.svg", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=dpi)


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def smooth(y, alpha=0.15):
    """Exponential moving average"""
    s = np.array(y, dtype=float)
    for i in range(1, len(s)):
        s[i] = alpha * s[i] + (1 - alpha) * s[i - 1]
    return s


# ============================================================
# Figure 1: Training loss dynamics
# ============================================================
def plot_losses(history, save_dir="visualization"):
    epochs = np.array(history["epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), facecolor=BG)
    fig.suptitle(
        "CQL Training Loss Dynamics",
        fontsize=10, fontweight="bold", x=0.0, y=0.99, ha="left"
    )

    # (a) Actor + BC loss
    ax = axes[0, 0]
    ax.plot(epochs, smooth(history["actor_loss"]), color=C_ACTOR, lw=1.0,
            label="Actor loss")
    bc = np.array(history["bc_loss"])
    ax.plot(epochs[1:], smooth(bc[1:]), color=C_CRITIC, lw=0.7, ls="--",
            label="BC loss")
    ax.set_ylabel("Actor Loss")
    ax.set_title("a  Actor / BC Convergence")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax.text(0.98, 0.65, f"final = {history['actor_loss'][-1]:.2e}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_ACTOR)
    ax.text(0.98, 0.45, f"BC = {bc[-1]:.2e}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_CRITIC)

    # (b) Critic total loss
    ax = axes[0, 1]
    ax.plot(epochs, smooth(history["critic_loss"]), color=C_CRITIC, lw=1.0)
    ax.set_ylabel("Critic Loss")
    ax.set_title("b  Critic Total Loss")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax.text(0.98, 0.92, f"final = {history['critic_loss'][-1]:.1f}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_CRITIC)

    # (c) Bellman TD error
    ax = axes[1, 0]
    ax.plot(epochs, smooth(history["bellman_loss"]), color=C_BELLMAN, lw=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bellman Loss")
    ax.set_title("c  Bellman TD Error")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax.text(0.98, 0.92, f"final = {history['bellman_loss'][-1]:.1f}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_BELLMAN)

    # (d) CQL penalty + alpha
    ax = axes[1, 1]
    ax.plot(epochs, smooth(history["cql_loss"]), color=C_CQL, lw=1.0,
            label="CQL loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CQL Loss", color=C_CQL)
    ax.tick_params(axis="y", labelcolor=C_CQL)
    ax.set_title("d  CQL Penalty & Lagrange alpha")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax2 = ax.twinx()
    ax2.plot(epochs, smooth(history["cql_alpha"]), color=C_ALPHA, lw=0.8,
             ls=":", label="alpha")
    ax2.set_ylabel("CQL alpha", color=C_ALPHA)
    ax2.tick_params(axis="y", labelcolor=C_ALPHA)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_ALPHA)
    ax2.spines["right"].set_linewidth(0.6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="center right")
    ax.text(0.98, 0.70, f"CQL = {history['cql_loss'][-1]:.2f}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_CQL)
    ax.text(0.98, 0.50, f"alpha = {history['cql_alpha'][-1]:.3f}",
            transform=ax.transAxes, fontsize=5.5, ha="right", color=C_ALPHA)

    plt.subplots_adjust(wspace=0.35, hspace=0.40,
                        left=0.10, right=0.94, top=0.91, bottom=0.10)

    save_pub(fig, f"{save_dir}/training_losses")
    plt.close(fig)
    print(f"[OK] training_losses -> {save_dir}/training_losses.{{svg,pdf,png}}")


# ============================================================
# Figure 2: Q-value separation (key RL evidence)
# ============================================================
def plot_q_and_return(history, save_dir="visualization"):
    epochs = np.array(history["epoch"])
    q_data = np.array(history["q_mean"])
    q_rand = np.array(history["q_rand_mean"])
    gap = q_data - q_rand

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), facecolor=BG)
    fig.suptitle(
        "Q-Value Separation & Return",
        fontsize=10, fontweight="bold", x=0.0, y=0.97, ha="left"
    )

    # (a) Q-value gap
    ax = axes[0]
    ax.plot(epochs, q_data, color=C_QDATA, lw=1.0, label="$\\bar{Q}_{\\mathrm{data}}$")
    ax.plot(epochs, q_rand, color=C_QRAND, lw=0.8, ls="--",
            label="$\\bar{Q}_{\\mathrm{rand}}$")
    ax.fill_between(epochs, q_rand, q_data, alpha=0.06, color=C_QDATA)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Q-value")
    ax.set_title("a  Q-value Gap ($\\Delta Q$ = %.0f)" % gap[-1])
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax.annotate(f"$Q_{{\\mathrm{{data}}}}$ = {q_data[-1]:.1f}",
                xy=(500, q_data[-1]), xytext=(340, q_data[-1] + 5),
                fontsize=6, color=C_QDATA, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_QDATA, lw=0.5))
    ax.annotate(f"$Q_{{\\mathrm{{rand}}}}$ = {q_rand[-1]:.1f}",
                xy=(500, q_rand[-1]), xytext=(340, q_rand[-1] - 10),
                fontsize=6, color=C_QRAND,
                arrowprops=dict(arrowstyle="->", color=C_QRAND, lw=0.5))

    # (b) Immediate return + CQL alpha
    ax = axes[1]
    ax.plot(epochs, smooth(history["return_est"]), color=C_RETURN, lw=1.0,
            label="Return (norm.)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Return", color=C_RETURN)
    ax.tick_params(axis="y", labelcolor=C_RETURN)
    ax.set_title("b  Average Return & Lagrange alpha")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)
    ax2 = ax.twinx()
    ax2.plot(epochs, smooth(history["cql_alpha"]), color=C_ALPHA, lw=0.8,
             ls=":", label="alpha")
    ax2.set_ylabel("CQL alpha", color=C_ALPHA)
    ax2.tick_params(axis="y", labelcolor=C_ALPHA)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_ALPHA)
    ax2.spines["right"].set_linewidth(0.6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")

    plt.subplots_adjust(wspace=0.42, left=0.10, right=0.90, top=0.87, bottom=0.18)

    save_pub(fig, f"{save_dir}/q_value_return")
    plt.close(fig)
    print(f"[OK] q_value_return -> {save_dir}/q_value_return.{{svg,pdf,png}}")


# ============================================================
# Figure 3: Comprehensive training summary (hero figure)
# ============================================================
def plot_summary(history, save_dir="visualization"):
    epochs = np.array(history["epoch"])
    q_data = np.array(history["q_mean"])
    q_rand = np.array(history["q_rand_mean"])

    fig = plt.figure(figsize=(7.2, 5.8), facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05],
                          wspace=0.40, hspace=0.45,
                          left=0.09, right=0.96, top=0.91, bottom=0.10)

    # --- (0,0): Critic loss decomposition ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, smooth(history["bellman_loss"]), color=C_BELLMAN, lw=0.9,
            label="Bellman (TD)")
    ax.plot(epochs, smooth(history["cql_loss"]), color=C_CQL, lw=0.9,
            label="CQL penalty")
    ax.plot(epochs, smooth(history["critic_loss"]), color=C_CRITIC, lw=1.2,
            label="Critic total", alpha=0.7)
    ax.set_ylabel("Loss")
    ax.set_title("a  Critic Loss Decomposition")
    ax.legend(fontsize=5.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # --- (0,1): Q-value gap ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, q_data, color=C_QDATA, lw=1.0, label="$\\bar{Q}_{\\mathrm{data}}$")
    ax.plot(epochs, q_rand, color=C_QRAND, lw=0.8, ls="--",
            label="$\\bar{Q}_{\\mathrm{rand}}$")
    ax.fill_between(epochs, q_rand, q_data, alpha=0.06, color=C_QDATA)
    ax.set_ylabel("Q-value")
    ax.set_xlabel("Epoch")
    ax.set_title("b  Q-value Separation (CQL Conservatism)")
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # --- (0,2): Action fidelity inset ---
    ax = fig.add_subplot(gs[0, 2])
    import torch, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import STATE_DIM
    from models.actor import Actor
    from data.dataset import load_dataset
    s_np, a_np, *_ = load_dataset("PI_OfflineData.csv")
    actor = Actor(STATE_DIM)
    actor.load_state_dict(torch.load("output/actor_best.pth", map_location="cpu"))
    actor.eval()
    with torch.no_grad():
        a_pred = actor(torch.tensor(s_np[:2000])).numpy().flatten()
    a_t = a_np[:2000].flatten()
    mae = np.abs(a_pred - a_t).mean()
    ax.scatter(a_t, a_pred, s=0.6, alpha=0.25, color=C_QDATA,
               edgecolors="none", rasterized=True)
    ax.plot([a_t.min(), a_t.max()], [a_t.min(), a_t.max()], "-",
            color=C_ACTOR, lw=0.7, alpha=0.6)
    ax.set_xlabel("PI Demand (rad)")
    ax.set_ylabel("Actor Output (rad)")
    ax.set_title("c  Action Fidelity")
    ax.text(0.04, 0.96, f"MAE = {mae:.4f} rad\nn = {len(a_t):,}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.9))
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # --- (1,0): Actor / BC loss ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, smooth(history["actor_loss"]), color=C_ACTOR, lw=1.0,
            label="Actor loss")
    bc = np.array(history["bc_loss"])
    ax.plot(epochs[1:], smooth(bc[1:]), color=C_CRITIC, lw=0.7, ls="--",
            label="BC loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Actor Loss")
    ax.set_title("d  Actor Loss (-Q_norm + BC)")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # --- (1,1): CQL alpha auto-tuning ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, smooth(history["cql_alpha"]), color=C_ALPHA, lw=1.0)
    ax.axhline(y=history["cql_alpha"][-1], color=C_ALPHA, lw=0.4, ls=":",
               alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CQL alpha")
    ax.set_title("e  Lagrange alpha Auto-tuning")
    ax.text(0.98, 0.92, f"final = {history['cql_alpha'][-1]:.3f}",
            transform=ax.transAxes, fontsize=6, ha="right", color=C_ALPHA)
    ax.grid(True, alpha=0.25, lw=0.3, color=GRID)

    # --- (1,2): Metrics table ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    gap = q_data[-1] - q_rand[-1]
    metrics = [
        ("Q_data", f"{q_data[-1]:.2f}"),
        ("Q_rand", f"{q_rand[-1]:.2f}"),
        ("Delta Q gap", f"{gap:.1f}"),
        ("BC loss", f"{bc[-1]:.1e}"),
        ("CQL alpha", f"{history['cql_alpha'][-1]:.3f}"),
        ("Bellman loss", f"{history['bellman_loss'][-1]:.1f}"),
        ("MAE (rad)", f"{mae:.4f}"),
    ]
    ax.text(0.0, 0.95, "f  Final Metrics",
            fontsize=8, fontweight="bold", va="top", transform=ax.transAxes)
    for i, (k, v) in enumerate(metrics):
        y = 0.80 - i * 0.10
        ax.text(0.05, y, k, fontsize=7, va="center", transform=ax.transAxes,
                color="#444444")
        ax.text(0.55, y, v, fontsize=7, va="center", ha="right",
                fontweight="bold", transform=ax.transAxes, color="#111111")
    verdict = ("CQL working: Delta Q = %.0f, Actor fidelity MAE = %.4f rad"
               % (gap, mae))
    ax.text(0.0, 0.02, verdict, fontsize=6.5, va="bottom",
            transform=ax.transAxes, color=C_QDATA, fontweight="bold")

    save_pub(fig, f"{save_dir}/training_summary")
    plt.close(fig)
    print(f"[OK] training_summary -> {save_dir}/training_summary.{{svg,pdf,png}}")


# ============================================================
if __name__ == "__main__":
    history = load_history("output/training_history.json")
    print(f"Loaded: {len(history['epoch'])} epochs")

    plot_losses(history)
    plot_q_and_return(history)
    plot_summary(history)

    print("\nDone - SVG + PDF + PNG exported.\n")
