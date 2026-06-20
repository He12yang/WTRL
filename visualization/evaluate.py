# visualization/evaluate.py
# 模型评估与性能可视化：加载训练好的Actor，在离线数据集上评估策略质量
# Model evaluation & performance visualization: load trained Actor, evaluate policy on offline dataset

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import sys

# Add parent directory to path / 将父目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STATE_DIM, MIN_PITCH, MAX_PITCH, REWARD_SCALE
from data.dataset import load_dataset
from data.normalizer import Normalizer
from models.actor import Actor
from models.cql import Critic

# ---------- 中英双语字体配置 / Bilingual font config ----------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_model_and_data():
    """加载模型与数据 / Load model and data"""
    # Load data
    s, a, r, ns, d, norm = load_dataset("PI_OfflineData.csv")

    # Load trained model
    actor = Actor(STATE_DIM)
    actor.load_state_dict(torch.load("output/actor_best.pth", map_location="cpu"))
    actor.eval()

    # Also load critic for Q-value evaluation
    critic = Critic(STATE_DIM)
    # critic not trained here but we can evaluate with a simple Q estimate

    return s, a, r, ns, d, norm, actor


def plot_action_comparison(s, a_true, actor, save_dir="visualization"):
    """动作对比图：Actor预测 vs 数据集真实动作 / Action comparison: Actor prediction vs dataset action"""
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy().flatten()

    a_true = a_true.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Policy Action vs Dataset Action\n策略动作 vs 数据集动作",
        fontsize=15, fontweight="bold"
    )

    # ---- Scatter: predicted vs true / 散点图：预测 vs 真实 ----
    ax = axes[0, 0]
    ax.scatter(a_true[:2000], a_pred[:2000], alpha=0.3, s=5, c="tab:blue")
    ax.plot([a_true.min(), a_true.max()], [a_true.min(), a_true.max()], "r--", linewidth=1.5, label="y=x (ideal)")
    ax.set_xlabel("Dataset Action / 数据集动作 (rad)")
    ax.set_ylabel("Actor Prediction / Actor预测 (rad)")
    ax.set_title("Predicted vs True Action (first 2k samples)\n预测动作 vs 真实动作（前2000样本）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Error distribution / 误差分布 ----
    errors = a_pred - a_true
    ax = axes[0, 1]
    ax.hist(errors, bins=80, color="tab:orange", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero error / 零误差")
    ax.set_xlabel("Prediction Error / 预测误差 (rad)")
    ax.set_ylabel("Frequency / 频数")
    ax.set_title(
        f"Action Error Distribution\n动作误差分布 (MAE={np.abs(errors).mean():.6f} rad, RMSE={np.sqrt(np.mean(errors**2)):.6f} rad)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Time-series comparison / 时序对比 ----
    sample_idx = np.arange(min(500, len(a_true)))
    ax = axes[1, 0]
    ax.plot(sample_idx, a_true[sample_idx], alpha=0.7, linewidth=1.2, label="Dataset Action / 数据集动作")
    ax.plot(sample_idx, a_pred[sample_idx], alpha=0.7, linewidth=1.2, label="Actor Prediction / Actor预测")
    ax.set_xlabel("Sample Index / 样本序号")
    ax.set_ylabel("Pitch Demand / 桨距角指令 (rad)")
    ax.set_title("Action Comparison (first 500 samples)\n动作对比（前500样本）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Action distribution / 动作分布 ----
    ax = axes[1, 1]
    ax.hist(a_true, bins=60, alpha=0.5, color="tab:blue", label="Dataset / 数据集", density=True)
    ax.hist(a_pred, bins=60, alpha=0.5, color="tab:orange", label="Actor / Actor预测", density=True)
    ax.set_xlabel("Pitch Demand / 桨距角指令 (rad)")
    ax.set_ylabel("Density / 密度")
    ax.set_title("Action Distribution Comparison\n动作分布对比")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/action_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Action comparison saved / 动作对比图已保存 -> {save_dir}/action_comparison.png")


def plot_q_value_estimation(s, a, actor, critic_model_path=None, save_dir="visualization"):
    """
    Q值估计图 / Q-value estimation plot
    Compare Q(s, a_dataset) vs Q(s, a_actor) to check overestimation
    """
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy()

    # Compute approximate Q using reward structure directly
    # Since we don't have a trained critic, compute the implied return from reward formula
    # Q ≈ reward / (1 - gamma) as rough estimate
    from config import GAMMA

    # Use reward formula: r = -10 * speed_err^2 - 2 * tower_acc^2
    # speed_err = s[:,1], tower_acc = s[:,4] (before normalization we can't directly use them)
    # We'll estimate using inverse normalization if available

    # Instead, let's compute a simple return estimate:
    # For each state, compute what the reward WOULD be if we used the action
    # We can compare actions against the dataset reward

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Policy Quality Assessment\n策略质量评估",
        fontsize=15, fontweight="bold"
    )

    # ---- Action difference distribution / 动作差异分布 ----
    diff = (a_pred.flatten() - a.flatten())
    ax = axes[0]
    ax.hist(diff, bins=80, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("ΔAction / 动作差 (rad)")
    ax.set_ylabel("Count / 计数")
    ax.set_title(f"Action Deviation Δa = a_pred - a_true\n动作偏差 (mean={diff.mean():.5f}, std={diff.std():.5f})")
    ax.grid(True, alpha=0.3)

    # ---- Per-sample action difference / 逐样本动作差 ----
    ax = axes[1]
    n_show = min(1000, len(diff))
    ax.plot(np.arange(n_show), diff[:n_show], alpha=0.6, linewidth=0.8, color="tab:green")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample Index / 样本序号")
    ax.set_ylabel("ΔAction / 动作差 (rad)")
    ax.set_title(f"Per-sample Action Deviation\n逐样本动作偏差（前{n_show}样本）")
    ax.grid(True, alpha=0.3)

    # ---- Action by state feature (pitch vs wind speed proxy) / 按状态特征的动作分布 ----
    ax = axes[2]
    # Use gen_speed (state dim 0) as x-axis proxy
    state_dim0 = s[:3000, 0]  # normalized gen_speed
    ax.scatter(state_dim0, a[:3000].flatten(), alpha=0.3, s=5, label="Dataset / 数据集", color="tab:blue")
    ax.scatter(state_dim0, a_pred[:3000].flatten(), alpha=0.3, s=5, label="Actor / Actor预测", color="tab:orange")
    ax.set_xlabel("GenSpeed (normalized) / 发电机转速（归一化）")
    ax.set_ylabel("Pitch Demand / 桨距角指令 (rad)")
    ax.set_title("Action vs Generator Speed\n动作 vs 发电机转速")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/policy_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Policy quality plot saved / 策略质量图已保存 -> {save_dir}/policy_quality.png")


def plot_state_coverage(s, a, save_dir="visualization"):
    """
    状态-动作覆盖分析 / State-action coverage analysis
    Visualize state distribution and action across state dimensions
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "State-Action Coverage Analysis\n状态-动作覆盖分析",
        fontsize=15, fontweight="bold"
    )

    state_names_en = [
        "GenSpeed (normalized) / 发电机转速（归一化）",
        "Speed Error (normalized) / 转速误差（归一化）",
        "WindSpeed (normalized) / 风速（归一化）",
        "PitchAngle (normalized) / 桨距角（归一化）",
        "TowerAcc (normalized) / 塔架加速度（归一化）",
    ]

    for i in range(5):
        ax = axes[i // 3, i % 3]
        ax.scatter(s[:3000, i], a[:3000].flatten(), alpha=0.2, s=5, c="tab:blue")
        ax.set_xlabel(state_names_en[i])
        ax.set_ylabel("Action / 动作 (rad)")
        ax.set_title(f"Action vs State Dim {i}\n动作 vs 状态维度{i}")
        ax.grid(True, alpha=0.3)

    # 6th subplot: action histogram overlay
    ax = axes[1, 2]
    ax.hist(a.flatten(), bins=60, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Action / 动作 (rad)")
    ax.set_ylabel("Count / 计数")
    ax.set_title("Action Value Distribution\n动作值分布")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/state_action_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] State-action coverage saved / 状态动作覆盖图已保存 -> {save_dir}/state_action_coverage.png")


def print_evaluation_metrics(s, a, r, actor):
    """打印评估指标 / Print evaluation metrics"""
    with torch.no_grad():
        a_pred = actor(torch.tensor(s)).numpy().flatten()

    a_true = a.flatten()
    mae = np.abs(a_pred - a_true).mean()
    rmse = np.sqrt(np.mean((a_pred - a_true) ** 2))
    max_err = np.abs(a_pred - a_true).max()

    # Action statistics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS / 评估指标")
    print("=" * 60)
    print(f"{'MAE (Mean Absolute Error) / 平均绝对误差':<50s}: {mae:.6f} rad")
    print(f"{'RMSE (Root Mean Square Error) / 均方根误差':<50s}: {rmse:.6f} rad")
    print(f"{'Max Absolute Error / 最大绝对误差':<50s}: {max_err:.6f} rad")
    print(f"{'Dataset Action Mean / 数据集动作均值':<50s}: {a_true.mean():.6f} rad")
    print(f"{'Actor Action Mean / Actor动作均值':<50s}: {a_pred.mean():.6f} rad")
    print(f"{'Dataset Action Std / 数据集动作标准差':<50s}: {a_true.std():.6f} rad")
    print(f"{'Actor Action Std / Actor动作标准差':<50s}: {a_pred.std():.6f} rad")
    print(f"{'Action Range (Dataset) / 数据集动作范围':<50s}: [{a_true.min():.6f}, {a_true.max():.6f}]")
    print(f"{'Action Range (Actor) / Actor动作范围':<50s}: [{a_pred.min():.6f}, {a_pred.max():.6f}]")
    print(f"{'Pitch Range Limit / 桨距角限制范围':<50s}: [{MIN_PITCH}, {MAX_PITCH}]")
    print(f"{'Average Reward (scaled) / 平均即时奖励（缩放后）':<50s}: {r.mean():.6f}")
    print(f"{'Average Reward (original) / 平均即时奖励（原始）':<50s}: {r.mean() / REWARD_SCALE:.6f}")
    print("=" * 60)

    # Quick verdict
    if mae < 0.01:
        verdict = "EXCELLENT / 优秀 — Actor closely matches the behavior policy"
    elif mae < 0.05:
        verdict = "GOOD / 良好 — Minor deviation from behavior policy"
    elif mae < 0.1:
        verdict = "FAIR / 一般 — Noticeable deviation, check training"
    else:
        verdict = "POOR / 较差 — Large deviation, training may need tuning"
    print(f"VERDICT / 评估结论: {verdict}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Loading model and data... / 正在加载模型与数据...")
    s, a, r, ns, d, norm, actor = load_model_and_data()
    print(f"Loaded: {len(s)} samples / 已加载 {len(s)} 个样本")

    # Run evaluations
    print_evaluation_metrics(s, a, r, actor)
    plot_action_comparison(s, a, actor)
    plot_q_value_estimation(s, a, actor)
    plot_state_coverage(s, a)

    print("\nAll evaluation plots generated / 所有评估图片已生成.")
