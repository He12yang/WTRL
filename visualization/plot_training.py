# visualization/plot_training.py
# 训练曲线可视化：绘制Actor/Critic损失、Q值估计、即时回报等训练过程指标
# Training curve visualization: plot Actor/Critic loss, Q-value estimation, immediate return over epochs

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------- 中英双语字体配置 / Bilingual font config ----------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_history(path):
    """加载训练历史 / Load training history"""
    with open(path, "r") as f:
        return json.load(f)


def smooth(y, window=10):
    """滑动窗口平滑 / Moving average smoothing"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_losses(history, save_dir="visualization"):
    """绘制损失曲线 / Plot loss curves"""
    epochs = np.array(history["epoch"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CQL Training Loss Curves\nCQL训练损失曲线",
        fontsize=15, fontweight="bold"
    )

    # ---- Actor Loss / Actor损失 ----
    ax = axes[0, 0]
    ax.plot(epochs, history["actor_loss"], alpha=0.3, color="tab:blue", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["actor_loss"], 10), color="tab:blue", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Actor Loss / Actor损失")
    ax.set_title("Actor Loss (max Q) / Actor损失（最大化Q值）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Critic Loss / Critic损失 ----
    ax = axes[0, 1]
    ax.plot(epochs, history["critic_loss"], alpha=0.3, color="tab:red", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["critic_loss"], 10), color="tab:red", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Critic Loss / Critic损失")
    ax.set_title("Critic Total Loss / Critic总损失")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Bellman Loss Component / Bellman误差分量 ----
    ax = axes[1, 0]
    ax.plot(epochs, history["bellman_loss"], alpha=0.3, color="tab:orange", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["bellman_loss"], 10), color="tab:orange", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Bellman Loss / Bellman损失")
    ax.set_title("Bellman Error (TD Error) / Bellman误差（TD误差）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- CQL Loss Component / CQL正则化分量 ----
    ax = axes[1, 1]
    ax.plot(epochs, history["cql_loss"], alpha=0.3, color="tab:purple", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["cql_loss"], 10), color="tab:purple", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("CQL Loss / CQL损失")
    ax.set_title("CQL Conservative Penalty / CQL保守性惩罚项")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/training_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Loss curves saved / 损失曲线已保存 -> {save_dir}/training_losses.png")


def plot_q_and_return(history, save_dir="visualization"):
    """绘制Q值和即时回报曲线 / Plot Q-value and immediate return curves"""
    epochs = np.array(history["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Q-Value & Return Estimation\nQ值估计与即时回报",
        fontsize=15, fontweight="bold"
    )

    # ---- Average Q-value / 平均Q值 ----
    ax = axes[0]
    ax.plot(epochs, history["q_mean"], alpha=0.3, color="tab:green", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["q_mean"], 10), color="tab:green", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Average Q-value / 平均Q值")
    ax.set_title("Average Q(s, a) on Dataset / 数据集上的平均Q值")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Q=0 baseline / 基线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Average immediate return / 平均即时回报 ----
    ax = axes[1]
    ax.plot(epochs, history["return_est"], alpha=0.3, color="tab:cyan", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["return_est"], 10), color="tab:cyan", linewidth=2, label="Smoothed / 平滑")
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Average Return / 平均即时回报")
    ax.set_title("Average Immediate Reward per Batch / 每批次平均即时奖励")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/q_value_return.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Q-value & return curves saved / Q值与回报曲线已保存 -> {save_dir}/q_value_return.png")


def plot_summary(history, save_dir="visualization"):
    """绘制综合摘要图 / Plot summary figure"""
    epochs = np.array(history["epoch"])
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "CQL Offline RL — Training Summary\nCQL离线强化学习训练总结",
        fontsize=16, fontweight="bold"
    )

    # ---- (0,0) Actor Loss ----
    ax = axes[0, 0]
    ax.plot(epochs, history["actor_loss"], alpha=0.3, color="tab:blue", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["actor_loss"], 10), color="tab:blue", linewidth=2)
    ax.set_ylabel("Actor Loss / Actor损失")
    ax.set_title("Actor Loss = E[-Q(s, π(s))] / Actor损失")
    ax.grid(True, alpha=0.3)

    # ---- (0,1) Critic Loss ----
    ax = axes[0, 1]
    ax.plot(epochs, history["critic_loss"], alpha=0.3, color="tab:red", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["critic_loss"], 10), color="tab:red", linewidth=2)
    ax.set_ylabel("Critic Loss / Critic损失")
    ax.set_title("Total Critic Loss = Bellman + CQL / Critic总损失")
    ax.grid(True, alpha=0.3)

    # ---- (1,0) Bellman Loss ----
    ax = axes[1, 0]
    ax.plot(epochs, history["bellman_loss"], alpha=0.3, color="tab:orange", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["bellman_loss"], 10), color="tab:orange", linewidth=2)
    ax.set_ylabel("Bellman Loss / Bellman损失")
    ax.set_title("Bellman TD Error / Bellman TD误差")
    ax.grid(True, alpha=0.3)

    # ---- (1,1) CQL Loss ----
    ax = axes[1, 1]
    ax.plot(epochs, history["cql_loss"], alpha=0.3, color="tab:purple", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["cql_loss"], 10), color="tab:purple", linewidth=2)
    ax.set_ylabel("CQL Penalty / CQL惩罚")
    ax.set_title("CQL Conservative Regularization / CQL保守性正则化")
    ax.grid(True, alpha=0.3)

    # ---- (2,0) Q Mean ----
    ax = axes[2, 0]
    ax.plot(epochs, history["q_mean"], alpha=0.3, color="tab:green", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["q_mean"], 10), color="tab:green", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Q-value Mean / Q值均值")
    ax.set_title("Average Q(s, a) on Dataset / 数据集平均Q值")
    ax.grid(True, alpha=0.3)

    # ---- (2,1) Immediate Return ----
    ax = axes[2, 1]
    ax.plot(epochs, history["return_est"], alpha=0.3, color="tab:cyan", linewidth=0.8)
    if len(epochs) >= 10:
        ax.plot(epochs[9:], smooth(history["return_est"], 10), color="tab:cyan", linewidth=2)
    ax.set_xlabel("Epoch / 轮次")
    ax.set_ylabel("Return / 回报")
    ax.set_title("Average Immediate Reward / 平均即时奖励")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{save_dir}/training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Summary figure saved / 训练总结图已保存 -> {save_dir}/training_summary.png")


if __name__ == "__main__":
    history = load_history("output/training_history.json")
    print(f"Loaded training history: {len(history['epoch'])} epochs / 已加载训练历史: {len(history['epoch'])} 轮")

    plot_losses(history)
    plot_q_and_return(history)
    plot_summary(history)
    print("\nAll plots generated / 所有图片已生成.")
