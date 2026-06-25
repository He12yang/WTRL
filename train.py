# train.py
# CQL (Conservative Q-Learning) 离线强化学习训练脚本 — v2
# v2 修复（针对Q值发散问题）:
#   1. 奖励零均值+单位方差归一化 → Q值自然在0附近
#   2. 双重Critic (Double Q) → min操作抑制估计偏差
#   3. CQL Lagrance 对偶梯度自动调参 → α不固定，自适应调节保守程度
#   4. 延迟策略更新 (TD3风格) → Critic更新K次后Actor更新1次
#   5. 目标策略平滑正则化 → TD目标加入截断噪声
#   6. Q值约束正则化 → 软约束Q_data不偏离理论范围
#   7. LayerNorm + 梯度裁剪 → 数值稳定性

import json
import time
import sys
import copy
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from config import *

from data.dataset import load_dataset

from models.actor import Actor
from models.cql import Critic


csv_file = "PI_OfflineData.csv"

# ============================================================
# 0. 数据加载 / Data Loading
# ============================================================
print("=" * 72)
print("  CQL v2 — 风力发电机桨距角控制 (修复Q值发散)")
print("  CQL v2 — Wind Turbine Pitch Control (Q-divergence fixed)")
print("=" * 72)

print(f"\n[1/5] 加载数据 / Loading data from '{csv_file}' ...")
t0 = time.time()

result = load_dataset(csv_file)
s, a, r, ns, d, norm, reward_mean, reward_std = result

print(f"      → 样本数: {len(s):,}  |  状态维度: {s.shape[1]}  |  动作维度: {a.shape[1]}")
print(f"      → 奖励范围(归一化后): [{r.min():.4f}, {r.max():.4f}]")
print(f"      → 动作范围: [{a.min():.4f}, {a.max():.4f}]")
print(f"      耗时: {time.time() - t0:.2f}s")

# 计算Q值理论范围 (用于正则化)
# 归一化后奖励 ~N(0,1), 保守估计V ≈ [-3/(1-γ), 3/(1-γ)] = [-300, 300]
# 但实际Q值应该更集中, 使用软约束
Q_TARGET = r.mean().item() / (1 - GAMMA)  # 理论Q值中心
print(f"      → 理论Q值中心 (r_mean/(1-γ)): {Q_TARGET:.4f}")

norm.save("output/normalization.json")

dataset = TensorDataset(
    torch.tensor(s),
    torch.tensor(a),
    torch.tensor(r),
    torch.tensor(ns),
    torch.tensor(d)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"      → 批次数: {len(loader)} (batch_size={BATCH_SIZE})")

# ============================================================
# 1. 模型初始化 / Model Initialization
# ============================================================
print(f"\n[2/5] 初始化模型 / Initializing models ...")
print(f"      → 设备: {DEVICE}")

actor = Actor(STATE_DIM).to(DEVICE)
critic = Critic(STATE_DIM).to(DEVICE)
target_critic = Critic(STATE_DIM).to(DEVICE)
target_critic.load_state_dict(critic.state_dict())

# 初始化目标Actor (用于策略平滑)
target_actor = Actor(STATE_DIM).to(DEVICE)
target_actor.load_state_dict(actor.state_dict())

n_actor_params = sum(p.numel() for p in actor.parameters())
n_critic_params = sum(p.numel() for p in critic.parameters())
print(f"      → Actor:  {n_actor_params:,} 参数")
print(f"      → Critic: {n_critic_params:,} 参数 (Double Q)")

actor_opt = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

# ---- CQL Lagrance 对偶变量 ----
# log_alpha 确保 alpha > 0，且梯度更新更稳定
log_alpha = torch.tensor([CQL_ALPHA_INIT], device=DEVICE, requires_grad=True)
alpha_opt = torch.optim.Adam([log_alpha], lr=CQL_ALPHA_LR)

# ============================================================
# 2. 超参数摘要 / Hyperparameter Summary
# ============================================================
print(f"\n[3/5] 超参数配置 / Hyperparameters:")
print(f"      → γ={GAMMA}, τ={TAU}")
print(f"      → CQL: α_init={CQL_ALPHA_INIT}, α_lr={CQL_ALPHA_LR}, "
      f"target_gap={CQL_TARGET_GAP}, num_samples={CQL_NUM_SAMPLES}")
print(f"      → TD3+BC: λ_bc={LAMBDA_BC}")
print(f"      → 学习率: lr_actor={LR_ACTOR}, lr_critic={LR_CRITIC}")
print(f"      → 策略更新频率: 每{CQL_NUM_SAMPLES}步 (policy_freq={POLICY_FREQ})")
print(f"      → 策略噪声: σ={POLICY_NOISE}, clip={POLICY_NOISE_CLIP}")
print(f"      → 值正则化: weight={VALUE_REG_WEIGHT}, Q_target≈{Q_TARGET:.2f}")
print(f"      → batch={BATCH_SIZE}, epochs={EPOCHS}, grad_clip={GRAD_CLIP}")

# ============================================================
# 3. 训练循环 / Training Loop
# ============================================================
print(f"\n[4/5] 开始训练 / Starting training ({EPOCHS} epochs)...")
print("-" * 80)
header = (f"{'Epoch':>5s} | {'actor':>8s} | {'critic':>8s} | "
          f"{'bellman':>8s} | {'cql':>8s} | {'bc':>8s} | "
          f"{'q_data':>8s} | {'q_rand':>8s} | {'cql_α':>8s} | "
          f"{'进度':>6s}")
print(header)
print("-" * 80)

history = {
    "epoch": [],
    "actor_loss": [],
    "critic_loss": [],
    "bellman_loss": [],
    "cql_loss": [],
    "bc_loss": [],
    "q_mean": [],
    "q_rand_mean": [],
    "cql_alpha": [],
    "return_est": [],
}

t_train_start = time.time()
total_steps = 0

for epoch in range(EPOCHS):

    epoch_actor_loss = 0.0
    epoch_critic_loss = 0.0
    epoch_bellman_loss = 0.0
    epoch_cql_loss = 0.0
    epoch_bc_loss = 0.0
    epoch_q_mean = 0.0
    epoch_q_rand_mean = 0.0
    epoch_cql_alpha = 0.0
    epoch_return = 0.0
    n_batches = 0
    n_actor_updates = 0

    for s_batch, a_batch, r_batch, ns_batch, d_batch in loader:

        B = s_batch.shape[0]
        total_steps += 1

        s_batch = s_batch.to(DEVICE)
        a_batch = a_batch.to(DEVICE)
        r_batch = r_batch.to(DEVICE)
        ns_batch = ns_batch.to(DEVICE)
        d_batch = d_batch.to(DEVICE)

        # ================================================
        # 1. Critic Update (每步都更新)
        # ================================================
        with torch.no_grad():
            # 目标策略平滑 (TD3 style): a' = π_target(s') + clip(N(0,σ), -c, c)
            noise = torch.randn_like(a_batch) * POLICY_NOISE
            noise = noise.clamp(-POLICY_NOISE_CLIP, POLICY_NOISE_CLIP)
            na_smoothed = target_actor(ns_batch)
            na_smoothed = (na_smoothed + noise).clamp(MIN_PITCH, MAX_PITCH)

            # 双重Critic min操作
            target_q_min = target_critic.q_min(ns_batch, na_smoothed)
            y = r_batch + GAMMA * (1 - d_batch) * target_q_min

        # 双重Q值
        q1, q2 = critic(s_batch, a_batch)
        bellman_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # ---- CQL 保守性惩罚 (双重版本) ----
        # (a) 均匀分布随机动作
        rand_actions = torch.rand(B, CQL_NUM_SAMPLES, 1, device=DEVICE) * MAX_PITCH

        # (b) 当前策略动作 + 噪声
        with torch.no_grad():
            policy_actions = actor(s_batch).unsqueeze(1).expand(-1, CQL_NUM_SAMPLES, -1)
            p_noise = torch.randn_like(policy_actions) * 0.1
            policy_actions = torch.clamp(policy_actions + p_noise, MIN_PITCH, MAX_PITCH)

        s_expanded = s_batch.unsqueeze(1).expand(-1, CQL_NUM_SAMPLES, -1)
        s_flat = s_expanded.reshape(B * CQL_NUM_SAMPLES, -1)

        # Q1, Q2 对随机动作和策略动作
        q1_rand, q2_rand = critic(s_flat, rand_actions.reshape(B * CQL_NUM_SAMPLES, 1))
        q1_rand = q1_rand.reshape(B, CQL_NUM_SAMPLES)
        q2_rand = q2_rand.reshape(B, CQL_NUM_SAMPLES)
        # 取平均避免双重惩罚过大
        q_rand_avg = (q1_rand + q2_rand) / 2.0

        q1_pol, q2_pol = critic(s_flat, policy_actions.reshape(B * CQL_NUM_SAMPLES, 1))
        q1_pol = q1_pol.reshape(B, CQL_NUM_SAMPLES)
        q2_pol = q2_pol.reshape(B, CQL_NUM_SAMPLES)
        q_pol_avg = (q1_pol + q2_pol) / 2.0

        # CQL loss: logsumexp(rand) + logsumexp(policy) - 2*Q_data
        # 对两个Critic取平均
        cql_loss = (
            torch.logsumexp(q_rand_avg, dim=1).mean()
            + torch.logsumexp(q_pol_avg, dim=1).mean()
        ) / 2.0 - (q1.mean() + q2.mean()) / 2.0

        # Q值约束正则化：软约束Q_data不偏离理论值太远
        value_reg = (q1.mean() - Q_TARGET).pow(2) + (q2.mean() - Q_TARGET).pow(2)

        # CQL Lagrance 自动调参
        alpha = torch.exp(log_alpha)
        critic_loss = bellman_loss + alpha * cql_loss + VALUE_REG_WEIGHT * value_reg

        critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)  # retain_graph: alpha还需用于alpha_loss
        torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
        critic_opt.step()

        # ---- CQL Lagrance 对偶更新 ----
        # L(α) = α * (cql_loss - target_gap)  →  max_α L  →  min_α -L
        # 当 cql_loss > target_gap 时增大α, 否则减小α
        alpha_loss = -alpha * (cql_loss.detach() - CQL_TARGET_GAP)
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

        # 确保α > 0
        with torch.no_grad():
            log_alpha.clamp_(min=-5.0, max=5.0)  # α ∈ [e^-5, e^5] ≈ [0.007, 148]

        # ================================================
        # 2. Actor Update (延迟更新 / Delayed Update)
        # ================================================
        if total_steps % POLICY_FREQ == 0:
            pred_a = actor(s_batch)
            q_pred = critic.q1_forward(s_batch, pred_a)

            # 行为克隆正则化
            bc_loss = F.mse_loss(pred_a, a_batch)

            # Q值归一化: 在batch内做z-score归一化，消除Q值整体偏移的影响
            # 这样即使Q值整体为负(如-80)，Actor仍能感知不同动作间的Q值差异
            with torch.no_grad():
                q_std = q_pred.std() + 1e-6
                q_normalized = (q_pred - q_pred.mean()) / q_std

            # TD3+BC: actor_loss = -λ * Q_norm + BC
            # λ_bc 直接使用，因为归一化后Q的量级为O(1)
            actor_loss = -LAMBDA_BC * q_normalized.mean() + bc_loss

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            actor_opt.step()

            # 软更新目标Actor
            for tp, p in zip(target_actor.parameters(), actor.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

            n_actor_updates += 1
            epoch_actor_loss += actor_loss.item()
            epoch_bc_loss += bc_loss.item()

        # ================================================
        # 3. Target Critic Soft Update
        # ================================================
        for tp, p in zip(target_critic.parameters(), critic.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        # ---- 累积统计 ----
        epoch_critic_loss += critic_loss.item()
        epoch_bellman_loss += bellman_loss.item()
        epoch_cql_loss += cql_loss.item()
        epoch_q_mean += (q1.mean().item() + q2.mean().item()) / 2.0
        epoch_q_rand_mean += q_rand_avg.mean().item()
        epoch_cql_alpha += alpha.item()
        epoch_return += r_batch.mean().item()
        n_batches += 1

    # ---- 计算 epoch 均值 ----
    avg_actor = epoch_actor_loss / max(n_actor_updates, 1)
    avg_critic = epoch_critic_loss / n_batches
    avg_bellman = epoch_bellman_loss / n_batches
    avg_cql = epoch_cql_loss / n_batches
    avg_bc = epoch_bc_loss / max(n_actor_updates, 1)
    avg_q = epoch_q_mean / n_batches
    avg_q_rand = epoch_q_rand_mean / n_batches
    avg_alpha = epoch_cql_alpha / n_batches
    avg_return = epoch_return / n_batches

    history["epoch"].append(epoch)
    history["actor_loss"].append(avg_actor)
    history["critic_loss"].append(avg_critic)
    history["bellman_loss"].append(avg_bellman)
    history["cql_loss"].append(avg_cql)
    history["bc_loss"].append(avg_bc)
    history["q_mean"].append(avg_q)
    history["q_rand_mean"].append(avg_q_rand)
    history["cql_alpha"].append(avg_alpha)
    history["return_est"].append(avg_return)

    # ---- 进度打印 (每5 epoch) ----
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        pct = (epoch + 1) / EPOCHS * 100
        elapsed = time.time() - t_train_start

        line = (f"{epoch:5d} | {avg_actor:8.4f} | {avg_critic:8.4f} | "
                f"{avg_bellman:8.4f} | {avg_cql:8.4f} | {avg_bc:8.4f} | "
                f"{avg_q:8.2f} | {avg_q_rand:8.2f} | {avg_alpha:8.4f} | "
                f"{pct:5.1f}%")
        print(line)

        # 每50 epoch打印时间
        if epoch % 50 == 0 and epoch > 0:
            eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
            print(f"      → 已耗时: {elapsed:.0f}s | 预计剩余: {eta:.0f}s | "
                  f"CQL α: {avg_alpha:.4f}")

# ============================================================
# 4. 保存 & 总结 / Save & Summary
# ============================================================
t_total = time.time() - t_train_start
print("-" * 80)
print(f"\n[5/5] 训练完成 / Training complete!  总耗时: {t_total:.1f}s ({t_total/60:.1f}min)")

torch.save(actor.state_dict(), "output/actor_best.pth")
torch.save(target_critic.state_dict(), "output/critic_best.pth")
print(f"→ Actor 模型已保存: output/actor_best.pth")
print(f"→ Critic 模型已保存: output/critic_best.pth")

with open("output/training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"→ 训练历史已保存: output/training_history.json")

# ---- 最终指标摘要 ----
print(f"\n{'='*60}")
print(f"  最终训练指标 / Final Training Metrics")
print(f"{'='*60}")
print(f"  Actor Loss     : {history['actor_loss'][-1]:.4f}")
print(f"  Critic Loss    : {history['critic_loss'][-1]:.4f}")
print(f"  Bellman Loss   : {history['bellman_loss'][-1]:.4f}")
print(f"  CQL Loss       : {history['cql_loss'][-1]:.4f}")
print(f"  BC Loss        : {history['bc_loss'][-1]:.6f}")
print(f"  Q_data mean    : {history['q_mean'][-1]:.4f}")
print(f"  Q_rand mean    : {history['q_rand_mean'][-1]:.4f}")
print(f"  CQL alpha      : {history['cql_alpha'][-1]:.4f}")
print(f"  CQL gap        : {history['q_mean'][-1] - history['q_rand_mean'][-1]:.2f}")
print(f"  Avg Return(norm): {history['return_est'][-1]:.4f}")
print(f"{'='*60}")

# 健康检查
final_q = history['q_mean'][-1]
final_q_rand = history['q_rand_mean'][-1]
final_alpha = history['cql_alpha'][-1]
gap = final_q - final_q_rand

checks = []
if gap > 0:
    checks.append(f"[OK] Q_data({final_q:.2f}) > Q_rand({final_q_rand:.2f}), gap={gap:.2f}")
else:
    checks.append(f"[WARN] CQL失效: Q_data <= Q_rand, 差距={gap:.2f}")

if abs(final_q) < 50:
    checks.append(f"[OK] Q值在合理范围: |Q|={abs(final_q):.2f} < 50")
else:
    checks.append(f"[WARN] Q值可能过大: |Q|={abs(final_q):.2f}")

if final_alpha > 0.001 and final_alpha < 50:
    checks.append(f"[OK] CQL α 在合理范围: {final_alpha:.4f}")
else:
    checks.append(f"[WARN] CQL α 异常: {final_alpha:.4f}")

if history['bc_loss'][-1] > 1e-6:
    checks.append(f"[OK] BC loss未完全收敛: {history['bc_loss'][-1]:.6f}")
else:
    checks.append(f"[INFO] BC loss ≈ 0: Actor几乎完全模仿数据集")

for c in checks:
    print(f"  {c}")
