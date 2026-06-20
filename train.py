# train.py
# CQL (Conservative Q-Learning) 离线强化学习训练脚本
# 从PI_OfflineData.csv加载离线数据，训练Actor-Critic网络，保存最优模型
# 修复内容：
#   1. CQL logsumexp 改为在动作维度(dim=1)计算，每状态采样N个随机动作
#   2. Actor 加入 TD3+BC 风格的自适应行为克隆正则化
#   3. 加入梯度裁剪提升训练稳定性
#   4. 奖励缩放至 ~[-2, 0]

import json
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from config import *

from data.dataset import load_dataset

from models.actor import Actor
from models.cql import Critic


csv_file = "PI_OfflineData.csv"

(
    s,
    a,
    r,
    ns,
    d,
    norm
) = load_dataset(csv_file)

norm.save("output/normalization.json")

dataset = TensorDataset(
    torch.tensor(s),
    torch.tensor(a),
    torch.tensor(r),
    torch.tensor(ns),
    torch.tensor(d)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

actor = Actor(STATE_DIM).to(DEVICE)

critic = Critic(STATE_DIM).to(DEVICE)

target_critic = Critic(STATE_DIM).to(DEVICE)

target_critic.load_state_dict(
    critic.state_dict()
)

actor_opt = torch.optim.Adam(
    actor.parameters(),
    lr=LR_ACTOR
)

critic_opt = torch.optim.Adam(
    critic.parameters(),
    lr=LR_CRITIC
)

history = {
    "epoch": [],
    "actor_loss": [],
    "critic_loss": [],
    "bellman_loss": [],
    "cql_loss": [],
    "bc_loss": [],
    "q_mean": [],
    "return_est": [],
}

for epoch in range(EPOCHS):

    epoch_actor_loss = 0.0
    epoch_critic_loss = 0.0
    epoch_bellman_loss = 0.0
    epoch_cql_loss = 0.0
    epoch_bc_loss = 0.0
    epoch_q_mean = 0.0
    epoch_return = 0.0
    n_batches = 0

    for s, a, r, ns, d in loader:

        B = s.shape[0]

        s = s.to(DEVICE)
        a = a.to(DEVICE)

        r = r.to(DEVICE)

        ns = ns.to(DEVICE)

        d = d.to(DEVICE)

        # ---- 1. Critic update / Critic更新 ----
        with torch.no_grad():

            na = actor(ns)

            target_q = target_critic(
                ns,
                na
            )

            y = r + GAMMA * (1 - d) * target_q

        q = critic(s, a)

        bellman_loss = F.mse_loss(
            q,
            y
        )

        # CQL保守性惩罚：采样N个随机动作，logsumexp在动作维度上计算
        # CQL conservative penalty: sample N random actions, logsumexp over action dim
        rand_actions = torch.rand(
            B, CQL_NUM_SAMPLES, 1, device=DEVICE
        ) * MAX_PITCH

        s_expanded = s.unsqueeze(1).expand(
            -1, CQL_NUM_SAMPLES, -1
        )  # [B, N, 5]

        q_rand_all = critic(
            s_expanded.reshape(B * CQL_NUM_SAMPLES, -1),
            rand_actions.reshape(B * CQL_NUM_SAMPLES, 1)
        ).reshape(B, CQL_NUM_SAMPLES)  # [B, N]

        # logsumexp over action dimension (dim=1), then mean over batch
        cql_loss = (
            torch.logsumexp(q_rand_all, dim=1).mean()
            - q.mean()
        )

        critic_loss = (
            bellman_loss
            + CQL_ALPHA * cql_loss
        )

        critic_opt.zero_grad()

        critic_loss.backward()

        # 梯度裁剪 / Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            critic.parameters(), GRAD_CLIP
        )

        critic_opt.step()

        # ---- 2. Actor update / Actor更新 ----
        pred_a = actor(s)

        q_pred = critic(s, pred_a)

        # 行为克隆正则化：MSE(actor(s), a_dataset)，防止actor偏离数据分布
        # BC regularization: keep actor close to dataset actions
        bc_loss = F.mse_loss(pred_a, a)

        # TD3+BC风格的自适应lambda：λ = α / mean(|Q|)
        # Adaptive lambda normalizes Q scale for robust BC-Q trade-off
        with torch.no_grad():
            lam = LAMBDA_BC / (q.detach().abs().mean() + 1e-6)

        actor_loss = -lam * q_pred.mean() + bc_loss

        actor_opt.zero_grad()

        actor_loss.backward()

        # 梯度裁剪 / Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            actor.parameters(), GRAD_CLIP
        )

        actor_opt.step()

        # ---- 3. Target network soft update / 目标网络软更新 ----
        for tp, p in zip(
                target_critic.parameters(),
                critic.parameters()):

            tp.data.copy_(
                TAU * p.data
                + (1 - TAU) * tp.data
            )

        epoch_actor_loss += actor_loss.item()
        epoch_critic_loss += critic_loss.item()
        epoch_bellman_loss += bellman_loss.item()
        epoch_cql_loss += cql_loss.item()
        epoch_bc_loss += bc_loss.item()
        epoch_q_mean += q.mean().item()
        epoch_return += r.mean().item()
        n_batches += 1

    avg_actor = epoch_actor_loss / n_batches
    avg_critic = epoch_critic_loss / n_batches
    avg_bellman = epoch_bellman_loss / n_batches
    avg_cql = epoch_cql_loss / n_batches
    avg_bc = epoch_bc_loss / n_batches
    avg_q = epoch_q_mean / n_batches
    avg_return = epoch_return / n_batches

    history["epoch"].append(epoch)
    history["actor_loss"].append(avg_actor)
    history["critic_loss"].append(avg_critic)
    history["bellman_loss"].append(avg_bellman)
    history["cql_loss"].append(avg_cql)
    history["bc_loss"].append(avg_bc)
    history["q_mean"].append(avg_q)
    history["return_est"].append(avg_return)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(
            f"{epoch:4d} | actor: {avg_actor:10.4f} | critic: {avg_critic:10.4f} "
            f"| bellman: {avg_bellman:10.4f} | cql: {avg_cql:10.4f} "
            f"| bc: {avg_bc:8.4f} | q_mean: {avg_q:8.2f} | return: {avg_return:8.4f}"
        )

torch.save(
    actor.state_dict(),
    "output/actor_best.pth"
)

with open("output/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\nTraining history saved to output/training_history.json")