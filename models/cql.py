# models/cql.py
# Critic（Q值）网络：输入状态+动作(5+1=6维)，输出Q值
# v2: Double Q-network (两个独立Critic)，取min抑制过估计

import torch
import torch.nn as nn


def build_mlp(state_dim):
    """构建单个Critic的MLP主干"""
    return nn.Sequential(
        nn.Linear(state_dim + 1, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


class Critic(nn.Module):
    """双Q Critic — 两个独立的Q网络，取min用于TD目标"""

    def __init__(self, state_dim):
        super().__init__()
        self.q1 = build_mlp(state_dim)
        self.q2 = build_mlp(state_dim)

    def forward(self, s, a):
        """返回两个Q值 [B, 1], [B, 1]"""
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, s, a):
        """返回min(Q1, Q2)，用于TD目标计算"""
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)

    def q1_forward(self, s, a):
        """仅使用Q1，用于Actor更新"""
        x = torch.cat([s, a], dim=-1)
        return self.q1(x)
