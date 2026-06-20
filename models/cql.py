# models/cql.py
# Critic（Q值）网络：输入状态+动作(5+1=6维)，输出Q值，用于CQL离线强化学习

import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(state_dim + 1, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, s, a):

        x = torch.cat([s, a], dim=-1)

        return self.net(x)