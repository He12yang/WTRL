# models/actor.py
# Actor（策略）网络：输入5维状态，输出桨距角指令 [MIN_PITCH, MAX_PITCH]
# 使用sigmoid将输出映射到有效范围内

import torch
import torch.nn as nn

from config import MIN_PITCH
from config import MAX_PITCH


class Actor(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, state):

        x = self.net(state)

        x = torch.sigmoid(x)

        pitch = MIN_PITCH + \
                x * (MAX_PITCH - MIN_PITCH)

        return pitch