# env/reward.py
# 奖励函数定义：基于转速误差和塔架加速度计算即时奖励
# 注意：当前build_reward函数位于data/dataset.py中，建议迁移至此文件以保持模块职责清晰

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