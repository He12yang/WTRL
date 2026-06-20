# config.py
# 全局配置文件：定义强化学习环境的状态/动作维度、CQL超参数、训练参数等

STATE_DIM = 5
ACTION_DIM = 1

MIN_PITCH = 0.0
MAX_PITCH = 1.57

BATCH_SIZE = 512

GAMMA = 0.99
TAU = 0.005

LR_ACTOR = 1e-4
LR_CRITIC = 1e-4          # 降低Critic学习率，与Actor一致，提升训练稳定性

CQL_ALPHA = 1.0           # 降低保守性惩罚权重 (原5.0过高，导致Q值极度悲观)
CQL_NUM_SAMPLES = 10      # 每个状态采样的随机动作数，用于logsumexp估计

LAMBDA_BC = 2.5           # TD3+BC风格的自适应行为克隆权重
REWARD_SCALE = 0.01       # 奖励缩放因子，将~[-200,0]映射到~[-2,0]

GRAD_CLIP = 10.0          # 梯度裁剪阈值

EPOCHS = 500

DEVICE = "cuda"
