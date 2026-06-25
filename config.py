# config.py
# 全局配置文件：定义强化学习环境的状态/动作维度、CQL超参数、训练参数等
# v2: 修复Q值发散问题 — 奖励归一化、Lagrance自动调参、双重Critic、延迟更新

STATE_DIM = 5
ACTION_DIM = 1

MIN_PITCH = 0.0
MAX_PITCH = 1.57

BATCH_SIZE = 512

GAMMA = 0.99
TAU = 0.005

LR_ACTOR = 1e-4
LR_CRITIC = 3e-4          # Critic学习率适当提高

# ---- CQL 超参数 (Lagrance 自动调参版) ----
CQL_ALPHA_INIT = 0.1       # CQL惩罚初始权重（降低，由Lagrance自动调节）
CQL_ALPHA_LR = 1e-3        # Lagrance乘子学习率
CQL_TARGET_GAP = 2.0       # 目标: Q_data - Q_ood ≈ 2.0 (保守但不过度)
CQL_NUM_SAMPLES = 32       # 增加采样数，提升logsumexp估计精度

# ---- TD3+BC 混合参数 ----
LAMBDA_BC = 2.5            # TD3+BC风格的自适应行为克隆权重
REWARD_SCALE = 0.01        # 奖励缩放因子，将~[-200,0]映射到~[-2,0]

# ---- TD3 改进 ----
POLICY_FREQ = 2            # Actor每2次Critic更新才更新一次
POLICY_NOISE = 0.05        # 目标策略平滑噪声标准差 (rad)
POLICY_NOISE_CLIP = 0.1    # 噪声裁剪范围

# ---- 正则化 ----
VALUE_REG_WEIGHT = 2.0     # Q值约束正则化权重 (增大以锚定Q值在0附近)
GRAD_CLIP = 10.0           # 梯度裁剪阈值

EPOCHS = 500

DEVICE = "cuda"
