# CQL 离线强化学习风力发电机桨距角控制 — 技术说明报告

> **项目名称**: offline_pitch_rl — 基于保守Q学习的风力发电机桨距角离线强化学习控制  
> **算法**: CQL (Conservative Q-Learning) + TD3+BC 自适应行为克隆正则化  
> **框架**: PyTorch  
> **日期**: 2026-06-20

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [马尔可夫决策过程建模](#3-马尔可夫决策过程建模)
4. [数据流水线](#4-数据流水线)
5. [神经网络模型](#5-神经网络模型)
6. [CQL 训练算法](#6-cql-训练算法)
7. [关键设计决策与超参数](#7-关键设计决策与超参数)
8. [模型导出与部署](#8-模型导出与部署)
9. [可视化与评估体系](#9-可视化与评估体系)
10. [模块清单](#10-模块清单)

---

## 1. 项目概述

### 1.1 问题背景

传统风力发电机桨距角控制依赖 PI 控制器。本项目的目标是**利用离线历史数据训练一个强化学习策略网络（Actor）**，替代 PI 控制器输出桨距角指令。核心挑战在于：

- **离线数据约束**：无法在线交互，仅能使用 PI 控制器产生的历史数据 (`PI_OfflineData.csv`)
- **分布外（OOD）动作过估计**：标准 Q-learning 会对数据分布外的动作产生过高的 Q 值估计
- **策略安全约束**：Actor 输出必须在物理可行范围 `[0, 1.57]` rad 内，且不能偏离数据分布太远

### 1.2 解决方案

采用 **CQL (Conservative Q-Learning)** 算法，核心思路：

1. **保守Q值估计**：对 Critic 施加 log-sum-exp 正则化，压低数据分布外随机动作的 Q 值
2. **行为克隆正则化**：Actor 损失中加入 BC 项 `MSE(π(s), a_dataset)`，防止策略偏离数据分布
3. **TD3+BC 自适应权重**：`λ = α / mean(|Q|)` 动态平衡 Q 最大化与行为克隆
4. **Z-score 归一化**：所有状态特征标准化，提升训练数值稳定性
5. **奖励缩放**：将原始 `~[-200, 0]` 奖励映射到 `~[-2, 0]`

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     offline_pitch_rl 系统架构                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ config.py │◄───│  配置文件     │    │  全局超参数 & 环境定义  │   │
│  └──────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    data/ 数据层                            │    │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐  │    │
│  │  │ dataset.py   │  │ normalizer.py                    │  │    │
│  │  │ · CSV加载     │  │ · Z-score fit/transform/inverse  │  │    │
│  │  │ · 状态构造     │  │ · save/load 持久化               │  │    │
│  │  │ · 奖励计算     │  │                                  │  │    │
│  │  │ · 五元组生成   │  │                                  │  │    │
│  │  └──────────────┘  └──────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   models/ 模型层                          │    │
│  │  ┌────────────────┐  ┌────────────────────────────────┐  │    │
│  │  │ actor.py       │  │ cql.py (Critic)                │  │    │
│  │  │ · 5→256→256→   │  │ · (5+1)→256→256→128→1         │  │    │
│  │  │   128→1        │  │ · Q(s, a) 值估计               │  │    │
│  │  │ · sigmoid 约束  │  │                                │  │    │
│  │  └────────────────┘  └────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    train.py 训练引擎                      │    │
│  │  · CQL Critic更新（Bellman + logsumexp保守惩罚）           │    │
│  │  · Actor更新（max Q + BC正则化 + 自适应λ）                 │    │
│  │  · 目标网络软更新（Polyak averaging）                      │    │
│  │  · 梯度裁剪 & 训练历史记录                                 │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    输出 & 部署层                           │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │    │
│  │  │ actor_best.pth│  │ actor.onnx    │  │ training_    │ │    │
│  │  │ (PyTorch模型)  │  │ (ONNX部署)    │  │ history.json │ │    │
│  │  └───────────────┘  └───────────────┘  └──────────────┘ │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                 visualization/ 可视化层                    │    │
│  │  plot_training.py (训练曲线)  │  evaluate.py (评估分析)    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 数据流图

```
PI_OfflineData.csv
       │
       ▼
  dataset.py ─────────────────────────────
  │ 读取CSV                               │
  │ 提取5维状态: [GenSpeed, SpeedError,   │
  │   WindSpeed, PitchAngle, TowerAcc]     │
  │ 构造奖励: -10·err² -2·tower_acc²     │
  │ 缩放奖励: ×0.01                       │
  │ 构造next_state: shift(-1)             │
  │ 构造done: 最后一行为1                  │
  │                                         │
  │ normalizer.fit(state)                  │
  │ state = normalizer.transform(state)    │
  │ next_state = normalizer.transform(ns)  │
  │                                         │
  │ → TensorDataset(s, a, r, ns, d)        │
  └────────────────────────────────────────┘
       │
       ▼
  DataLoader (batch_size=512, shuffle=True)
       │
       ▼
  train.py ── 500 epochs ──► output/
```

---

## 3. 马尔可夫决策过程建模

### 3.1 状态空间 (State Space)

| 维度 | 变量名 | 含义 | 单位 | 归一化 |
|------|--------|------|------|--------|
| 0 | GenSpeed | 发电机转速 | - | Z-score |
| 1 | SpeedError | 转速误差 (Ref - GenSpeed) | - | Z-score |
| 2 | WindSpeed | 风速 | - | Z-score |
| 3 | PitchAngle | 当前桨距角测量值 | - | Z-score |
| 4 | TowerAcc | 塔架加速度 | - | Z-score |

状态维度 `STATE_DIM = 5`，所有状态经 Z-score 归一化（均值0，标准差1）。

### 3.2 动作空间 (Action Space)

| 属性 | 值 |
|------|-----|
| 维度 | 1 (`ACTION_DIM = 1`) |
| 含义 | 桨距角指令 (Pitch Demand) |
| 范围 | `[MIN_PITCH, MAX_PITCH] = [0.0, 1.57]` rad |

Actor 输出通过 `sigmoid` 映射到 `[0, 1.57]`。

### 3.3 奖励函数 (Reward Function)

```
r = -10.0 × (speed_error)² - 2.0 × (tower_acc)²
r_scaled = r × 0.01   # 缩放至 ~[-2, 0]
```

- **转速误差惩罚**（权重 10）：约束发电机转速跟随参考值
- **塔架加速度惩罚**（权重 2）：抑制塔架振动，保护机械结构
- **奖励缩放**（因子 0.01）：将原始范围 `~[-200, 0]` 映射到 `~[-2, 0]`，避免大梯度破坏训练稳定性

### 3.4 转移与终止

- **下一状态 `next_state`**: 由于是离线数据，采用 `shift(-1)` 近似——`state[i+1]` 作为 `state[i]` 的下一状态，最后一行的 `next_state` 设为自身
- **终止标志 `done`**: 全零向量，仅最后一行设为 1，表示轨迹结束
- **折扣因子**: `γ = 0.99`

---

## 4. 数据流水线

### 4.1 数据加载 (`data/dataset.py`)

核心函数 `load_dataset(csv_file)` 执行以下流程：

```
pd.read_csv → 提取列 → 构造 state/action/reward/next_state/done → Normalizer → 返回五元组
```

**状态构造细节**：
```python
state = np.column_stack([gen_speed, speed_error, wind_speed, pitch, tower_acc])
# shape: [N, 5]
```

**奖励构造**：
```python
reward = -10.0 * speed_error² - 2.0 * tower_acc²   # shape: [N]
reward = reward * 0.01  # 缩放
```

**下一状态构造**：
```python
next_state[i] = state[i+1]    # i ∈ [0, N-2]
next_state[-1] = state[-1]     # 最后一行自循环
```

**终止标志**：
```python
done = zeros(N)
done[-1] = 1
```

### 4.2 数据归一化 (`data/normalizer.py`)

`Normalizer` 类实现 Z-score 标准化：

| 方法 | 功能 |
|------|------|
| `fit(x)` | 计算各维度均值 `mean` 和标准差 `std`；若 `std < 1e-8` 则置为 1.0（防零除） |
| `transform(x)` | `(x - mean) / std` |
| `inverse(x)` | `x * std + mean` |
| `save(path)` | 保存 mean/std 到 JSON 文件 |
| `load(path)` | 从 JSON 加载 mean/std（类方法） |

归一化参数在训练时保存至 `output/normalization.json`，供推理部署时使用。

---

## 5. 神经网络模型

### 5.1 Actor 策略网络 (`models/actor.py`)

```
输入: state [B, 5]
  │
  ▼ Linear(5, 256) + ReLU
  │
  ▼ Linear(256, 256) + ReLU
  │
  ▼ Linear(256, 128) + ReLU
  │
  ▼ Linear(128, 1)
  │
  ▼ sigmoid → [0, 1]
  │
  ▼ scale: MIN_PITCH + x * (MAX_PITCH - MIN_PITCH)
  │
输出: pitch_demand [B, 1] ∈ [0, 1.57]
```

**设计要点**：
- 三层隐藏层（256→256→128），使用 ReLU 激活
- 输出层不做激活，而是通过 `sigmoid` 缩放至合法动作范围
- 参数量：约 101,633

### 5.2 Critic Q值网络 (`models/cql.py`)

```
输入: state [B, 5] + action [B, 1] = [B, 6]
  │
  ▼ concat(s, a) along dim=-1
  │
  ▼ Linear(6, 256) + ReLU
  │
  ▼ Linear(256, 256) + ReLU
  │
  ▼ Linear(256, 128) + ReLU
  │
  ▼ Linear(128, 1)
  │
输出: Q(s, a) [B, 1]
```

**设计要点**：
- 状态与动作在最后一维拼接后送入网络
- 与 Actor 结构对称，便于训练平衡
- 目标网络 `target_critic` 为 Critic 的 Polyak 平均副本

---

## 6. CQL 训练算法

### 6.1 训练循环 (`train.py`)

每个 epoch 对每个 batch 执行三步更新：

```
for epoch in 1..500:
    for batch in DataLoader:
        1. Critic更新
        2. Actor更新
        3. 目标网络软更新
```

### 6.2 Critic 更新

#### Bellman 误差（标准 TD 学习）

```
na = actor(ns)                                    # 目标策略动作
target_q = target_critic(ns, na)                   # 目标Q值
y = r + γ * (1 - d) * target_q                    # TD目标
bellman_loss = MSE(q, y)                           # 标准TD误差
```

#### CQL 保守性惩罚

```
rand_actions ~ Uniform(0, MAX_PITCH)               # 采样N个随机动作 [B, N, 1]
q_rand = critic(s, rand_actions)                   # 随机动作的Q值 [B, N]
cql_loss = mean(logsumexp(q_rand, dim=1)) - mean(q)# 保守性惩罚
```

- **logsumexp 项**: 推高所有随机动作的 Q 值 → 优化时该项降低 → 随机动作 Q 值被压低
- **-mean(q)**: 确保数据集中真实动作的 Q 值不被压低

#### 总 Critic 损失

```
critic_loss = bellman_loss + CQL_ALPHA * cql_loss
```

其中 `CQL_ALPHA = 1.0` 控制保守性强度。

### 6.3 Actor 更新

#### TD3+BC 风格自适应行为克隆

```
q_pred = critic(s, actor(s))       # 当前策略的Q值
bc_loss = MSE(actor(s), a)         # 行为克隆损失（与数据集动作一致）

λ = LAMBDA_BC / (|q|.mean() + 1e-6)  # 自适应权重

actor_loss = -λ * mean(q_pred) + bc_loss
```

**自适应 λ 机制**：当 Q 值幅度大时自动降低 BC 权重，当 Q 值幅度小时增大 BC 权重——保证 BC 与 Q 最大化的稳健平衡。

### 6.4 目标网络软更新

```
θ_target = τ * θ + (1 - τ) * θ_target    # τ = 0.005
```

Polyak 平均使目标网络平滑变化，稳定训练。

### 6.5 训练稳定性措施

| 措施 | 参数/方式 | 目的 |
|------|-----------|------|
| 梯度裁剪 | `GRAD_CLIP = 10.0` | 防止梯度爆炸 |
| 奖励缩放 | `REWARD_SCALE = 0.01` | 数值稳定 |
| 状态归一化 | Z-score | 消除量纲差异 |
| 降低保守性 | `CQL_ALPHA = 1.0`（原 5.0） | 避免Q值过度悲观 |
| 降低Critic学习率 | `LR_CRITIC = 1e-4`（与Actor一致） | 训练平衡 |

---

## 7. 关键设计决策与超参数

### 7.1 超参数总表

| 参数 | 值 | 说明 |
|------|-----|------|
| `STATE_DIM` | 5 | 状态维度 |
| `ACTION_DIM` | 1 | 动作维度 |
| `MIN_PITCH` | 0.0 | 最小桨距角 (rad) |
| `MAX_PITCH` | 1.57 | 最大桨距角 (rad) |
| `BATCH_SIZE` | 512 | 批大小 |
| `GAMMA` | 0.99 | 折扣因子 |
| `TAU` | 0.005 | 目标网络软更新系数 |
| `LR_ACTOR` | 1e-4 | Actor 学习率 |
| `LR_CRITIC` | 1e-4 | Critic 学习率 |
| `CQL_ALPHA` | 1.0 | CQL 保守性惩罚权重 |
| `CQL_NUM_SAMPLES` | 10 | 每状态随机动作采样数 |
| `LAMBDA_BC` | 2.5 | BC 正则化基础权重 |
| `REWARD_SCALE` | 0.01 | 奖励缩放因子 |
| `GRAD_CLIP` | 10.0 | 梯度裁剪阈值 |
| `EPOCHS` | 500 | 训练轮数 |

### 7.2 关键设计取舍

| 设计选择 | 理由 |
|----------|------|
| CQL over BCQ/BEAR | CQL 实现简单，logsumexp 惩罚在动作维度上直接且有效 |
| TD3+BC 自适应 λ over 固定 λ | Q 值幅度随训练变化，固定权重难以平衡 |
| Z-score over MinMax 归一化 | Z-score 对异常值更鲁棒，保持分布形状 |
| 单 Critic over 双 Critic | 简化实现；CQL 本身的保守性已提供足够的低估偏置 |
| ONNX opset 11 | 广泛兼容的部署标准 |
| 512 批大小 over 256 | 离线数据充足时，大批量提供更稳定的梯度估计 |

---

## 8. 模型导出与部署

### 8.1 ONNX 导出 (`export_onnx.py`)

```
Actor(pth) → model.eval() → dummy_input [1, 5] → torch.onnx.export → actor.onnx
```

**导出配置**：
- `input_names`: `["state"]`
- `output_names`: `["pitch_demand"]`
- `dynamic_axes`: batch 维度设为动态，支持任意批次大小
- `opset_version`: 11

### 8.2 推理部署所需文件

| 文件 | 用途 |
|------|------|
| `output/actor.onnx` | ONNX 模型（跨平台推理） |
| `output/actor_best.pth` | PyTorch 权重（可继续训练） |
| `output/normalization.json` | 归一化参数（推理时需对输入做相同归一化） |

---

## 9. 可视化与评估体系

### 9.1 训练过程监控 (`visualization/plot_training.py`)

生成三类图表：

| 图表 | 内容 | 输出文件 |
|------|------|----------|
| Loss Curves | Actor/Critic/Bellman/CQL 四合一损失曲线 | `training_losses.png` |
| Q-Value & Return | 平均 Q 值趋势 & 平均即时奖励 | `q_value_return.png` |
| Training Summary | 6合1综合训练摘要图 | `training_summary.png` |

所有曲线包含原始值和 10-epoch 滑动平均平滑线。支持中英双语标注。

### 9.2 策略评估 (`visualization/evaluate.py`)

| 评估维度 | 分析方法 | 输出文件 |
|----------|----------|----------|
| 动作对比 | 散点图 (pred vs true)、误差直方图、时序对比、分布对比 | `action_comparison.png` |
| 策略质量 | 动作偏差分布、逐样本偏差、状态-动作关系 | `policy_quality.png` |
| 状态覆盖 | 5维状态 vs 动作散点图、动作值分布 | `state_action_coverage.png` |

**定量指标**：MAE、RMSE、Max Error、动作均值/标准差对比、奖励统计。

**评估结论等级**：
| MAE 范围 | 等级 |
|-----------|------|
| < 0.01 | 优秀 (EXCELLENT) |
| < 0.05 | 良好 (GOOD) |
| < 0.10 | 一般 (FAIR) |
| ≥ 0.10 | 较差 (POOR) |

---

## 10. 模块清单

| 文件路径 | 职责 | 行数 |
|----------|------|------|
| `config.py` | 全局超参数配置 | 29 |
| `train.py` | CQL 训练主循环 | 241 |
| `export_onnx.py` | ONNX 模型导出 | 47 |
| `data/dataset.py` | CSV 加载、状态/奖励构造、五元组生成 | 83 |
| `data/normalizer.py` | Z-score 归一化器（含持久化） | 46 |
| `models/actor.py` | Actor 策略网络 | 40 |
| `models/cql.py` | Critic Q值网络 | 33 |
| `env/reward.py` | 奖励函数占位（实际在 dataset.py 中） | 33 |
| `utils/save_scaler.py` | 归一化器保存占位（已集成至 Normalizer） | 2 |
| `utils/load_scaler.py` | 归一化器加载占位（已集成至 Normalizer） | 2 |
| `visualization/plot_training.py` | 训练曲线可视化（中英双语） | 208 |
| `visualization/evaluate.py` | 策略评估与可视化（中英双语） | 268 |

---

## 附录 A: 运行命令

```bash
# 1. 训练
python train.py

# 2. 生成训练曲线
python visualization/plot_training.py

# 3. 策略评估
python visualization/evaluate.py

# 4. ONNX 导出
python export_onnx.py
```

## 附录 B: 依赖环境

```
torch, numpy, pandas, matplotlib, onnx
```

## 附录 C: CQL 损失函数公式总结

**Critic 总损失**：

$$\mathcal{L}_{critic} = \underbrace{\mathbb{E}_{(s,a,s')\sim\mathcal{D}}\left[(Q(s,a) - (r + \gamma (1-d) Q_{\text{target}}(s', \pi(s'))))^2\right]}_{\text{Bellman TD Error}} + \alpha \cdot \underbrace{\left[\mathbb{E}_{s\sim\mathcal{D}}\left[\log\sum_{a'}\exp Q(s,a')\right] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]\right]}_{\text{CQL Conservative Penalty}}$$

**Actor 总损失**：

$$\mathcal{L}_{actor} = -\underbrace{\frac{\lambda_{bc}}{\text{mean}(|Q|)}}_{\text{自适应}\lambda} \cdot \mathbb{E}_{s\sim\mathcal{D}}[Q(s, \pi(s))] + \underbrace{\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[(\pi(s) - a)^2\right]}_{\text{BC Regularization}}$$

---

> **文档版本**: v1.0 | **生成日期**: 2026-06-20 | **覆盖范围**: 全项目 12 个 Python 模块
