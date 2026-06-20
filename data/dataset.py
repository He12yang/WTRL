# data/dataset.py
# 离线数据集加载与预处理：读取CSV、构造状态/动作/奖励/下一状态/done五元组，并进行归一化

import pandas as pd
import numpy as np

from config import REWARD_SCALE
from data.normalizer import Normalizer


def load_dataset(csv_file):

    df = pd.read_csv(csv_file, index_col=False)

    gen_speed = df["GenSpeed"].values

    gen_ref = df["GenSpeed_Ref"].values

    speed_error = gen_ref - gen_speed

    wind_speed = df["WindSpeed"].values

    pitch = df["PitchAngle_Mea"].values

    tower_acc = df["TowerAcc"].values

    action = df["Action_PI_Dem"].values

    state = np.column_stack(
        [
            gen_speed,
            speed_error,
            wind_speed,
            pitch,
            tower_acc
        ]
    )

    reward = build_reward(
        speed_error,
        tower_acc
    )

    # 奖励缩放：将~[-200,0]映射到~[-2,0]，提升训练数值稳定性
    reward = reward * REWARD_SCALE

    next_state = np.vstack(
        [
            state[1:],
            state[-1]
        ]
    )

    done = np.zeros(len(state))

    done[-1] = 1

    norm = Normalizer()

    norm.fit(state)

    state = norm.transform(state)

    next_state = norm.transform(next_state)

    return (
        state.astype(np.float32),
        action.reshape(-1, 1).astype(np.float32),
        reward.reshape(-1, 1).astype(np.float32),
        next_state.astype(np.float32),
        done.reshape(-1, 1).astype(np.float32),
        norm
    )


def build_reward(speed_error, tower_acc):

    reward = (
        -10.0 * np.square(speed_error)
        -2.0 * np.square(tower_acc)
    )

    return reward