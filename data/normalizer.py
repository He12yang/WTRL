# data/normalizer.py
# Z-score归一化器：fit计算均值/标准差，transform进行标准化，支持保存/加载归一化参数

import json
import numpy as np


class Normalizer:

    def fit(self, x):

        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

        self.std[self.std < 1e-8] = 1.0

    def transform(self, x):

        return (x - self.mean) / self.std

    def inverse(self, x):

        return x * self.std + self.mean

    def save(self, path):

        obj = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

        with open(path, "w") as f:
            json.dump(obj, f, indent=4)

    @classmethod
    def load(cls, path):

        with open(path, "r") as f:
            obj = json.load(f)

        norm = cls()

        norm.mean = np.array(obj["mean"])
        norm.std = np.array(obj["std"])

        return norm