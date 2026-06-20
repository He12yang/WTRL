# export_onnx.py
# 将训练好的Actor模型导出为ONNX格式，支持动态batch维度，用于部署推理

import torch

from models.actor import Actor
from config import STATE_DIM


model = Actor(STATE_DIM)

model.load_state_dict(
    torch.load(
        "output/actor_best.pth",
        map_location="cpu"
    )
)

model.eval()

dummy = torch.randn(
    1,
    STATE_DIM
)

torch.onnx.export(
    model,
    dummy,
    "output/actor.onnx",

    input_names=["state"],

    output_names=["pitch_demand"],

    dynamic_axes={
        "state": {
            0: "batch"
        },
        "pitch_demand": {
            0: "batch"
        }
    },

    opset_version=11
)

print("ONNX export finished")