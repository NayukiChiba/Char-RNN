"""
默认参数配置

- ModelParams: 模型相关参数
- TrainingParams: 训练相关参数
- InferenceParams: 推理相关参数

"""

from typing import Literal

import torch


class DefaultParams:
    SEED = 42

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelParams:
    """模型结构参数"""

    # RNN类型
    RNN_TYPE: Literal["RNN", "LSTM", "GRU"] = "LSTM"

    # 字符嵌入维度
    EMBEDDING_DIM = 256

    # RNN隐藏层维度
    HIDDEN_DIM = 512

    # RNN层数
    NUM_LAYERS = 3

    # 层间 dropout
    DROPOUT = 0.3


class TrainingParams:
    """训练相关参数"""

    # 学习率
    LEARNING_RATE = 0.001

    # 训练轮数
    EPOCHS = 20

    # 梯度裁剪阈值
    CLIP_GRAD = 5.0

    # 优化器
    OPTIMIZER: Literal["Adam", "SGD", "AdamW"] = "Adam"

    # 学习率调度器
    LR_SCHEDULER: Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"] = "StepLR"


class InferenceParams:
    """推理相关参数"""

    # 生成文本的最大长度
    MAX_GEN_LENGTH = 500

    # 采样温度
    TEMPERATURE = 0.8

    # top-k采样的k值
    TOP_K = 50

    # top-p采样的p值
    TOP_P = 0.9
