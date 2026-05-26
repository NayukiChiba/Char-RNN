"""训练辅助函数"""

import math

import torch
from torch import nn

from config.defaults import ModelParams
from src.models.lstm import CharLSTM


def init_hidden(model: nn.Module, batch_size: int):
    """初始化 RNN 隐藏状态,LSTM 返 (h, c),RNN/GRU 返 h"""
    device = next(model.parameters()).device
    h = torch.zeros(
        ModelParams.NUM_LAYERS, batch_size, ModelParams.HIDDEN_DIM, device=device
    )
    if isinstance(model, CharLSTM):
        c = torch.zeros(
            ModelParams.NUM_LAYERS, batch_size, ModelParams.HIDDEN_DIM, device=device
        )
        return (h, c)
    return h


def calc_perplexity(loss: float) -> float:
    """交叉熵 -> 困惑度"""
    return math.exp(loss)
