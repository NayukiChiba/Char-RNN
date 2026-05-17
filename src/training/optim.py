"""优化器与学习率调度器工厂"""

import torch


def create_optimizer(params, name: str, lr: float, weight_decay: float = 1e-4):
    """根据名称创建优化器"""
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    elif name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知的优化器: {name}")


def create_lr_scheduler(optimizer: torch.optim.Optimizer, name: str):
    """根据名称创建学习率调度器"""
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
    elif name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        raise ValueError(f"未知的学习率调度器: {name}")
