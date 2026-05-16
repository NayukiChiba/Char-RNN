"""检查点存取"""

from pathlib import Path

import torch
from torch import nn

from config.defaults import DefaultParams, ModelParams


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict,
    filepath: Path,
):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "model_params": {
                "rnn_type": ModelParams.RNN_TYPE,
                "embed_dim": ModelParams.EMBEDDING_DIM,
                "hidden_dim": ModelParams.HIDDEN_DIM,
                "num_layers": ModelParams.NUM_LAYERS,
            },
        },
        filepath,
    )


def load_checkpoint(
    filepath: Path, model: nn.Module, optimizer: torch.optim.Optimizer = None
):
    checkpoint = torch.load(
        filepath, map_location=DefaultParams.DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("history", {})
