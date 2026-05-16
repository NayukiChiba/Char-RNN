"""
GRU 字符级语言模型
"""

import torch.nn as nn
from config.defaults import ModelParams


class CharGRU(nn.Module):
    """Embedding → GRU → Dropout → Linear"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ModelParams.EMBEDDING_DIM)
        self.gru = nn.GRU(
            input_size=ModelParams.EMBEDDING_DIM,
            hidden_size=ModelParams.HIDDEN_DIM,
            num_layers=ModelParams.NUM_LAYERS,
            dropout=ModelParams.DROPOUT if ModelParams.NUM_LAYERS > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(ModelParams.HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden
