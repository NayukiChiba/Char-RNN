"""
模型模块

用法:
    from src.models import create_model

    model = create_model("lstm", vocab_size=len(vocab))
"""

from src.models.rnn import CharRNN
from src.models.lstm import CharLSTM
from src.models.gru import CharGRU


_MODEL_MAP = {
    "rnn": CharRNN,
    "lstm": CharLSTM,
    "gru": CharGRU,
}


def create_model(rnn_type: str, vocab_size: int):
    """根据类型创建模型"""
    if rnn_type not in _MODEL_MAP:
        raise ValueError(f"未知模型类型: {rnn_type}，可选: {list(_MODEL_MAP.keys())}")
    return _MODEL_MAP[rnn_type](vocab_size)

__all__ = ["create_model", "CharRNN", "CharLSTM", "CharGRU"]