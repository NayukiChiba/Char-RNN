"""
数据模块

用法:
    from src.data import train_loader, val_loader, test_loader, vocab
"""

from src.data.char_vocab import CharVocab
from src.data.data_loader import create_dataloaders
from src.data.process import get_indices_and_vocab


def get_dataloaders():
    """
    返回 train / val / test DataLoader 和 vocab

    数据流:
        shakespeare.txt -> process(优先读缓存) -> create_dataloaders
    """
    indices, vocab = get_indices_and_vocab()
    train_loader, val_loader, test_loader = create_dataloaders(indices)
    return train_loader, val_loader, test_loader, vocab


train_loader, val_loader, test_loader, vocab = get_dataloaders()

__all__ = ["CharVocab", "train_loader", "val_loader", "test_loader", "vocab"]
