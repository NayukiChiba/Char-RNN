"""
数据模块

用法:
    from src.data import get_dataloaders

    train_loader, val_loader, test_loader, vocab = get_dataloaders()
"""

from config import paths
from config.datasets import DataParams

# from src.data.char_vocab import CharVocab
from src.data.data_loader import create_dataloaders
from src.data.process import build_vocab, load_text, text_to_indices


def get_dataloaders():
    """
    返回 train / val / test DataLoader 和 vocab

    数据流:
        shakespeare.txt → load_text → build_vocab → text_to_indices → create_dataloaders
    """
    text = load_text(paths.SHAKESPEARE_DATASET_PATH)
    vocab = build_vocab(text, DataParams.MIN_FREQ)
    indices = text_to_indices(text, vocab)

    train_loader, val_loader, test_loader = create_dataloaders(indices)

    return train_loader, val_loader, test_loader, vocab


trainLoader, val_loader, test_loader, vocab = get_dataloaders()


__all__ = ["trainLoader", "val_loader", "test_loader", "vocab"]
