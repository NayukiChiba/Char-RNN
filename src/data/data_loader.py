"""
滑动窗口数据加载器

将索引序列切分为 (input, target) 对:
    input = chars[i : i + seq_length]
    target = chars[i + 1 : i + seq_length + 1]

Returns:
    tuple[DataLoader, DataLoader, DataLoader]: 训练集、验证集、测试集的数据加载器
"""

import torch
from torch.utils.data import DataLoader, Dataset

from config.datasets import DataParams


class CharDataset(Dataset):
    """字符滑动窗口数据集"""

    def __init__(
        self,
        indices: list[int],
        seq_length: int = DataParams.SEQ_LENGTH,
        step: int = DataParams.STEP,
    ):
        super().__init__()
        self.indices = indices
        self.seq_length = seq_length
        self.step = step
        # 计算样本数量
        self.num_samples = (len(indices) - seq_length) // step

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取 (input, target) 对
        使用方法:
            from src.data.data_loader import CharDataset
            dataset = CharDataset(indices)
            x, y = dataset[0]  # 获取第一个样本
        """

        start = idx * self.step
        end = start + self.seq_length
        x = torch.tensor(self.indices[start:end], dtype=torch.long)
        y = torch.tensor(self.indices[start + 1 : end + 1], dtype=torch.long)

        return x, y


def create_dataloaders(
    indices: list[int],
    seq_length: int = DataParams.SEQ_LENGTH,
    batch_size: int = DataParams.BATCH_SIZE,
    train_split: float = DataParams.TRAIN_SPLIT,
    val_split: float = DataParams.VAL_SPLIT,
    step: int = DataParams.STEP,
    num_workers: int = DataParams.NUM_WORKERS,
    pin_memory: bool = DataParams.PIN_MEMORY,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """创建 train / val / test 三个 DataLoader"""

    total = len(indices)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    def _make_loader(indices, shuffle):
        dataset = CharDataset(indices, seq_length, step)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = _make_loader(train_indices, shuffle=True)
    val_loader = _make_loader(val_indices, shuffle=False)
    test_loader = _make_loader(test_indices, shuffle=False)

    return train_loader, val_loader, test_loader
