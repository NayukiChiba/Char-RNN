"""
评估模块

加载检查点并在验证集/测试集上计算损失与困惑度。
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.defaults import DefaultParams, ModelParams
from src.data import test_loader, val_loader, vocab
from src.models import create_model
from src.training.checkpoint import load_checkpoint
from src.training.utils import calc_perplexity, init_hidden


class Evaluator:
    """模型评估器"""

    def __init__(self, checkpoint_path: Path):
        self.device = DefaultParams.DEVICE

        # 重建模型并加载权重
        self.model = create_model(ModelParams.RNN_TYPE, vocab_size=len(vocab))
        _, _ = load_checkpoint(checkpoint_path, self.model)
        self.model.to(self.device)
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str) -> tuple[float, float]:
        """计算给定数据加载器的平均损失和困惑度"""
        total_loss = 0

        pbar = tqdm(loader, desc=desc)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            hidden = init_hidden(self.model, x.size(0))
            output, _ = self.model(x, hidden)
            loss = self.criterion(output.transpose(1, 2), y)

            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{calc_perplexity(loss.item()):.2f}",
                }
            )

        avg_loss = total_loss / len(loader)
        return avg_loss, calc_perplexity(avg_loss)

    def eval_val(self) -> tuple[float, float]:
        """验证集评估"""
        return self.evaluate(val_loader, desc="[Val]")

    def eval_test(self) -> tuple[float, float]:
        """测试集评估"""
        return self.evaluate(test_loader, desc="[Test]")
