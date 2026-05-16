"""
训练模块
1. 单epoch训练
2. 验证
3. 训练循环
4. 检查点存取
5. log存储

"""

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from config.defaults import DefaultParams, ModelParams, TrainingParams
from config.paths import get_best_checkpoint_path, get_latest_checkpoint_path
from src.data import test_loader, train_loader, val_loader
from src.models import create_model
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.utils import calc_perplexity, init_hidden

optimizer = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}


class Trainer:
    def __init__(self):
        self.device = DefaultParams.DEVICE
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer.get(TrainingParams.OPTIMIZER)
        self.criterion = nn.CrossEntropyLoss()

        # 训练超参数
        self.epochs = TrainingParams.EPOCHS
        self.clip = TrainingParams.CLIP_GRAD
        self.lr = TrainingParams.LEARNING_RATE
        self.lr_scheduler = TrainingParams.LR_SCHEDULER

        # 模型参数
        # 训练时会重新创建并加载词表大小
        self.model = create_model(ModelParams.RNN_TYPE, vocab_size=None)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
        }
        self.current_epoch = 0

        # 保存模型和优化器状态的路径
        self.best_checkpoint_path = get_best_checkpoint_path(
            model_name=ModelParams.RNN_TYPE
        )
        self.latest_checkpoint_path = get_latest_checkpoint_path(
            model_name=ModelParams.RNN_TYPE
        )

    def train_one_epoch(self) -> tuple[float, float]:
        """单epoch训练"""

        self.model.train()
        total_loss = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {self.current_epoch + 1:2d}/{self.epochs}",
        )
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            hidden = init_hidden(self.model, x.size(0))
            output, _ = self.model(x, hidden)
            loss = self.criterion(output.transpose(1, 2), y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "perplexity": f"{calc_perplexity(loss.item()):.4f}",
                }
            )

        return total_loss / len(self.train_loader), calc_perplexity(
            total_loss / len(self.train_loader)
        )

    def validate(self) -> tuple[float, float]:
        """
        验证

        """
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"[Val] Epoch {self.current_epoch + 1:2d}/{self.epochs}",
            )
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)

                hidden = init_hidden(self.model, x.size(0))
                output, _ = self.model(x, hidden)
                loss = self.criterion(output.transpose(1, 2), y)

                total_loss += loss.item()
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "perplexity": f"{calc_perplexity(loss.item()):.4f}",
                    }
                )

        return total_loss / len(self.val_loader), calc_perplexity(
            total_loss / len(self.val_loader)
        )

    def fit(self, resume_from: Path | None = None):
        """
        训练循环
        """
        start_epoch = 1

        if resume_from is not None and resume_from.exists():
            start_epoch, self.history = load_checkpoint(
                resume_from, self.model, self.optimizer
            )

            start_epoch += 1  # 从下一个epoch开始训练
            print(
                f"Resuming training from epoch {start_epoch} with history: {self.history}"
            )

        for epoch in range(start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_loss, train_perplexity = self.train_one_epoch()
            val_loss, val_perplexity = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_perplexity"].append(train_perplexity)
            self.history["val_perplexity"].append(val_perplexity)

            # 每个epoch结束后保存检查点
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.history,
                self.latest_checkpoint_path,
            )

        return self.history
