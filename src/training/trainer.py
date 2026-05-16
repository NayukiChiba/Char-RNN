"""
训练器

Trainer 类封装了完整的训练管线:
- 单 epoch 训练 (train_one_epoch)
- 验证 (validate)
- 完整循环 (fit),含断点续训和周期性 checkpoint 保存
"""

import torch
from torch import nn
from tqdm import tqdm

from config import paths
from config.defaults import DefaultParams, TrainingParams
from src.training.utils import init_hidden, calc_perplexity
from src.training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    训练器,封装训练 / 验证 / 完整训练循环

    用法:
        trainer = Trainer(model, train_loader, val_loader)
        history = trainer.fit()                        # 从头训练
        history = trainer.fit(resume_from=ckpt_path)   # 断点续训
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 交叉熵损失 — 语言模型标准 loss
        self.criterion = nn.CrossEntropyLoss()

        # Adam 优化器,weight_decay 做 L2 正则,防止过拟合
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TrainingParams.LEARNING_RATE,
            weight_decay=TrainingParams.WEIGHT_DECAY,
        )

        # 训练历史,记录每个 epoch 的 loss 和困惑度
        self.history: dict[str, list] = {
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
        }
        self.current_epoch = 0

    def train_one_epoch(self) -> float:
        """
        训练一个 epoch

        每个 batch:
          1. 前向传播得到 logits
          2. 交叉熵计算 loss(将 logits 转置以匹配 CrossEntropyLoss 的输入要求)
          3. 反向传播 + 梯度裁剪 + 参数更新

        返回该 epoch 的平均训练 loss
        """
        self.model.train()
        total_loss = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {self.current_epoch}/{TrainingParams.EPOCHS}",
            unit="batch",
        )
        for x, y in pbar:
            # 移到设备
            x, y = x.to(DefaultParams.DEVICE), y.to(DefaultParams.DEVICE)

            # 每个 batch 重新初始化隐藏状态,避免跨 batch 梯度传播
            hidden = init_hidden(self.model, x.size(0))

            # 前向传播
            self.optimizer.zero_grad()
            logits, _ = self.model(x, hidden)

            # logits: (batch, seq_len, vocab_size) → (batch, vocab_size, seq_len)
            # y:      (batch, seq_len)
            loss = self.criterion(logits.transpose(1, 2), y)
            loss.backward()

            # 梯度裁剪,防止梯度爆炸(LSTM 序列训练常见问题)
            nn.utils.clip_grad_norm_(self.model.parameters(), TrainingParams.CLIP_GRAD)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{calc_perplexity(loss.item()):.1f}",
            })

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        验证

        与训练的区别:
        - model.eval() 关闭 dropout,确保结果稳定
        - torch.no_grad() 关闭梯度计算,节省显存和计算
        - 不更新参数

        返回平均验证 loss
        """
        self.model.eval()
        total_loss = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"[Val ] Epoch {self.current_epoch}/{TrainingParams.EPOCHS}",
            unit="batch",
        )
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(DefaultParams.DEVICE), y.to(DefaultParams.DEVICE)
                hidden = init_hidden(self.model, x.size(0))
                logits, _ = self.model(x, hidden)
                loss = self.criterion(logits.transpose(1, 2), y)
                total_loss += loss.item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{calc_perplexity(loss.item()):.1f}",
                })

        return total_loss / len(self.val_loader)

    def fit(self, resume_from=None):
        """
        完整训练循环

        流程:
          1. (可选) 从 checkpoint 恢复模型、优化器、history
          2. 按 epoch 循环:训练 → 记录 → 验证 → 记录 → 打印
          3. 每 CHECKPOINT_INTERVAL 轮保存一次 checkpoint
          4. 训练结束后保存 final checkpoint

        返回 history 字典,可用于绘制 loss / perplexity 曲线
        """
        start_epoch = 1

        # 断点续训:恢复模型权重、优化器状态、历史记录
        if resume_from is not None and resume_from.exists():
            start_epoch, self.history = load_checkpoint(
                resume_from, self.model, self.optimizer
            )
            start_epoch += 1
            print(f"从 checkpoint 恢复: epoch {start_epoch}")

        for epoch in range(start_epoch, TrainingParams.EPOCHS + 1):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_one_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_ppl"].append(calc_perplexity(train_loss))

            # 验证
            val_loss = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_ppl"].append(calc_perplexity(val_loss))

            # 打印 epoch 摘要
            print(
                f"Epoch {epoch:2d}/{TrainingParams.EPOCHS} | "
                f"train_loss: {train_loss:.4f} ({calc_perplexity(train_loss):.1f} ppl) | "
                f"val_loss: {val_loss:.4f} ({calc_perplexity(val_loss):.1f} ppl)"
            )

            # 周期性保存 checkpoint
            if epoch % TrainingParams.CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.history,
                    paths.CHECKPOINTS_DIR / f"epoch_{epoch}.pt",
                )

        # 最终保存
        save_checkpoint(
            self.model, self.optimizer, TrainingParams.EPOCHS, self.history,
            paths.CHECKPOINTS_DIR / "final.pt",
        )
        return self.history
