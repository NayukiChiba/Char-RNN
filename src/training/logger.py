"""
训练日志模块

基于 logging + TensorBoard 双后端.

用法:
    logger = Logger(name="shakespeare_lstm")
    logger.start()               # 打开日志 + 写入开始信息
    logger.log_config()          # 记录超参数

    for epoch in range(epochs):
        train_loss, val_loss = ...
        logger.log_epoch(epoch, train_loss, val_loss)
        if val_loss < best_loss:
            logger.log_best(epoch, val_loss)

    logger.finish(best_epoch, best_val_loss)  # 写入结束信息 + 关闭日志

输出:
  终端: 实时打印
  文件: outputs/logs/{name}_{timestamp}.log
  TensorBoard: outputs/tensorboard/{name}_{timestamp}/
"""

import logging
import math
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from config import paths
from config.defaults import DefaultParams, ModelParams, TrainingParams


class Logger:
    """训练日志器"""

    def __init__(
        self,
        model: str = ModelParams.RNN_TYPE,
        dataset: str = "shakespeare",
        use_tensorboard: bool = True,
    ):
        self.name = f"{model}_{dataset}"
        self.use_tensorboard = use_tensorboard

        # 时间戳,保证每次运行不互相覆盖
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- logging 后端 ----
        self._log = logging.getLogger(f"char-rnn.{self.name}.{self._timestamp}")
        self._log.setLevel(logging.INFO)
        self._log.handlers.clear()

        # 终端 handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        self._log.addHandler(console)

        # 文件 handler
        self._log_path = paths.LOGS_DIR / f"{self.name}_{self._timestamp}.log"
        file_handler = logging.FileHandler(str(self._log_path), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        )
        self._log.addHandler(file_handler)

        # ---- TensorBoard 后端 ----
        self._writer: SummaryWriter | None = None
        self._tensorboard_dir = paths.TENSORBOARD_DIR / f"{self.name}_{self._timestamp}"

    # ---- 生命周期:start / finish 控制训练起止 ----

    def start(self):
        """开始记录:打开 TensorBoard + 写入日志头"""
        if self.use_tensorboard:
            self._tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self._tensorboard_dir))

        self._log.info("=" * 60)
        self._log.info(" 训练开始")
        self._log.info("  时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log.info("  日志: %s", self._log_path)
        if self.use_tensorboard:
            self._log.info("  TensorBoard: %s", self._tensorboard_dir)
        self._log.info("=" * 60)

    def finish(self, best_epoch: int, best_val_loss: float):
        """结束记录:写入摘要 + 关闭 TensorBoard"""
        self._log.info("-" * 60)
        self._log.info(
            "训练完成 — best epoch: %d (val_loss: %.4f, ppl: %.1f)",
            best_epoch,
            best_val_loss,
            math.exp(best_val_loss),
        )
        self._log.info("结束时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log.info("=" * 60)

        if self._writer:
            self._writer.close()
            self._writer = None
        self._log.handlers.clear()

    # ---- 中间过程日志 ----

    def log_config(self):
        """记录超参数配置"""
        self._log.info(">>> 模型参数")
        self._log.info("  RNN 类型:     %s", ModelParams.RNN_TYPE)
        self._log.info("  嵌入维度:     %d", ModelParams.EMBEDDING_DIM)
        self._log.info("  隐藏层维度:   %d", ModelParams.HIDDEN_DIM)
        self._log.info("  RNN 层数:     %d", ModelParams.NUM_LAYERS)
        self._log.info("  Dropout:      %.2f", ModelParams.DROPOUT)
        self._log.info(">>> 训练参数")
        self._log.info("  学习率:       %.4f", TrainingParams.LEARNING_RATE)
        self._log.info("  训练轮数:     %d", TrainingParams.EPOCHS)
        self._log.info("  梯度裁剪:     %.1f", TrainingParams.CLIP_GRAD)
        self._log.info("  权重衰减:     %.0e", TrainingParams.WEIGHT_DECAY)
        self._log.info("  优化器:       %s", TrainingParams.OPTIMIZER)
        self._log.info("  学习率调度:   %s", TrainingParams.LR_SCHEDULER)
        self._log.info(">>> 运行时")
        self._log.info("  设备:         %s", DefaultParams.DEVICE)
        self._log.info("  随机种子:     %d", DefaultParams.SEED)
        self._log.info("-" * 60)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """记录单个 epoch 指标(文本 + TensorBoard 标量)"""
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        self._log.info(
            "Epoch %3d | train_loss: %.4f (%.1f ppl) | val_loss: %.4f (%.1f ppl)",
            epoch,
            train_loss,
            train_ppl,
            val_loss,
            val_ppl,
        )

        if self._writer:
            self._writer.add_scalar("Loss/train", train_loss, epoch)
            self._writer.add_scalar("Loss/val", val_loss, epoch)
            self._writer.add_scalar("Perplexity/train", train_ppl, epoch)
            self._writer.add_scalar("Perplexity/val", val_ppl, epoch)

    def log_best(self, epoch: int, val_loss: float):
        """记录最佳模型更新事件"""
        self._log.info("  -> best 模型! epoch %d (val_loss: %.4f)", epoch, val_loss)
