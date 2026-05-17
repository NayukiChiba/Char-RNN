"""命令行参数解析"""

import argparse

from config.defaults import InferenceParams, ModelParams, TrainingParams
from config.paths import get_best_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    """构建参数解析器，包含 train / eval / generate 三个子命令"""
    parser = argparse.ArgumentParser(
        description="Char-RNN 字符级文本生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="运行模式")

    # ---- train ----
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument(
        "--rnn_type",
        type=str,
        default=None,
        choices=["RNN", "LSTM", "GRU"],
        help=f"RNN 类型 (默认: {ModelParams.RNN_TYPE})",
    )
    p_train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"训练轮数 (默认: {TrainingParams.EPOCHS})",
    )
    p_train.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"学习率 (默认: {TrainingParams.LEARNING_RATE})",
    )
    p_train.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["Adam", "SGD", "AdamW"],
        help=f"优化器 (默认: {TrainingParams.OPTIMIZER})",
    )
    p_train.add_argument(
        "--lr_scheduler",
        type=str,
        default=None,
        choices=["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
        help=f"学习率调度器 (默认: {TrainingParams.LR_SCHEDULER})",
    )
    p_train.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练",
    )

    # ---- eval ----
    p_eval = sub.add_parser("eval", help="评估模型")
    p_eval.add_argument(
        "--checkpoint",
        type=str,
        default=str(get_best_checkpoint_path("LSTM")),
        help="检查点路径，默认使用 LSTM best.pth",
    )
    p_eval.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="评估数据集 (默认: val)",
    )

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="生成文本")
    p_gen.add_argument(
        "--checkpoint",
        type=str,
        default=str(get_best_checkpoint_path("LSTM")),
        help="检查点路径，默认使用 LSTM best.pth",
    )
    p_gen.add_argument("--prompt", type=str, default="ROMEO:", help="提示词")
    p_gen.add_argument(
        "--length",
        type=int,
        default=InferenceParams.MAX_GEN_LENGTH,
        help="生成文本的最大长度",
    )
    p_gen.add_argument(
        "--temperature",
        type=float,
        default=InferenceParams.TEMPERATURE,
        help="采样温度 (<1 保守, >1 随机)",
    )
    p_gen.add_argument(
        "--top_k",
        type=int,
        default=InferenceParams.TOP_K,
        help="top-k 采样的 k 值",
    )

    return parser
