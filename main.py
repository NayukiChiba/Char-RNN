"""Char-RNN 主入口

用法:
    python main.py train --rnn_type LSTM --epochs 30              # 训练
    python main.py eval --checkpoint outputs/checkpoints/LSTM/best.pth   # 评估
    python main.py generate --prompt "ROMEO:"                     # 生成
    python main.py                                                # 无参数 → 菜单模式
"""

import sys
from pathlib import Path

from config.defaults import ModelParams, TrainingParams
from src.cli.menu import run as run_menu
from src.cli.parser import build_parser
from src.evaluation import Evaluator
from src.inference import Generator
from src.training.trainer import Trainer


def _apply_train_overrides(args):
    """将命令行参数覆盖到配置类上"""
    if args.rnn_type is not None:
        ModelParams.RNN_TYPE = args.rnn_type
    if args.epochs is not None:
        TrainingParams.EPOCHS = args.epochs
    if args.lr is not None:
        TrainingParams.LEARNING_RATE = args.lr
    if args.optimizer is not None:
        TrainingParams.OPTIMIZER = args.optimizer
    if args.lr_scheduler is not None:
        TrainingParams.LR_SCHEDULER = args.lr_scheduler


def _run_train(args):
    _apply_train_overrides(args)
    resume = Path(args.resume) if args.resume else None
    trainer = Trainer()
    trainer.fit(resume_from=resume)


def _run_eval(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"检查点不存在: {checkpoint}")
        sys.exit(1)

    evaluator = Evaluator(checkpoint)
    if args.split == "val":
        loss, ppl = evaluator.eval_val()
    else:
        loss, ppl = evaluator.eval_test()

    print(f"\n[{args.split.upper()}] loss: {loss:.4f}  ppl: {ppl:.2f}")


def _run_generate(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"检查点不存在: {checkpoint}")
        sys.exit(1)

    print(f"加载检查点: {checkpoint}")
    generator = Generator(checkpoint)

    print(f"提示词: {args.prompt}")
    print(
        f"参数: temperature={args.temperature}, top_k={args.top_k}, "
        f"max_length={args.length}"
    )
    print("-" * 60)

    result = generator.generate(
        prompt=args.prompt,
        max_length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(result)


def main():
    # 无参数 → 菜单模式
    if len(sys.argv) == 1:
        run_menu()
        return

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "generate":
        _run_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
