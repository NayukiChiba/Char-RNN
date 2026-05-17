"""Char-RNN 主入口

用法:
    python main.py                                                # 无参数 → 菜单模式
    python main.py --checkpoint LSTM --prompt "ROMEO:"            # CLI 模式
"""

import sys
from pathlib import Path

from src.cli.menu import run as run_menu
from src.cli.parser import build_parser
from src.inference import Generator


def main():
    parser = build_parser()

    # 无参数 → 菜单模式
    if len(sys.argv) == 1:
        run_menu()
        return

    args = parser.parse_args()

    # 解析检查点路径
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"检查点不存在: {args.checkpoint}")
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


if __name__ == "__main__":
    main()
