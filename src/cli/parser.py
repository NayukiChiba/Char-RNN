"""命令行参数解析"""

import argparse

from config.defaults import InferenceParams
from config.paths import get_best_checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    """构建参数解析器"""
    parser = argparse.ArgumentParser(
        description="Char-RNN 文本生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --checkpoint LSTM --prompt "ROMEO:"
  python main.py                                                # 无参数进入菜单
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(get_best_checkpoint_path("LSTM")),
        help="检查点路径，默认使用 LSTM best.pth",
    )
    parser.add_argument("--prompt", type=str, default="ROMEO:", help="提示词")
    parser.add_argument(
        "--length",
        type=int,
        default=InferenceParams.MAX_GEN_LENGTH,
        help="生成文本的最大长度",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=InferenceParams.TEMPERATURE,
        help="采样温度 (<1 保守, >1 随机)",
    )
    parser.add_argument(
        "--top_k", type=int, default=InferenceParams.TOP_K, help="top-k 采样的 k 值"
    )
    return parser
