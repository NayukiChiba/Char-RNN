"""
文本预处理

原始文本 -> 清洗截断 -> 构建词表 -> 索引序列

"""

from pathlib import Path

from config.datasets import DataParams
from src.data.char_vocab import CharVocab


def load_text(file_path: Path) -> str:
    """加载文本"""
    with file_path.open(encoding="utf-8") as f:
        text = f.read()

    # 截断文本
    if DataParams.CHAR_LIMIT is not None and len(text) > DataParams.CHAR_LIMIT:
        text = text[: DataParams.CHAR_LIMIT]

    return text


def build_vocab(text: str, min_freq: int = 1) -> CharVocab:
    """构建字符映射表"""
    return CharVocab(text, min_freq)


def text_to_indices(text: str, vocab: CharVocab) -> list[int]:
    """文本转索引序列"""
    return vocab.encode(text)
