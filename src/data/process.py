"""
文本预处理

原始文本 -> 构建词表 -> 索引序列
优先读取 processed 缓存,无缓存时走完整管线并写入.
"""

import json
from pathlib import Path

from config import paths
from config.datasets import DataParams
from src.data.char_vocab import CharVocab


def load_text(file_path: Path) -> str:
    """加载原始文本"""
    with file_path.open(encoding="utf-8") as f:
        text = f.read()

    # 截断文本
    if DataParams.CHAR_LIMIT is not None and len(text) > DataParams.CHAR_LIMIT:
        text = text[: DataParams.CHAR_LIMIT]

    return text


def build_vocab(text: str, min_freq: int = DataParams.MIN_FREQ) -> CharVocab:
    """从文本构建字符映射表"""
    return CharVocab(text, min_freq)


def text_to_indices(text: str, vocab: CharVocab) -> list[int]:
    """文本 -> 索引序列"""
    return vocab.encode(text)


def get_indices_and_vocab() -> tuple[list[int], CharVocab]:
    """
    获取索引序列 + 词表

    优先读 processed 缓存文件,不存在则走完整预处理管线并写入缓存.
    """
    processed_path = paths.PROCESSED_SHAKESPEARE_DATASET_PATH

    if processed_path.exists():
        with processed_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # vocab 反序列化 + indices 直接读取
        vocab = CharVocab.from_dict(data["vocab"])
        indices = data["indices"]
    else:
        text = load_text(paths.SHAKESPEARE_DATASET_PATH)
        vocab = build_vocab(text, DataParams.MIN_FREQ)
        indices = text_to_indices(text, vocab)

        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with processed_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "indices": indices,
                    "vocab": vocab.to_dict(),
                },
                f,
            )

    return indices, vocab
