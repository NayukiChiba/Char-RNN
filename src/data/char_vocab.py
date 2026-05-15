"""
字符映射表
构建 char <-> index 双向映射
提供 encode 和 decode 方法

"""


class CharVocab:
    """char <-> index 双向映射表"""

    def __init__(self, text: str, min_freq: int = 1):
        """
        统计字符频率并构建映射表
        Args:
            text: 输入文本
            min_freq: 字符最少出现次数
        """
        freq: dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # 按照频率过滤字符 + 排序
        chars = sorted([char for char, count in freq.items() if count >= min_freq])

        # 构建映射表
        self.char2idx: dict[str, int] = {char: i for i, char in enumerate(chars)}
        self.idx2char: dict[int, str] = {i: char for char, i in self.char2idx.items()}

    def encode(self, text: str) -> list[int]:
        """将文本编码为索引列表"""
        return [self.char2idx[char] for char in text if char in self.char2idx]

    def decode(self, indices: list[int]) -> str:
        return "".join(
            self.idx2char[index] for index in indices if index in self.idx2char
        )

    def __len__(self) -> int:
        """返回词表大小"""
        return len(self.char2idx)
