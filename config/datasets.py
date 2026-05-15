"""
数据加载与预处理配置
"""


class DataParams:
    """数据加载与预处理参数"""

    # 序列长度(滑动窗口大小)
    SEQ_LENGTH: int = 100

    # 窗口滑动步长
    STEP: int = 10

    # 批大小
    BATCH_SIZE: int = 64

    # DataLoader 工作进程数
    NUM_WORKERS: int = 4

    # 是否将数据 pin 到 GPU 内存
    PIN_MEMORY: bool = True

    # 是否打乱数据
    SHUFFLE: bool = True

    # 训练集比例
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1

    # 截取前 N 个字符(None = 全量)
    CHAR_LIMIT: int | None = None

    # 字符最少出现次数(低于此频率映射为 <unk>)
    MIN_FREQ: int = 1
