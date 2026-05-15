from pathlib import Path

# === 根目录 ===
ROOT_DIR = Path(__file__).resolve().parent.parent


def get_dir(path: Path):
    """确保目录存在，如果不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# === 数据目录 ===
DATASETS_DIR = get_dir(ROOT_DIR / "datasets")
RAW_DATASETS_DIR = get_dir(DATASETS_DIR / "raw")
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / "processed")
SHAKESPEARE_DATASET_PATH = RAW_DATASETS_DIR / "shakespeare.txt"
PROCESSED_SHAKESPEARE_DATASET_PATH = (
    PROCESSED_DATASETS_DIR / "shakespeare_processed.json"
)

# === 输出目录 ===
OUTPUTS_DIR = get_dir(ROOT_DIR / "outputs")
CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
VISUALIZATIONS_DIR = get_dir(OUTPUTS_DIR / "visualizations")
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")
