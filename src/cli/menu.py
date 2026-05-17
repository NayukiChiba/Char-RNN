"""交互式菜单模式"""

from pathlib import Path

from config.defaults import InferenceParams
from config.paths import CHECKPOINTS_DIR
from src.inference import Generator


def _find_checkpoints() -> dict[str, Path]:
    """扫描检查点目录，返回 {显示名: 路径}"""
    if not CHECKPOINTS_DIR.exists():
        return {}

    checkpoints: dict[str, Path] = {}
    for model_dir in CHECKPOINTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        for f in model_dir.iterdir():
            if f.suffix == ".pth":
                checkpoints[f"{model_dir.name}/{f.name}"] = f
    return checkpoints


def _pick_checkpoint() -> Path | None:
    """交互式选择检查点"""
    checkpoints = _find_checkpoints()
    if not checkpoints:
        print("没有找到检查点文件，请先训练模型。")
        return None

    items = sorted(checkpoints.items())
    print("\n可用的检查点:")
    for i, (label, _) in enumerate(items, 1):
        print(f"  [{i}] {label}")

    choice = input(f"\n选择检查点 [1-{len(items)}]: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx][1]
    except ValueError:
        pass
    print("无效选择。")
    return None


def run():
    """启动交互式菜单"""
    print("=" * 50)
    print("  Char-RNN 文本生成")
    print("=" * 50)

    checkpoint = _pick_checkpoint()
    if checkpoint is None:
        return

    print(f"\n加载: {checkpoint}")
    generator = Generator(checkpoint)

    prompt = input("\n提示词 (例如 ROMEO:): ").strip()
    if not prompt:
        print("提示词不能为空。")
        return

    raw = input(f"生成长度 [默认 {InferenceParams.MAX_GEN_LENGTH}]: ").strip()
    max_length = int(raw) if raw else InferenceParams.MAX_GEN_LENGTH

    raw = input(f"temperature [默认 {InferenceParams.TEMPERATURE}]: ").strip()
    temperature = float(raw) if raw else InferenceParams.TEMPERATURE

    raw = input(f"top_k [默认 {InferenceParams.TOP_K}]: ").strip()
    top_k = int(raw) if raw else InferenceParams.TOP_K

    print("\n" + "-" * 50)
    result = generator.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    print(result)
    print("-" * 50)
