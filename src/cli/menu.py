"""交互式菜单模式

支持训练、评估、生成三种模式。
"""

from pathlib import Path

from config.defaults import InferenceParams, ModelParams, TrainingParams
from config.paths import CHECKPOINTS_DIR
from src.evaluation import Evaluator
from src.inference import Generator
from src.training.trainer import Trainer


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


def _menu_train():
    """训练子菜单"""
    print("\n" + "-" * 40)
    print("  训练配置")
    print("-" * 40)

    # RNN 类型
    raw = input(
        f"RNN 类型 [1=RNN, 2=LSTM, 3=GRU, 默认 {ModelParams.RNN_TYPE}]: "
    ).strip()
    rnn_map = {"1": "RNN", "2": "LSTM", "3": "GRU"}
    if raw in rnn_map:
        ModelParams.RNN_TYPE = rnn_map[raw]

    # 训练轮数
    raw = input(f"训练轮数 [默认 {TrainingParams.EPOCHS}]: ").strip()
    if raw:
        TrainingParams.EPOCHS = int(raw)

    # 学习率
    raw = input(f"学习率 [默认 {TrainingParams.LEARNING_RATE}]: ").strip()
    if raw:
        TrainingParams.LEARNING_RATE = float(raw)

    # 优化器
    raw = input(
        f"优化器 [1=Adam, 2=SGD, 3=AdamW, 默认 {TrainingParams.OPTIMIZER}]: "
    ).strip()
    opt_map = {"1": "Adam", "2": "SGD", "3": "AdamW"}
    if raw in opt_map:
        TrainingParams.OPTIMIZER = opt_map[raw]

    # 学习率调度器
    raw = input(
        f"学习率调度器 [1=StepLR, 2=CosineAnnealingLR, 3=ReduceLROnPlateau, "
        f"默认 {TrainingParams.LR_SCHEDULER}]: "
    ).strip()
    sched_map = {"1": "StepLR", "2": "CosineAnnealingLR", "3": "ReduceLROnPlateau"}
    if raw in sched_map:
        TrainingParams.LR_SCHEDULER = sched_map[raw]

    # 恢复训练
    raw = input("从检查点恢复? [输入路径或回车跳过]: ").strip()
    resume = Path(raw) if raw else None

    print("\n开始训练...")
    trainer = Trainer()
    trainer.fit(resume_from=resume)


def _menu_eval():
    """评估子菜单"""
    checkpoint = _pick_checkpoint()
    if checkpoint is None:
        return

    raw = input("评估数据集 [val/test, 默认 val]: ").strip().lower()
    split = raw if raw in ("val", "test") else "val"

    print(f"\n加载: {checkpoint}")
    evaluator = Evaluator(checkpoint)
    if split == "val":
        loss, ppl = evaluator.eval_val()
    else:
        loss, ppl = evaluator.eval_test()

    print(f"\n[{split.upper()}] loss: {loss:.4f}  ppl: {ppl:.2f}")


def _menu_generate():
    """生成子菜单"""
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


def run():
    """启动交互式菜单"""
    while True:
        print("\n" + "=" * 40)
        print("  Char-RNN")
        print("=" * 40)
        print("  [1] 训练")
        print("  [2] 评估")
        print("  [3] 生成")
        print("  [4] 退出")

        choice = input("\n选择 [1-4]: ").strip()

        if choice == "1":
            _menu_train()
        elif choice == "2":
            _menu_eval()
        elif choice == "3":
            _menu_generate()
        elif choice == "4":
            print("退出。")
            break
        else:
            print("无效选择。")
