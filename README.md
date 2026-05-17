# Char-RNN

基于 PyTorch 的字符级 RNN 文本生成项目，使用莎士比亚数据集训练，支持 RNN / LSTM / GRU 三种模型。

## 功能

- **训练**：支持 RNN / LSTM / GRU，可自定义超参数，支持从检查点恢复
- **评估**：在验证集 / 测试集上计算损失和困惑度
- **生成**：temperature + top-k 采样的逐字续写
- **日志**：Python logging + TensorBoard 双后端
- **检查点**：自动保存 latest.pth 和 best.pth
- **CLI**：命令行 + 交互菜单两种使用方式

## 项目结构

```
Char-RNN/
├── main.py                        # 主入口
├── config/
│   ├── defaults.py                # 模型、训练、推理参数
│   ├── datasets.py                # 数据处理参数
│   └── paths.py                   # 路径管理
├── src/
│   ├── data/                      # 数据模块
│   │   ├── char_vocab.py          # 字符映射表
│   │   ├── process.py             # 数据预处理
│   │   └── data_loader.py         # DataLoader
│   ├── models/                    # 模型定义
│   │   ├── rnn.py                 # CharRNN
│   │   ├── lstm.py                # CharLSTM
│   │   └── gru.py                 # CharGRU
│   ├── training/                  # 训练模块
│   │   ├── trainer.py             # Trainer 类
│   │   ├── checkpoint.py          # 检查点存取
│   │   ├── logger.py              # 日志器 (logging + TensorBoard)
│   │   ├── optim.py               # 优化器 / 调度器工厂
│   │   └── utils.py               # 辅助函数
│   ├── evaluation/                # 评估模块
│   │   └── evaluator.py           # Evaluator 类
│   ├── inference/                 # 推理模块
│   │   └── inference.py           # Generator 类
│   └── cli/                       # 命令行模块
│       ├── parser.py              # 参数解析
│       └── menu.py                # 交互式菜单
├── outputs/                       # 输出目录
│   ├── checkpoints/               # 模型检查点
│   ├── logs/                      # 训练日志
│   └── tensorboard/               # TensorBoard 文件
└── notebooks/                     # Jupyter Notebook
```

## 安装

```bash
git clone <repo-url> && cd Char-RNN
uv sync                    # 安装依赖
uv run pre-commit install  # 安装 git hooks
```

## 快速开始

```bash
# 训练（使用默认 LSTM）
python main.py train

# 训练（指定超参数）
python main.py train --rnn_type GRU --epochs 30 --lr 0.002 --optimizer AdamW

# 从检查点恢复训练
python main.py train --resume outputs/checkpoints/LSTM/latest.pth

# 评估
python main.py eval --checkpoint outputs/checkpoints/LSTM/best.pth

# 生成文本
python main.py generate --prompt "ROMEO:" --temperature 0.8

# 交互式菜单
python main.py
```

## 命令行参数

### train

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--rnn_type` | RNN/LSTM/GRU | LSTM | RNN 类型 |
| `--epochs` | int | 20 | 训练轮数 |
| `--lr` | float | 0.001 | 学习率 |
| `--optimizer` | Adam/SGD/AdamW | Adam | 优化器 |
| `--lr_scheduler` | StepLR/CosineAnnealingLR/ReduceLROnPlateau | StepLR | 学习率调度器 |
| `--resume` | path | - | 恢复训练的检查点路径 |

### eval

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--checkpoint` | path | LSTM best.pth | 检查点路径 |
| `--split` | val/test | val | 评估数据集 |

### generate

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--checkpoint` | path | LSTM best.pth | 检查点路径 |
| `--prompt` | str | ROMEO: | 提示词 |
| `--length` | int | 500 | 最大生成长度 |
| `--temperature` | float | 0.8 | 采样温度 |
| `--top_k` | int | 50 | top-k 采样 k 值 |

## 配置

修改 `config/defaults.py` 调整默认参数：

```python
class ModelParams:
    RNN_TYPE = "LSTM"        # RNN / LSTM / GRU
    EMBEDDING_DIM = 256      # 嵌入维度
    HIDDEN_DIM = 256         # 隐藏层维度
    NUM_LAYERS = 2           # RNN 层数
    DROPOUT = 0.5            # Dropout 比例

class TrainingParams:
    LEARNING_RATE = 0.001
    EPOCHS = 20
    CLIP_GRAD = 5.0          # 梯度裁剪阈值
    OPTIMIZER = "Adam"
    LR_SCHEDULER = "StepLR"
```

## 开发

```bash
uv run ruff check .    # 代码检查
uv run ruff format .   # 代码格式化
uv run pytest          # 运行测试
```

## 数据集

使用莎士比亚作品集 (`shakespeare.txt`)
请自行在`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`下载
首次运行会自动下载并预处理，缓存到 `datasets/processed/`。
