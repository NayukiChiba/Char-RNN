"""
文本生成模块

从检查点加载模型,支持 temperature 和 top-k 采样的逐字续写.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from config.defaults import DefaultParams, InferenceParams, ModelParams
from src.data import vocab
from src.models import create_model
from src.training.checkpoint import load_checkpoint


class Generator:
    """文本生成器"""

    def __init__(self, checkpoint_path: Path):
        self.device = DefaultParams.DEVICE

        self.model = create_model(ModelParams.RNN_TYPE, vocab_size=len(vocab))
        _, _ = load_checkpoint(checkpoint_path, self.model)
        self.model.to(self.device)
        self.model.eval()

        self.vocab = vocab

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = InferenceParams.MAX_GEN_LENGTH,
        temperature: float = InferenceParams.TEMPERATURE,
        top_k: int = InferenceParams.TOP_K,
    ) -> str:
        """
        Args:
            prompt: 生成文本的提示词,例如 "ROMEO:"
            max_length: 生成文本的最大长度(包含提示词)
            temperature: 采样温度,控制生成文本的随机程度
            top_k: top-k 采样的 k 值,控制生成文本的合理程度
        Returns:
            生成的文本字符串

        从提示词开始逐字续写文本

        整体流程:
            提示词 -> 一次性编码 + 前向 -> 拿到初始隐藏状态和最后一个 logits
            循环 (每次生成一个字符):
                1. logits 除以 temperature,控制随机程度
                2. 只保留 top-k 个候选,其余置为 -inf
                3. softmax 转为概率,按概率采样一个字符
                4. 把采样的字符喂回模型,拿到新的 logits 和 hidden
            解码所有索引 -> 返回文本
        """
        # ---- 第一阶段: 编码提示词并获取初始状态 ----
        # 将字符串编码为索引列表,例如 "ROMEO:" -> [44, 32, 26, 18, 32, 2]
        indices = self.vocab.encode(prompt)
        if not indices:
            raise ValueError("提示词编码后为空,请检查输入文本是否在词表中")

        # 转为 (batch=1, seq_len) 的张量
        x = torch.tensor(indices, device=self.device).unsqueeze(0)

        # 提示词一次性通过模型,得到:
        #   logits:  (1, seq_len, vocab_size) — 每个位置对下一个字符的预测分数
        #   hidden:  RNN 的最终隐藏状态,包含了提示词的上下文信息
        logits, hidden = self.model(x)

        # 只取最后一个位置的 logits,因为我们要预测提示词之后的下一个字符
        # 形状: (1, vocab_size),即词表中每个字符的原始分数
        logits = logits[:, -1, :]

        # 记录所有已生成的索引(提示词 + 后续生成的字符)
        generated = list(indices)

        # ---- 第二阶段: 逐字生成 ----
        for _ in range(max_length):
            # 1. temperature 缩放
            # temperature < 1 -> 分布变尖锐,高分字符更突出(更保守)
            # temperature > 1 -> 分布变平滑,低分字符也有机会(更随机)
            # temperature = 1 -> 不做缩放
            if temperature != 1.0:
                logits = logits / temperature

            # 2. top-k 过滤
            # 只保留分数最高的 k 个字符,其余置为 -inf
            # softmax(e^{-inf}) = 0,这些字符永远不会被采样到
            # 目的: 截断概率分布的长尾,避免采到明显不合理的字符
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                # 找到分数最高的 k 个值及其索引
                top_k_values, top_k_indices = torch.topk(logits, k)
                # 创建全 -inf 的容器,只把 top-k 的值填回去
                mask = torch.full_like(logits, float("-inf"))
                logits = mask.scatter(1, top_k_indices, top_k_values)

            # 3. 概率采样
            # softmax 将 logits 转为概率分布(所有值加起来 = 1)
            probs = F.softmax(logits, dim=-1)
            # multinomial 按概率分布随机采样一个索引
            # 注意: 这里不是取 argmax(贪心),而是按概率掷骰子
            # 例如 ['e':0.3, 'a':0.2, 'o':0.15, ...] -> 可能采样到 'a'
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            # 4. 将新采样的字符作为输入,更新模型状态
            # 输入形状: (batch=1, seq_len=1),即只有刚生成的那一个字符
            x = torch.tensor([[next_token]], device=self.device)
            # hidden 从上次的状态继续,保持上下文的连续性
            logits, hidden = self.model(x, hidden)

        # 将所有索引解码回字符串
        return self.vocab.decode(generated)
