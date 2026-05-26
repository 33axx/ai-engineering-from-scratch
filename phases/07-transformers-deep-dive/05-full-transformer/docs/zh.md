# 完整Transformer——编码器 + 解码器

> 注意力机制是明星。其他一切——残差连接、归一化、前馈网络、交叉注意力——是让你能够深度堆叠它的脚手架。

**类型：** 构建
**语言：** Python
**前置课程：** 阶段7·02（自注意力）、阶段7·03（多头注意力）、阶段7·04（位置编码）
**时间：** ~75分钟

## 问题

单层注意力层是一个特征提取器，不是模型。每层一次矩阵乘法对于语言能力来说容量不够。你需要深度——而深度在没有正确架构的情况下会崩溃。

2017 年的 Vaswain 论文打包了六个设计决策，将一层注意力变成了一个可堆叠的块。此后的每一个Transformer——编码器专用（BERT）、解码器专用（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到了 2026 年，这些块已经被精炼（RMSNorm、SwiGLU、Pre-Norm、RoPE），但骨架是一样的。

这节课就是这个骨架。接下来的课程会把它专业化——06 针对编码器，07 针对解码器，08 针对编码器-解码器。

## 概念

![编码器和解码器块内部结构，连接图](../assets/full-transformer.svg)

### 六个组件

1. **嵌入 + 位置信号。** 词元 → 向量。通过 RoPE（现代）或正弦函数（经典）注入位置信息。
2. **自注意力。** 每个位置关注所有其他位置。解码器中会进行掩码处理。
3. **前馈网络（FFN）。** 位置独立的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比为 4 倍。
4. **残差连接。** `x + sublayer(x)`。没有这个，梯度在超过约 6 层后会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

### 编码器块（用于 BERT、T5 编码器）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器块（用于 GPT、T5 解码器）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每块有三个子层。中间的——交叉注意力——是信息从编码器流向解码器的唯一地方。在纯解码器架构（GPT）中，交叉注意力被省略，你只有掩码自注意力 + FFN。

### Pre-Norm 与 Post-Norm

原始论文：`x + sublayer(LN(x))` 对比 `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有仔细的预热很难深层训练。Pre-norm（`LN` 在子层*前*）是 2026 年的默认做法：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化块

Vaswani 2017 推出了 LayerNorm + ReLU。现代栈把两者都替换了。生产块的实际样子：

| 组件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量匹配） |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 全量 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少了一次减法），节省了计算，并且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中始终优于 ReLU/GELU FFN，ppl 大约降低 0.5。

### 参数量

对于一块，`d_model = d`，FFN 扩展比 `r`：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略

当 `d = 4096, r = 2.6, layers = 32`（大致相当于 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数每层 × 32 ≈ 7B`（加上嵌入和输出头）。与公布的参数量一致。

## 构建它

### 步骤 1：构建块

使用课程 03 中的小 `Matrix` 类（已复制到本文件以保持独立）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。无均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

参见 `code/main.py` 中的完整连接。

### 步骤 2：连接一个 2 层编码器和一个 2 层解码器

堆叠它们。将编码器输出传递给每个解码器交叉注意力。在输出投影前添加一个最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在玩具示例上前向传播

输入一个 6 个词元的源序列和一个 5 个词元的目标序列。验证输出形状为 `(5, vocab)`。不进行训练——本课关注架构，而非损失。

### 步骤 4：替换为 RMSNorm + SwiGLU

将 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认形状仍匹配。这是通过一次函数替换完成的 2026 年现代化。

## 使用它

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年生产代码会自己实现块，因为：

- Flash Attention 在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认值。

HF `transformers` 有清晰的参考块，你应该阅读：`modeling_llama.py` 是规范的 2026 年解码器块。大约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器——何时选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器赢得了语言领域，因为它扩展最干净，同时处理理解和生成。当输入具有清晰的"源序列"身份时（翻译、语音识别、结构化任务），编码器-解码器仍然是最佳选择。

## 交付它

参见 `outputs/skill-transformer-block-reviewer.md`。该技能检查新的 transformer 块实现是否符合 2026 年默认值，并标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 在你的 `encoder_block` 中计算参数数量，参数为 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`。通过实现该块并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，并在随机输入上测量 12 层堆叠后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应保持有界。
3. **困难。** 在玩具复制任务（复制 `x` 的反向序列）上实现一个 4 层编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失会下降吗？

## 关键术语

| 术语 | 大家说的 | 实际含义 |
|------|----------|----------|
| 块 | "一个Transformer层" | 由归一化、注意力、归一化、FFN 堆叠而成，包裹在残差连接中。 |
| 残差 | "跳跃连接" | `x + f(x)` 输出；使梯度能够流过深层。 |
| Pre-norm | "在前归一化，不在后" | 现代：`x + sublayer(LN(x))`。无需预热技巧即可训练更深。 |
| RMSNorm | "没有均值的LayerNorm" | 除以 RMS；少一次操作，经验稳定性相同。 |
| SwiGLU | "人人都换用的FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型 ppl 上优于 ReLU/GELU。 |
| 交叉注意力 | "解码器如何看到编码器" | 查询来自解码器，键/值来自编码器输出的多头注意力。 |
| FFN 扩展比 | "中间MLP有多宽" | 隐藏大小与 d_model 的比值，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 | "去掉 +b 项" | 现代栈在线性层中省略偏置；稍优的 ppl，更小的模型。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么深层 pre-norm 胜过 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范的 2026 年仅解码器块。
