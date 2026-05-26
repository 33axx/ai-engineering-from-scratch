# 多头注意力

> 一个注意力头一次只能学习一种关系。八个头就能学习八种。头是自由的。多用几个。

**类型：** 构建  
**语言：** Python  
**先修要求：** 阶段 7 · 02（从零实现自注意力）  
**时间：** 约 75 分钟  

## 问题

单个自注意力头只计算一个注意力矩阵。这个矩阵只能捕捉一种关系——通常是那种能最小化训练信号损失的。如果数据中混杂着主谓一致、指代消解、长程篇章依赖以及句法组块，单个头会把它们压进一个 softmax 分布里，结果丢失一半信号。

2017 年 Vaswani 论文给出的解决办法：并行运行多个注意力函数，每个有自己的 Q、K、V 投影，然后将输出拼接起来。每个头工作在维度为 `d_model / n_heads` 的更小子空间中。总参数量保持不变，但表达能力大大提升。

多头注意力是 2026 年所有 Transformer 的默认配置。唯有的争论在于使用*多少个*头，以及键和值是否共享投影（分组查询注意力、多查询注意力、多头潜注意力）。

## 概念

![多头注意力：分割、注意力计算、拼接](../assets/multi-head-attention.svg)

**分割。** 取形状为 `(N, d_model)` 的 X。分别投影到 Q、K、V，每个形状为 `(N, d_model)`。重新变形为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行注意力计算。** 在每个头内运行缩放点积注意力。每个头产生 `(N, d_head)`。这些头操作在嵌入的不同子空间中，在注意力计算过程中彼此不通信。

**拼接并投影。** 将头堆叠回 `(N, d_model)`，乘以一个形状为 `(d_model, d_model)` 的学习到的输出矩阵 W_o。W_o 就是头之间混合信息的地方。

**为什么有效。** 每个头可以专注于自己的任务，而不必与其他头争夺表示预算。2019–2024 年的探测研究显示了不同的头角色：位置头、关注前一个 token 的头、复制头、命名实体头、归纳头（这是上下文学习的基础）。

**2026 年各变体的谱系：**

| 变体 | Q 头数量 | K/V 头数量 | 使用案例 |
|------|----------|------------|---------|
| 多头注意力 (MHA) | N | N | GPT-2, BERT, T5 |
| 多查询注意力 (MQA) | N | 1 | PaLM, Falcon |
| 分组查询注意力 (GQA) | N | G (例如 N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜注意力 (MLA) | N | 压缩为低秩 | DeepSeek-V2, V3 |

GQA 是现代默认选项，因为它将 KV 缓存内存削减了 `N/G` 倍，同时几乎保持完全相同的质量。MLA 更进一步，将 K/V 压缩到潜空间，然后在计算时再投影回来——这会消耗更多 FLOPs，但能节省更多内存。

## 动手实践

### 第 1 步：将头从已有的单头注意力中分离出来

获取第 02 课中的 `SelfAttention`，并用一对分割/拼接来包裹它。参见 `code/main.py` 中的 numpy 实现，逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 和一次 transpose。没有循环。这正是 PyTorch 中 `nn.MultiheadAttention` 底层做的事情。

### 第 2 步：在每个头上运行缩放点积注意力

每个头获得自己的 Q、K、V 切片。注意力变成批处理的矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在实际硬件上，`Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到单个批处理矩阵乘法，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头数是零成本的。

### 第 3 步：分组查询注意力变体

只有键和值投影发生变化。Q 有 `n_heads` 个组；K 和 V 有 `n_kv_heads < n_heads` 个组，并通过重复来匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时这样可以节省内存，因为 KV 缓存中只保存了 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个查询头和 8 个 KV 头——缓存缩小了 8 倍。

### 第 4 步：探测每个头学到了什么

在包含 4 个头的一条短句子上运行 MHA。对每个头，打印尺寸为 `(N, N)` 的注意力矩阵。你会看到不同的头即使随机初始化也会挑选出不同的结构——这有一部分是信号的作用，也有一部分是子空间中的旋转对称性。

## 使用方式

在 PyTorch 中，一行代码版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**需要多少个头？** 来自 2026 年生产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型 (~125M) | 768 | 12 | 64 |
| 基础 (~350M) | 1024 | 16 | 64 |
| 大型 (~1B) | 2048 | 16 | 128 |
| 前沿 (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它代表了一个头能“看到”多少信息的基本单元。低于 32 时，头会开始与缩放因子 `sqrt(d_head)` 对抗；高于 256 时，你会失去“许多小型专家”的优势。

## 交付产物

参见 `outputs/skill-mha-configurator.md`。这个技能文档会针对一个新的 Transformer，根据参数量预算、序列长度和部署目标，推荐头数量、KV 头数量以及投影策略。

## 练习

1. **简单级别。** 获取 `code/main.py` 中的 MHA，在固定 `d_model=64` 的情况下将 `n_heads` 从 1 改为 16。在合成复制任务上绘制一个小型单层模型的损失。更多头是帮助、平台还是有害？
2. **中等级别。** 实现 MQA（一个 KV 头被所有查询头共享）。测量参数量相比完整 MHA 减少了多少。计算在推理时对于 N=2048，KV 缓存大小缩小了多少。
3. **困难级别。** 实现一个小规模的多头潜注意力变体：将 K、V 压缩为秩-`r` 的潜变量，将潜变量存入 KV 缓存，在注意力计算时解压。在验证困惑度保持在一个比特以内的前提下，找到使得缓存内存低于完整 MHA 的 1/8 的 `r` 值。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Head | “一个注意力电路” | 维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，带有自己的注意力矩阵。 |
| d_head | “头维度” | 每个头的隐藏宽度；生产中几乎总是 64 或 128。 |
| Split / combine | “变形技巧” | 注意力计算前后 `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape+transpose。 |
| W_o | “输出投影” | 拼接头之后应用的 `(d_model, d_model)` 矩阵；头之间混合信息的场所。 |
| MQA | “单 KV 头” | 多查询注意力：单一的共享 K/V 投影。KV 缓存最小，但质量有所损失。 |
| GQA | “Llama 2 以来的默认选项” | 分组查询注意力，`n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA | “DeepSeek 的窍门” | 多头潜注意力：K/V 压缩为低秩潜变量，在注意力计算时解压缩。 |
| Induction head | “上下文学习背后的电路” | 一对头，检测到之前的出现并复制其后跟随的内容。 |

## 扩展阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 及其在缓存内存上优于 MHA/GQA 的原因。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 关于头实际作用的机制性探究。
