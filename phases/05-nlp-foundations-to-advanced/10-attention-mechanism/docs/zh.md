# 注意力机制 —— 突破

> 解码器不再眯着眼看压缩摘要，而是开始审视整个源文本。此后的一切都是注意力加上工程。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 09（序列到序列模型）
**时间：** ~45 分钟

## 问题

第 09 课以一次经过衡量的失败告终。在一个玩具复制任务上训练的 GRU 编码器-解码器，长度 5 时准确率为 89%，到了长度 80 则接近随机水平。原因是结构性的，而非训练 bug：编码器收集的每一点信息都必须塞进一个固定大小的隐藏状态，而解码器再也看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了三行修复。不再只给解码器最后一个编码器状态，而是保留每个编码器状态。在每个解码步骤，计算编码器状态的加权平均值，其中的权重表示“解码器现在需要多关注编码器位置 `i`？”这个加权平均值就是上下文，并且它在每个解码步骤都会变化。

这就是全部想法。Transformer 将其扩展。自注意力将其应用到单个序列上。多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈，一旦你掌握了它，转向 Transformer 就只是工程问题，而非概念问题。

## 概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码步骤 `t`：

1. 使用先前的解码器隐藏状态 `s_{t-1}` 作为 **查询**。
2. 对每个编码器隐藏状态 `h_1, ..., h_T` 打分。每个编码器位置得到一个标量分数。
3. 对分数进行 softmax 得到注意力权重 `α_{t,1}, ..., α_{t,T}`，总和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均值。
5. 解码器接收 `c_t` 加上前一个输出 token，产生下一个 token。

加权平均值是关键。当解码器需要将 "Je" 翻译成 "I" 时，它对 "Je" 上的编码器状态赋予高权重，其他则较低。当需要 "not" 时，它对 "pas" 赋予高权重。上下文向量每一步都会重新塑造。

## 形状（最常坑人的地方）

这是每个注意力实现第一次都会出错的地方。请仔细阅读。

| 事物 | 形状 | 说明 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后得到 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`, `h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
- 它们在 tanh 内的和形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 的内积降为标量。**这就是 `v_α` 的作用。** 这不是魔法。它是将注意力维度的向量投影为标量分数的投影。

**Luong（乘性）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。严格约束。如果编码器是双向的，跳过这个。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 形状为 `(d_s, d_h)`。去除了等维约束。
- `concat`：本质上是 Bahdanau 形式。很少使用，因为前两种更廉价。

**一个值得指出的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词 *之前* 的解码器状态）。Luong 使用 `s_t`（生成 *之后* 的状态）。混淆它们会产生微妙错误的梯度，极难调试。选一篇论文，坚持其约定。

## 构建它

### 第一步：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查你的形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)` 并广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。搞定。

### 第二步：Luong 点积和通用形式

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每段三行。这就是 Luong 论文受欢迎的原因：在大多数任务上精度相同，代码少得多。

### 第三步：一个手算数值示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布集中在位置 0。如果解码器状态变为与最后一个对齐，注意力则移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后移动解码器状态使其接近第三个编码器状态，观察权重转移。就是这样。注意力就是显式的对齐。

### 第四步：为什么这是通往 Transformer 的桥梁

将上面的语言转换为 Q/K/V：

- **查询** = 解码器状态 `s_{t-1}`
- **键** = 编码器状态（我们根据它打分）
- **值** = 编码器状态（我们加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力将它们分开：你可以用不同的学习投影针对序列自身查询，K 和 V 学习不同的投影。多头注意力用不同的学习投影并行运行。Transformer 将整个阶段堆叠多次并去掉 RNN。

数学是一样的。形状是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要在于记号。

## 使用它

PyTorch 和 TensorFlow 直接提供了注意力模块。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

那是一个 Transformer 注意力层。查询批次 5 个位置，键/值批次 10 个位置，每个 128 维，8 个头。`output` 是新的上下文增强的查询。`weights` 是 5x10 的对齐矩阵，你可以可视化。

### 经典注意力何时仍然重要

- 教学：单头、单层、基于 RNN 的版本使每个概念可见。
- 设备上的序列任务，Transformer 不适用。
- 任何 2014-2017 年的论文。不了解 Bahdanau 的约定，你会误读。
- 机器翻译中的细粒度对齐分析。原始注意力权重即使在 Transformer 模型上也是一种可解释性工具，阅读它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可解释。它们是跨位置总和为 1 的权重；你可以画出它们；高值表示“看过这里”。审稿人喜欢它们。

但它们并不像看起来那么可解释。Jain 和 Wallace（2019）表明，在某些任务中，注意力分布可以被置换并替换为任意替代值，而不改变模型预测。没有消融或反事实检验，永远不要将注意力权重作为推理的证据来报告。

## 交付它

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 softmax 掩码，使编码器中的填充 token 获得零注意力权重。在包含可变长度序列的批次上测试。
2. **中等。** 在 Luong `general` 形式中添加多头注意力。将 `d_h` 拆分为 `n_heads` 组，每头运行注意力，拼接。验证单头情况与之前的实现一致。
3. **困难。** 在第 09 课的玩具复制任务上训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率 vs 序列长度图。与无注意力的基线比较。你应该看到差距随着长度增加而扩大，证实注意力解除了瓶颈。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 注意力 | 看东西 | 值序列的加权平均，权重由查询-键相似度计算得出。 |
| 查询、键、值 | QKV | 三个投影：Q 提问，K 是要匹配的内容，V 是要返回的内容。 |
| 加性注意力 | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| 乘性注意力 | Luong 点积 / general | 分数为 `q^T k` 或 `q^T W k`。更廉价，在大多数任务上精度相同。 |
| 对齐矩阵 | 漂亮的图 | 形状为 `(T_dec, T_enc)` 的注意力权重网格。读取它可以看到模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性注意事项。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 实践。
