# 位置编码 — Sinusoidal、RoPE、ALiBi

> 注意力机制是排列不变的。“猫坐在垫子上”和“垫子上坐猫在”在没有位置信号的情况下会产生相同的输出。有三种算法可以解决这一问题——每种算法对“位置”的含义有不同的假设。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 7·02（自注意力），阶段 7·03（多头注意力）  
**时间：** 约 45 分钟

## 问题

缩放点积注意力是顺序盲的。注意力矩阵 `softmax(Q K^T / √d) V` 是根据成对相似度计算得到的。打乱 `X` 的行，输出的行也会以相同方式被打乱。注意力内部没有任何机制关心位置。

在词袋模型中这并非缺陷。但对于语言、代码、音频、视频——任何顺序承载意义的事物——这是致命的。

解决方案是以某种方式将位置注入嵌入中。有三种时期的答案：

1. **绝对正弦（Absolute sinusoidal）**（Vaswani 2017）。将 `sin/cos` 位置信息加到嵌入上。简单、无需学习，但在训练长度之外外推能力差。
2. **RoPE — 旋转位置嵌入**（Su 2021）。将 Q 和 K 向量旋转一个与位置成比例的角度。直接将*相对*位置编码在点积中。2026 年主导地位。
3. **ALiBi — 具有线性偏置的注意力**（Press 2022）。完全跳过嵌入；向注意力分数添加一个基于距离的逐头线性惩罚。出色的长度外推能力。

截至 2026 年，几乎所有前沿开放模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦已成为历史。

## 概念

![绝对正弦 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦

预计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前执行 `X' = X + PE[:N]`。每个维度是一个不同频率的正弦波。模型学习从相位模式中读取位置。超出 `max_len` 时失败：当模型只见过位置 0–2047，无法告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（而不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的键应用相同的旋转。点积 `q'_m · k'_n` 将仅成为 `(m - n)` 的函数。即：**注意力分数仅取决于相对距离**，尽管旋转是基于绝对位置的。巧妙的技巧。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长的上下文。Llama 3 通过此方法从 8K 扩展到了 128K 上下文。

### ALiBi

跳过嵌入技巧。直接对注意力分数施加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是特定于头的斜率（例如 `1 / 2^(8·h/H)`）。较近的 token 获得提升；较远的 token 受到惩罚。没有训练时的成本。论文显示长度外推能力优于正弦，在原始训练长度上与 RoPE 相当。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用方 |
|---------|---------------|---------------|---------|
| 绝对正弦 | 差 | 免费 | 原始 Transformer，早期 BERT |
| 学习型绝对位置 | 无 | 极小 | GPT-2，GPT-3 |
| RoPE | 好（带缩放） | 免费 | Llama 2/3/4，Qwen 2/3，Mistral，DeepSeek-V3，Kimi |
| RoPE + YaRN | 极好 | 微调阶段 | Qwen2-1M，Llama 3.1 128K |
| ALiBi | 极好 | 免费 | BLOOM，MPT，Baichuan |

RoPE 胜出，因为它嵌入注意力而不改变架构，编码了相对位置，并且其 `base` 超参数为长上下文微调提供了一个干净的调节旋钮。

## 构建它

### 第 1 步：正弦编码

参见 `code/main.py`。一个 4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其添加到嵌入矩阵中。

### 第 2 步：将 RoPE 应用于 Q、K

RoPE 在 Q 和 K 上原地操作。对于每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：将相同的函数应用于位置 `m` 处的 Q 和位置 `n` 处的 K。它们的点积在每个坐标对上会得到一个 `cos((m-n)·θ_i)` 因子。注意力由此免费学习相对位置。

### 第 3 步：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 添加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵中，然后执行 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

选取两个随机向量 `a, b`。用 `(pos_a, pos_b)` 旋转。然后用 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差范围内匹配。这个性质是整个 RoPE 的要点——它对绝对偏移不变，仅相对间距重要。

## 使用它

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具函数。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware 插值。** 当从 4K 扩展到 16K+ 时，将 `base` 按比例缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文中保持注意力熵。Llama 3.1 128K 使用了它。
- **LongRoPE。** 微软 2024 年的方法，使用进化搜索来为每个维度选择缩放因子。Phi-3-Long 使用了它。
- **位置插值 + 微调。** 简单地将位置按扩展因子缩小，并用 1–5B token 微调。效果出奇地好。

## 交付它

参见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热力图。确认“随着维度索引增加，条纹变宽”的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。训练一个小型 LM，序列长度为 256，然后在有缩放和无缩放的情况下测试长度为 1024 的性能。测量困惑度。
3. **困难。** 在同一注意力模块中实现 ALiBi 和 RoPE。训练一个 4 层 Transformer 执行序列长度为 512 的复制任务。在测试时外推至 2048。比较性能下降情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 位置编码 | “告诉注意力关于顺序的信息” | 添加到嵌入或注意力中的任何编码位置的信号。 |
| 正弦 | “原始的那个” | `sin/cos` 以几何频率添加到嵌入中；不具外推能力。 |
| RoPE | “旋转嵌入” | 通过依赖于位置的角旋转 Q、K；点积编码了相对距离。 |
| ALiBi | “线性偏置技巧” | 向注意力分数添加 `-m·|i-j|`；不需要嵌入，外推能力强。 |
| base | “RoPE 的调节旋钮” | RoPE 中的频率缩放因子；增加它以在推理时扩展上下文。 |
| NTK-aware | “一种 RoPE 缩放技巧” | 重新缩放 `base`，使得高频率维度在上下文扩展时不被压缩。 |
| YaRN | “花哨的那个” | 逐维插值+外推，保持注意力熵。 |
| 外推 | “在训练长度之外工作” | 位置方案能否在训练中看到的 `max_len` 之外产生正确输出。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软的方法，被 Phi-3-Long 使用并在“使用它”部分中引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每个 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）的生产级实现。
