# 从零构建 Transformer——顶点项目

> 十三节课。一个模型。不走捷径。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 7 · 第 01 到 13 课。不可跳过。
**预计耗时：** ~120 分钟

## 问题

你已读遍每篇论文。你实现了注意力、多头拆分、位置编码、编码器与解码器模块、BERT 与 GPT 的损失函数、MoE、KV 缓存。现在，让它们在一个真实任务上协同工作。

顶点项目：端到端训练一个仅含解码器的小型 Transformer，完成字符级语言建模任务。它读取莎士比亚文本。它生成新的莎士比亚文本。它足够小，可以在笔记本上 10 分钟内完成训练。它也足够正确，只要换成更大的数据集并延长训练时间，就能得到一个真正的语言模型。

这是本课程的“nanoGPT”。它并非原创——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的标准实现。我们借鉴其架构，并根据已学内容重新调整。

## 概念

![从零构建 Transformer 模块示意图](../assets/capstone.svg)

架构，附注说明：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们提供的内容

- `GPTConfig` —— 一处配置所有超参数。
- `MultiHeadAttention` —— 因果、批量化，可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代 FFN。
- `Block` —— 预归一化、带残差连接的注意力 + FFN。
- `GPT` —— 嵌入层、堆叠模块、语言模型头、`generate()` 方法。
- 训练循环，包含 AdamW、余弦学习率、梯度裁剪。
- 基于莎士比亚文本的字符级分词器。

### 我们不提供的内容

- RoPE —— 已在第 04 课概念上实现。这里为了简单使用可学习的位置嵌入。练习要求你替换为 RoPE。
- 生成过程中的 KV 缓存 —— 每个生成步骤重新计算整个前缀的注意力。较慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention —— PyTorch 2.0+ 会在输入匹配时自动分派；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个模块只有一个 FFN。你在第 11 课中看到了 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2000 步：

- 训练损失从 ~4.2（随机）收敛到 ~1.5，大约 6 分钟。
- 采样输出具有莎士比亚风格：古词、换行、出现“ROMEO:”等专有名词。
- 验证损失（保留最后的 10% 文本）与训练损失紧密跟随；在此规模/预算下未出现过拟合。

## 构建

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。该脚本处理：

- 若缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 的训练/验证集划分。
- 训练循环，支持硬件上使用 bf16 自动混合精度。
- 训练完成后进行采样。

### 第一步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。词汇量极小。适配 4 字节的 vocab_size。没有 BPE，没有分词器烦恼。

### 第二步：模型

参见 `code/main.py`。模块是第 05 课的标准形式——预归一化、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数数量：约 800K。

### 第三步：训练循环

获取一个随机批次，窗口长度为 256 的 token。前向传播。移位一位的交叉熵。反向传播。AdamW 步进。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样

给定一个提示，重复前向传播，从 top-p logits 采样，追加，继续。500 个 token 后停止。

### 第五步：读取输出

2000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚，但具有莎士比亚的形状。约 800K 参数在笔记本上 6 分钟的明确胜利。

## 运用

这个顶点项目是一个参考架构。三个扩展方向，使其成为真正可用的东西：

1. **更换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇量从 65 跳到约 50,000。模型容量需要相应扩大。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上训练 100 亿 token 大约需要 24 小时（125M 参数 GPT）。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习会逐步引导你完成。

最终得到一个 125M 参数的 GPT，能够生成流利的英语。不是前沿模型，但同样的代码路径——只是更大——正是 Karpathy、EleutherAI 和 Allen 研究所在 2026 年用于训练研究检查点的方法。

## 交付

参见 `outputs/skill-transformer-review.md`。该技能评审了从零构建 Transformer 的实现，涵盖了之前全部 13 课的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练后模型最后一步的验证损失低于 2.0。将 `max_steps` 从 2000 改为 5000——验证损失是否持续改善？
2. **中等。** 用 RoPE 替换可学习的位置嵌入。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。分别生成 500 个 token（带缓存和不带缓存）。在笔记本上，实际时间应提升 5-20 倍。
4. **困难。** 为模型添加第二个头，用于预测下一个且再下一个 token（MTP——DeepSeek-V3 的多 token 预测）。联合训练。是否有帮助？
5. **困难。** 将每个模块的单个 FFN 替换为 4 个专家的 MoE。路由器 + top-2 路由。在匹配活跃参数的情况下观察验证损失的变化。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|---------|
| nanoGPT | “Karpathy 的教程仓库” | 最小化的仅解码器 Transformer 训练代码，约 300 行；标准参考实现。 |
| tinyshakespeare | “标准玩具语料库” | 约 1.1 MB 文本；自 2015 年以来每个字符级 LM 教程都在使用它。 |
| 绑定嵌入（Tied embeddings） | “共享输入/输出矩阵” | 语言模型头权重 = token 嵌入矩阵的转置；节省参数，提高质量。 |
| bf16 自动混合精度 | “训练精度技巧” | 前向/反向在 bf16 中运行，优化器状态保持在 fp32；自 2021 年起成为标准。 |
| 梯度裁剪 | “阻止尖峰” | 将全局梯度范数限制在 1.0；防止训练崩溃。 |
| 余弦学习率调度 | “2020 年后的默认方案” | 学习率线性上升（预热），然后按余弦形状衰减至峰值的 10%。 |
| MFU | “模型 FLOP 利用率” | 实际 FLOPs / 理论峰值；2026 年密集模型 40%、MoE 30% 属于优秀水平。 |
| 验证损失 | “保留损失” | 模型从未见过的数据的交叉熵；过拟合检测器。 |

## 进一步阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的注释实现。
