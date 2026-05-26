# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步）。VAR 在尺度上迭代采样——它先预测一个 1x1 token，然后是 2x2、4x4，直到最终分辨率，每个尺度以之前尺度为条件。2024 年的论文表明，VAR 在图像生成上匹配了 GPT 风格的缩放定律，并在相同计算预算下超越了 DiT。本课将构建其核心机制。

**类型：** 构建  
**语言：** Python（使用 PyTorch）  
**前置知识：** 阶段 7 第 03 课（多头注意力）、阶段 8 第 06 课（DDPM）  
**时间：** 约 90 分钟

## 问题所在

自回归生成在语言建模中占据主导地位，因为它可预测地扩展：更多计算、更多参数、更低困惑度、更好输出。在 2024 年之前，图像生成的主要自回归尝试有两种：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 编码上逐 token）。

两者都面临生成顺序问题。像素和 token 排列在二维网格中，但自回归模型必须以一维扫描线顺序访问它们。早期的角落像素完全不知道图像最终会变成什么。生成质量随计算量缩放的效果不如 GPT-on-text，并且从未在同等计算量下达到扩散模型的质量。

VAR 通过改变生成的内容解决了生成顺序问题。它不是逐个预测图像 token 的空间位置，而是以递增的分辨率预测整个图像。第一步：预测一个 1x1 token（图像整体“摘要”）。第二步：预测一个 2x2 token 网格（粗特征）。第三步：预测一个 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度关注所有之前的尺度（在“尺度顺序”上因果）并在自身尺度内并行。顺序问题消失了：尺度 k 的整个图像在一次 Transformer 前向传播中产生。

## 概念

### VQ-VAE 多尺度 Tokenizer

VAR 需要一个**多尺度离散 tokenizer**。对于图像 x，它产生一系列逐渐更高分辨率的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小 4096-16384）。每个尺度的 token 化不是独立的——它的训练方式是使得在每个尺度上残差求和可重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是**残差 VQ** 的变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器取所有尺度嵌入的和并生成图像。

多尺度 VQ tokenizer 训练一次（类似 VQGAN）然后冻结。所有生成工作由位于其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 Transformer，它看到所有之前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置编码同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token，以及尺度 k 自身中按照某种尺度内顺序更早的 token（VAR 使用固定的位置注意力，没有尺度内因果性——一个尺度内的所有位置被并行预测）。

训练损失：在每个尺度 k，给定所有之前尺度的 token 预测 token z_k。对离散 VQ 编码使用交叉熵损失。结构与 GPT 相同，只是“序列”现在是尺度结构的。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K=10 个尺度，生成需要 10 次 Transformer 前向传播。每次前向传播并行生成其整个尺度——尺度内没有逐 token 自回归。对于 256x256 图像，这大约是 10 次前向传播，而 DiT 需要 28-50 次。

### 为什么下一尺度优于下一 Token

三个结构性优势：
1. **从粗到细符合自然图像统计。** 人类视觉感知和图像数据集都表现出尺度依赖的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 风格的 token AR 不同，VAR 一步生成一个尺度内的所有 token。有效生成长度是对数尺度而非线性。
3. **没有生成顺序偏差。** 尺度 k 的 token 看到尺度 k-1 的全部；不存在“左边”或“上方”偏差，这种偏差会迫使早期 token 在后期上下文可用之前做出决策。

### 缩放定律

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——就像 GPT 在困惑度上一样。参数或计算量加倍能可靠地使误差减半。这是第一个像语言模型一样清晰展现这种缩放行为的图像生成模型。其结果是 VAR 尺度的预测变得可由计算量预测，而不是针对每种架构凭经验猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为一系列更容易的子问题。

- 扩散：逐渐添加噪声，学习逆转一步。
- VAR：逐渐增加分辨率，学习预测下一个尺度。

它们是问题空间的不同轴。两者都产生易于处理的条件分布。经验上，VAR 推理更快（更少的前向传播，尺度内全部并行），并在类条件 ImageNet 上匹配或超越 DiT。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

## 构建环节

在 `code/main.py` 中，你将：
1. 在合成“图像”数据（二维高斯环）上构建一个微型**多尺度 VQ tokenizer**。
2. 训练一个**VAR 风格 Transformer** 来下一尺度预测 token。
3. 通过调用 Transformer 4 次（4 个尺度）并解码来采样。
4. 验证尺度顺序训练使得生成在尺度内并行。

这是一个玩具实现。重点是看到尺度结构的注意力掩码以及尺度内并行生成的实际工作。

## 产出环节

本课生成 `outputs/skill-var-tokenizer-designer.md`——一个用于设计多尺度 tokenizer 的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 使用 4、6、8、10 个尺度训练 VAR。衡量重建质量与自回归前向传播次数。更多尺度 = 更细的残差 = 更好的质量但更多前向传播。

2. **码本大小。** 使用码本大小 512、4096、16384 训练 tokenizer。更大的码本提供更好的重建但更难的预测。找到拐点。

3. **尺度内并行检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置但不关注尺度内位置？验证掩码实现。

4. **VAR 与 DiT 缩放。** 对于相同的 ImageNet 类条件任务，在匹配的参数预算下（例如 33M、130M、458M）训练 VAR 和 DiT。绘制 FID 与计算量的关系图。VAR 应在每种规模下领先 DiT——在小规模复现论文结果。

5. **文本条件。** 通过 adaLN 将文本嵌入（CLIP 池化）作为额外条件输入扩展 VAR。这是 HART 方法。在文本对齐采样上 FID 改善多少？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| VAR | "视觉自回归" | 通过在 VQ token 网格金字塔上进行下一尺度预测来生成图像 |
| 下一尺度预测 | "先预测粗的，再预测细的" | 模型预测分辨率逐渐增大的 token 网格，以所有之前的尺度为条件 |
| 多尺度 VQ tokenizer | "残差 VQ" | VQ-VAE 产生 K 个分辨率递增的 token 网格，解码器对所有尺度求和 |
| 尺度 k | "金字塔层级 k" | K 个分辨率层级之一，从 k=1 时的 1x1 到 k=K 时的 (H/p)x(W/p) |
| 尺度内并行 | "每尺度一次前向传播" | 一个尺度内的所有 token 在一次 Transformer 前向传播中预测，而非自回归 |
| 跨尺度因果 | "尺度顺序注意力" | 尺度 k 的 token 可以关注尺度 1..k 的所有 token，但不能关注尺度 k+1..K |
| 残差 VQ | "加性 token 化" | 每个尺度的 token 编码较低尺度留下的残差；解码器对所有尺度嵌入求和 |
| VAR 缩放定律 | "图像 GPT 缩放" | FID 在计算量上遵循可预测的幂律，如同语言模型的困惑度 |
| HART | "混合 VAR + 文本" | 文本条件 VAR 变体，将 MaskGIT 风格的迭代解码与 VAR 的尺度结构结合 |
| 尺度位置嵌入 | "(尺度，行，列) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，标准参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 的多尺度 tokenizer 所扩展的 tokenizer 家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 token 化的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
