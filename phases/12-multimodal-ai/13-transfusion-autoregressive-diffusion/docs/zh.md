# Transfusion：一个Transformer中的自回归文本+扩散图像

> Chameleon 和 Emu3 将一切押注在离散 token 上。它们有效，但量化瓶颈是可见的——图像质量在连续空间扩散模型以下趋于平缓。Transfusion（Meta，Zhou 等人，2024 年 8 月）采取了相反的赌注：保持图像连续，完全摒弃 VQ-VAE，并通过两种损失训练一个 Transformer。文本 token 采用下一个 token 预测。图像 patch 采用流匹配/扩散损失。两个目标优化相同的权重。Stable Diffusion 3（MMDiT）的架构是其近亲。本课阅读 Transfusion 论文，构建一个玩具双损失训练器，并追溯让一个 Transformer 同时完成两项任务的注意力掩码。

**类型：** 构建
**语言：** Python（stdlib，在 MNIST 规模玩具上的双损失训练器）
**先决条件：** 阶段 12 · 11（Chameleon），阶段 8（生成式 AI）
**时间：** 约 180 分钟

## 学习目标

- 构建一个 Transformer，在一个主干网络上运行两种损失（文本 token 的 NTP，图像 patch 的扩散 MSE）。
- 解释为什么图像 patch 间的双向注意力加上文本 token 上的因果注意力是正确的掩码选择。
- 比较 Transfusion 风格（连续图像，扩散损失）与 Chameleon 风格（离散图像，NTP）在计算量、质量和代码复杂度上的差异。
- 说出 MMDiT 的贡献：每个模块中的模态特定权重，残差流上的联合注意力。

## 问题

离散与连续图像 token 的争论比 LLM 更古老。连续表示（原始像素、VAE 潜变量）保留了细节。离散 token（VQ 索引）适合 Transformer 的原生词汇表，但在量化步骤中丢失了细节。

Chameleon / Emu3 选择了离散：一种损失，一种架构，但图像质量受限于 tokenizer 质量。

扩散模型选择了连续：卓越的图像质量，但模型与 LLM 分离，噪声调度工程复杂，且与文本生成没有干净的整合。

Transfusion 问：能否两者兼得？保持图像连续，仍然训练一个模型，使用两种损失缝合到一个梯度步骤中。

## 概念

### 双损失架构

一个单一的仅解码器 Transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词汇表）。
- 图像 patch（连续，16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入相同）。
- `<image>` 和 `</image>` 标签，标记连续 patch 所在位置。

前向传播运行一次。损失根据每个 token 选择两个头之一：

- 对于文本 token：在词汇表 logits 头上的标准交叉熵。
- 对于图像 patch：连续 patch 上的扩散损失——预测添加到每个 patch 上的噪声。

梯度流过共享的 Transformer 主体。两种损失同时改善共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——你不能让文本 token 关注未来的文本，否则教师强制会失效。然而，图像 patch 表示一次快照；它们应在同一图像块内相互进行双向关注。

掩码：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在训练和推理时实现为块三角掩码。

### Transformer 内部的扩散损失

扩散损失是标准的：向图像 patch 添加噪声，要求模型预测噪声（或等效地预测干净 patch）。Transfusion 版本使用流匹配——预测从噪声到干净数据的速度场。

训练期间：
1. 对于每个图像 patch x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（用于流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与来自同一序列的文本 NTP 损失一起反向传播。

推理时，生成过程为：
- 文本 token：标准自回归采样。
- 图像 patch：以先前文本 token 为条件的扩散采样循环（典型 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）在差不多同一时间推出了 MMDiT（多模态扩散 Transformer）。这些架构是兄弟关系。

MMDiT 的关键不同之处：

- 每个模块的模态特定权重。每个 Transformer 模块对文本 token 和图像 patch 有单独的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他一切都按模态特定。
- 整流流训练。一种特定的流匹配变体，具有已知的采样方法和比 DDPM 更简单的数学。
- 规模。MMDiT 是 SD3（2B 和 8B 参数变体）的主干。Transfusion 论文扩展到 7B。

两者都收敛于同一个核心思想：一个 Transformer 在文本上运行 NTP，在连续图像表示上运行扩散。

### 为何这优于 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数时，在 FID 上比同等规模的 Chameleon 风格模型好 3-5 个点。
- 无需训练 tokenizer——图像编码器更简单（线性投影到隐藏，与 ViT 的输入层相同）。
- 推理可以并行化图像 patch 去噪，这与自回归图像 token 不同。

缺点：Transfusion 是一个双损失模型，使得训练动态更棘手。损失权重需要调整。NTP 和扩散之间的调度不匹配可能导致一个头占主导。

### 后续发展

Janus-Pro（第 12.15 课）通过将视觉编码器解耦为理解和生成——一个用 SigLIP，另一个用 VQ——同时共享 Transformer 主体，改进了 Transfusion 的想法。Show-o（第 12.14 课）将扩散替换为离散扩散（掩码预测）。在 Transfusion 之后，统一生成家族迅速分支。

2026 年生产级能输出图像的 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎确定使用了这个家族的某个后代。细节是专有的。

## 使用

`code/main.py` 在一个类似 MNIST 的小型玩具问题上构建了一个玩具 Transfusion：

- 文本标题是描述数字（0-9）的短整数序列。
- 图像是 4x4 字节网格。
- 一对共享权重的线性投影作为 Transformer 的替身；文本上的 NTP 损失，噪声 patch 上的 MSE 损失。
- 训练循环交替两种损失，注意力掩码是显式的。
- 生成过程在一次前向传播中产生文本标题和 4x4 图像。

Transformer 是玩具。双损失管道、注意力掩码构建和推理循环是真正的产物。

## 交付

本课产生 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本+图像、文本+音频、文本+视频），它设计双损失调度（损失权重、掩码形状、共享与模态特定模块）并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像扩散损失的大小约为文本 NTP 损失的 10 倍。什么损失权重可以平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。将每个条目标记为 0 或 1。

3. MMDiT 具有模态特定的 QKV 权重。与 Transfusion 的完全共享 Transformer 相比，这增加了多少参数开销？在 7B 参数下，值得吗？

4. 生成：给定一个文本提示，模型运行 NTP 50 个 token，然后遇到 `<image>`，然后对 256 个 patch 运行扩散 20 个去噪步骤。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流，并解释为什么它在推理步骤上比 DDPM 收敛得更少。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 双损失训练 | "NTP + 扩散" | 一个单一的 Transformer 在同一个梯度步骤中同时优化文本 token 上的交叉熵和连续图像 patch 上的 MSE |
| 流匹配 | "整流流" | 扩散变体，预测从噪声到干净数据的速度场；数学比 DDPM 更简单 |
| MMDiT | "多模态 DiT" | Stable Diffusion 3 的架构：联合注意力，模态特定的 MLP 和归一化 |
| 块三角掩码 | "因果文本 + 双向图像" | 注意力掩码，在文本上是因果的，但在图像区域内是双向的 |
| 连续图像表示 | "无 VQ" | 图像 patch 作为实值向量，而不是整数码本索引 |
| 速度预测 | "v-参数化" | 网络输出是噪声与数据之间的速度场，而不是噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
