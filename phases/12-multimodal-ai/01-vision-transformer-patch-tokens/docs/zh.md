# Vision Transformers 与 Patch-Token 基元

> 在处理任何多模态内容之前，图像必须先转化为 Transformer 能够处理的 token 序列。2020 年的 ViT 论文通过 16x16 像素的图块、线性投影和位置编码解决了这一问题。五年后的 2026 年，所有前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍以此方式开始——编码器从 ViT 演变为 DINOv2 再到 SigLIP 2，加入了注册 token，位置方案变为 2D-RoPE，但基元保持不变。本节课将端到端地解读 patch-token 流水线，并用 stdlib Python 实现它，以便 Phase 12 的其他内容能够对“视觉 token”有一个具体的心理模型。

**类型：** 学习
**语言：** Python（stdlib，patch tokenizer + 几何计算器）
**前置条件：** Phase 7（Transformer），Phase 4（计算机视觉）
**时长：** 约 120 分钟

## 学习目标

- 将一张 HxWx3 的图像转化为具有正确位置编码的 patch token 序列。
- 针对给定的（patch 大小、分辨率、隐藏维数、深度）ViT，计算序列长度、参数量和 FLOPs。
- 列举使 ViT 从 2020 年研究走向 2026 年生产的三个升级：自监督预训练（DINO / MAE）、注册 token 和原生分辨率打包。
- 为下游任务在 CLS 池化、均值池化和注册 token 之间做出选择。

## 问题背景

Transformer 处理的是向量序列。文本天然是序列（字节或 token）。而图像是一个具有三个颜色通道的二维像素网格——不是序列。如果将每个像素展平，一张 224x224 的 RGB 图像将产生 150,528 个 token，而此长度下的自注意力无法处理（序列长度的二次复杂度）。

2020 年之前的做法是在前端嵌入一个 CNN 特征提取器：ResNet 生成 7x7 的 2048 维特征图，然后将这 49 个 token 送入 Transformer。这个方法可行，但继承了 CNN 的偏置（平移等变性、局部感受野），并失去了 Transformer 对规模扩展的渴望。

Dosovitskiy 等人（2020）提出了一个直白的问题：如果我们跳过 CNN 呢？将图像分割成固定大小的图块（例如 16x16 像素），将每个图块线性投影成一个向量，加上位置编码，然后送入纯 Transformer。在当时这被视为异端——不用卷积做视觉。但有了足够多的数据（JFT-300M，然后是 LAION），它在 ImageNet 上超越了 ResNet，并持续提升。

到 2026 年，ViT 基元已成为毋庸置疑的基础。每个开源权重 VLM 的视觉塔都是其某种后裔（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“我们应该使用图块吗？”，而是“使用多大的 patch 尺寸、什么分辨率调度、什么预训练目标、什么位置编码？”

## 概念

### 图块作为 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，你将图像切割成 `(H/P) x (W/P)` 个互不重叠的图块网格。每个图块是一个 `P x P x 3` 的像素立方体。将每个立方体展平成一个 `3 P^2` 的向量。应用一个形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，将每个图块映射到模型的隐藏维数 `D`。

对于 ViT-B/16 的标准配置：
- 分辨率 224，patch 尺寸 16 → 网格 14x14 → 196 个 patch token。
- 每个图块是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

图块投影在数学上等价于一个二维卷积，卷积核大小为 `P`，步长为 `P`，输出通道为 `D`。生产代码正是以这种方式实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”的表述是概念性的；卷积的表述是高效的。

### 位置编码

图块本身没有固有的顺序——Transformer 将其视为一个集合。早期 ViT 添加了一个可学习的 1D 位置编码（每个位置一个 768 维向量，共 197 个）。这种方法可行，但将模型绑定到了训练分辨率：在推理时如果需要改变网格，就必须对位置表进行插值。

现代视觉主干使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解的 2D 位置。2D-RoPE 根据图块的（行、列）索引旋转 query 和 key 向量，使模型能够从旋转角度推断相对的二维位置。没有位置表。模型在推理时可以处理任意网格尺寸。

### CLS token、池化输出与注册 token

什么是图像级别的表示？有三种并存的选择：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 Transformer 块后，CLS token 的隐藏状态即为图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. 均值池化。对 patch token 的输出隐藏状态取平均。由 SigLIP、DINOv2 及大多数现代 VLM 使用。
3. 注册 token。Darcet 等人（2023）观察到，未经显式 sink token 训练的 ViT 会发展出高范数的“伪影”图块，这些图块劫持了自注意力。添加 4–16 个可学习的注册 token 可以吸收这种负载，并改善密集预测（分割、深度）的质量。DINOv2 和 SigLIP 2 均内置了注册 token。

选择对下游任务有影响。CLS 适用于分类。对于将 patch token 馈入 LLM 的 VLM，你会跳过池化——每个 patch 都成为 LLM 的输入 token。注册 token 在移交前会被丢弃（它们只是脚手架，而非内容）。

### 预训练：监督式、对比式、掩码式、自蒸馏式

2020 年的 ViT 使用 JFT-300M 上的监督分类进行预训练。很快被以下方案取代：

- CLIP（2021）：基于 4 亿对图文对的对比学习。第 12.02 课。
- MAE（2021，He 等人）：掩码 75% 的图块，重建像素。自监督，适用于纯图像。
- DINO（2021）/ DINOv2（2023）：使用学生-教师机制进行自蒸馏，无需标签，无需描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉主干，也是“密集特征”用例的默认选择。
- SigLIP / SigLIP 2（2023、2025）：使用 sigmoid 损失和用于原生宽高比的 NaFlex 技术的 CLIP。2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）中的主导视觉塔。

你对预训练的选择决定了主干擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游微调的起点。

### 缩放定律

ViT 缩放（Zhai 等人，2022）确立了 ViT 的质量遵循关于模型大小、数据大小和计算量的可预测规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch 尺寸是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）每张图像产生的 token 数多于 patch 16；有利于 OCR 和密集任务，但速度更慢。
- 分辨率是另一个重要杠杆。从 224 到 384 再到 512 几乎总是有帮助，但计算量按二次方增长。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两个工作马编码器。

### ViT 的参数量

完整计算位于 `code/main.py` 中。对于 ViT-B/16 在 224 分辨率下：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载检查点之前，请按此方法估算每个 ViT。在后续任何 VLM 中，主干大小决定了你的显存下限。

### 2026 年生产配置

2026 年大多数开源 VLM 附带的编码器是在原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它具有：
- 400M 参数。
- Patch 尺寸 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务使用均值池化；对于 VQA，所有 729 个 patch 都流入 LLM。
- 4 个注册 token，在 LLM 移交前丢弃。
- 带有图像级缩放的 2D-RoPE，用于原生宽高比。

该配置中的每一个决策都可以追溯到一篇你可以阅读的论文。

## 使用它

`code/main.py` 是一个 patch tokenizer 和几何计算器。它接收（图像 H、W、patch P、隐藏维数 D、深度 L）并输出：

- 分块后的网格形状和序列长度。
- 一个合成 8x8 像素玩具图像的 token 序列（逐步演示展平 + 投影路径）。
- 参数量分解：patch 嵌入、位置嵌入、Transformer 块和头部。
- 在目标分辨率下每次前向传播的 FLOPs。
- 一个对比表格，涵盖 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384。

运行它。将参数量与已发表数字进行匹配。调整 patch 尺寸和分辨率，感受 token 数量的代价。

## 交付产出

本节课产生 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 尺寸、分辨率、隐藏维数、深度），它会生成一个 token 数量、参数量和显存估算，并附有理由说明。当你在选择 VLM 的视觉主干时使用这项技能——它可以防止“token 爆炸导致我的 LLM 上下文填满”的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 尺寸 14 下的 patch-token 序列长度。与仅使用 CLS 的表示相比如何？

2. 一个 1080p 帧（1920x1080）在 patch 14 下会生成多少个 token？在 30 FPS、5 分钟的视频中，总共会有多少个视觉 token？哪种成本节省最大：池化、帧采样还是 token 合并？

3. 用纯 Python 实现 patch token 的均值池化。验证在对 DINOv2 输出的 196 个 token 进行均值池化后，结果与模型 `forward` 在请求池化嵌入时返回的内容一致。

4. 阅读《Vision Transformers Need Registers》论文的第 3 节（arXiv:2309.16588）。用两句话描述注册 token 吸收了哪种伪影，以及为什么它对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一个不同分辨率的图像列表，生成单个打包序列和块对角注意力掩码。在后续学习第 12.06 课时进行验证。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Patch | “16x16 像素正方形” | 输入图像中的固定大小非重叠区域；成为一个 token |
| Patch embedding | “线性投影” | 一个共享的 learned 矩阵（或 stride=P 的 Conv2d），用于将展平的图块像素映射到 D 维向量 |
| CLS token | “分类 token” | 前置的可学习向量，其最终隐藏状态代表整幅图像；2026 年可选 |
| Register token | “吸收 token” | 额外的可学习 token，用于吸收 ViT 在预训练过程中发展出的高范数注意力伪影 |
| Position embedding | “位置信息” | 逐位置向量或旋转，使序列感知顺序；2D-RoPE 是现代默认方案 |
| Grid | “图块网格” | 给定分辨率和 patch 尺寸下的 (H/P) x (W/P) 二维图块阵列 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 特性：单一模型无需重新训练即可服务多种宽高比和分辨率 |
| Backbone | “视觉塔” | 预训练的图像编码器，其 patch token 输出在 VLM 中馈入 LLM |
| Pooling | “图像级摘要” | 将 patch token 转化为一个向量的策略：CLS、均值、注意力池化或基于注册 token |
| Patch 14 vs 16 | “更细粒度 vs 更粗粒度” | Patch 14 每张图像产生更多 token，OCR 保真度更高，速度更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — 注册 token 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
