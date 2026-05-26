# CLIP 与对比视觉语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动未来五年的核心思想：仅使用噪声网页图像-标题对和一个对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4亿对。得到的嵌入空间实现了零样本分类、图像-文本检索，并作为每个 2026 年 VLM 的视觉塔插入其中。SigLIP 2（2025）用 sigmoid 替换了 softmax，以更低的成本超越了 CLIP。本课从 InfoNCE 到 sigmoid 对损失的数学推导，并用标准库 Python 构建训练步骤。

**类型：** 构建  
**语言：** Python（标准库，包含 InfoNCE 和 sigmoid 损失实现）  
**前置知识：** 阶段 12·01（ViT 分块），阶段 7（Transformer）  
**时间：** 约 180 分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 对损失（SigLIP）可以扩展到批次大小 32768+，而无需 softmax 所需的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行 ImageNet 零样本分类。
- 指出 CLIP/SigLIP 预训练提供的四个杠杆：批次大小、温度、提示模板、数据质量。

## 问题

CLIP 之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，然后发布。标签昂贵，标签偏向标注者能达成一致的内容，且标签无法直接迁移到新任务而无需微调。

图像-标题网络中有超过十亿个松散标注的对可以免费使用。一张金毛犬的图片，其替代文本为“我的狗 Max 在公园里”，携带了监督信号——文本描述了图像。问题在于：你能将其转化为有用的训练吗？

CLIP 的答案：将图像-标题对视为匹配任务。给定一批 N 张图像和 N 个标题，学习将每张图像与其自己的标题匹配，并从 N-1 个干扰项中区分出来。监督信号是“这两个东西属于一起；这 N-1 个不属于。”没有类别标签。没有人工注释。只有对比损失。

得到的嵌入空间所做的远不止 CLIP 训练的任务。ImageNet 零样本之所以有效，是因为“一张猫的照片”嵌入到附近一些从未被明确标记为猫的照片附近。这就是催生每个 2026 年 VLM 的赌注。

## 概念

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每个标题输出一个 D 维向量。

两个塔都将其输出归一化为单位长度。相似度是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于一个包含 N 个（图像，标题）对的批次，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的温度（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 使用对称的交叉熵，在行和列上计算：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与其标题的匹配程度超过批次中的其他所有标题。“负样本”是批次中所有其他项。更大的批次 = 更多负样本 = 更强的信号。CLIP 以批次大小 32k 训练；规模很重要。

### 温度

`tau` 控制 softmax 的尖锐程度。低 tau → 尖锐分布，产生硬负样本挖掘效果。高 tau → 平滑，所有样本均有贡献。CLIP 学习 `log(1/tau)`，并进行裁剪以防止坍塌。SigLIP 2 固定初始 tau 并使用学习的偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须将每个嵌入 all-gather 到所有副本，然后执行 softmax。这在通信上与世界大小呈二次关系。

SigLIP 用逐元素 sigmoid 替换 softmax：对于每一对 `(i, j)`，损失是“这是匹配对吗？”的二元分类。正类标签是对角线，其他所有都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 如果 `i == j`，否则为 0。每一对的损失是独立的。不需要 all-gather。每个 GPU 计算其本地块并求和。SigLIP 2 可以廉价地扩展到批次大小 32k-512k，而 CLIP 则需要比例更多的通信。

### 零样本分类

给定 N 个类别名称，对每个类别构造文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。余弦相似度的 argmax = 预测类别。不需要在目标类别上训练。

提示模板很重要。CLIP 原始论文对每个类别使用了 80 个模板（普通、艺术、照片、绘画等）并平均了嵌入。提升了 3 个 ImageNet 百分点。现代用法通常选择一个或两个模板。

### 线性探针与微调

零样本是一个基线。线性探针（在冻结的 CLIP 特征之上为你的目标类别训练一个线性层）在域内任务上胜过零样本。完全微调在域内优于线性探针，但可能损害零样本迁移。三种方案各有取舍。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增：
- NaFlex：单个模型处理可变的宽高比和分辨率。
- 更好的密集特征，用于分割和深度估计，目标是在 VLM 中用作冻结骨干。
- 多语言：在 100 多种语言上训练，而 CLIP 仅限英语。
- 10 亿参数规模，而 CLIP 最高为 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍然是纯图像-文本检索的默认选择，当特定的 LAION-2B 训练分布与查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思想，18 亿对规模，90% 噪声。证明了噪声数据可扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开源复现，多种规模，是主流的开源检查点。EVA-CLIP：从掩码图像建模初始化；是 VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合体。都是同一家族，数据和调优不同。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上大约封顶 76%（CLIP-G、OpenCLIP-G）。要超越需要更大的数据（SigLIP 2 达到 80%+）或架构变更（有监督头部、更多参数）。基准正在饱和；真正的价值是下游 VLM 消费的嵌入空间。

## 使用它

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征，文本字符特征），使你在不使用 numpy 的情况下看到 InfoNCE 的形状。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 实现数值稳定性）。
3. 用于对比的 sigmoid 对损失。
4. 一个零样本分类例程：计算与一组文本提示的余弦相似度，argmax 进行预测。

运行它并观察损失曲线。绝对数值是玩具性质的；但形状与实际 CLIP 训练器发出的匹配。

## 交付

本节课产生 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它使用 CLIP 模板构建文本提示，用指定的检查点（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回 top-1 / top-5 预测及相似度分数。该技能拒绝对提示列表之外的类别做出任何断言。

## 练习

1. 手动实现一个批次大小为 4 的 InfoNCE。构造 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用你的 Python 实现验证这个手算结果。

2. SigLIP 在温度之外使用了偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次中存在很大的类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个零样本分类器。尝试两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成能胜过单个模板吗？

4. 计算在 512 个 GPU、批次大小 32k 运行时，softmax InfoNCE 相对于 sigmoid 对损失的通信成本。哪个是 O(N)，哪个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等人）。从图中复现他们关于数据缩放的结论：在固定模型大小下，ImageNet 零样本准确率与训练数据大小之间是什么 log-linear 关系？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| InfoNCE | “对比损失” | 批次的相似度矩阵上的交叉熵；每个项的正样本是其配对项，负样本是所有其他项 |
| Sigmoid 损失 | “SigLIP 损失” | 逐对二元交叉熵；无 softmax，无 all-gather，在分布式训练中低成本扩展 |
| 温度 | “tau” | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布的尖锐程度 |
| 零样本 | “无微调分类” | 使用文本提示构造类别嵌入，通过余弦相似度进行分类；无需在目标类别上训练 |
| 提示模板 | “a photo of a ...” | 围绕类别名称的文本脚手架；影响零样本准确率 1-5 个百分点 |
| 双编码器 | “双塔” | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中 |
| 硬负样本 | “难分干扰项” | 与正样本足够相似的负样本，使得模型必须努力分离它们 |
| 线性探针 | “冻结+一层” | 仅在冻结的特征之上训练一个线性分类器；衡量特征质量 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 能够以任何宽高比和分辨率输入图像而无需调整大小 |
| 温度缩放 | “对数参数化的 tau” | CLIP 将 tau 参数化为 `log(1/tau)` 以便梯度表现良好；通过裁剪防止崩溃到接近零的 tau |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 使用噪声网络数据进行扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
