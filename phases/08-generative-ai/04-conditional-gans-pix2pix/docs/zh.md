# 条件生成对抗网络 & Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 生成的内容。附加一个标签、一张图像或一个句子。Pix2Pix 实现了图像版本，在狭窄的图像到图像任务上，它仍然胜过所有通用的文本到图像模型。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段8·03（GANs），阶段4·06（U-Net），阶段3·07（CNNs）
**时间：** 约75分钟

## 问题

无条件的 GAN 会采样任意的人脸。对演示有用，但在生产中无用。你需要的是：*将草图映射为照片*，*将地图映射为航拍图*，*将白天场景映射为夜晚*，*为灰度图像上色*。在所有这些任务中，你得到一张输入图像 `x`，必须输出具有某种语义对应关系的 `y`。每个 `x` 对应多个合理的 `y`。均方误差会将它们模糊成一团。对抗损失不会这样，因为“看起来真实”是清晰的。

条件 GAN（Mirza & Osindero, 2014）将一个条件 `c` 作为输入添加到生成器 `G` 和判别器 `D` 中。Pix2Pix（Isola 等人，2017）将其特化：条件是完整的输入图像，生成器是 U-Net，判别器是基于*块*的分类器（PatchGAN），损失函数是对抗损失 + L1。即使在 2026 年，这种方案在狭窄的图像到图像领域仍然优于从头训练的文本到图像模型，因为它是基于*成对数据*训练的——你恰好拥有所需的信号。

## 概念

![Pix2Pix：U-Net生成器，PatchGAN判别器](../assets/pix2pix.svg)

**条件生成器 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件判别器 D。** `D(x, y) → [0, 1]`。输入是*对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 看起来真实。

**U-Net 生成器。** 带有跨瓶颈连接的编码器-解码器。对于输入和输出共享低层结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节就会消失。

**PatchGAN 判别器。** 判别器不是输出单个真实/虚假得分，而是输出一个 `N×N` 的网格，其中每个单元格判断一个约 70×70 像素的感受野。然后取平均。这是一个马尔可夫随机场假设：真实性是局部的。训练更快，参数更少，输出更清晰。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定了训练，并促使生成器向已知目标靠近。L1 比 L2 产生更清晰的边缘（中位数而非均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN —— 当你没有成对数据时

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu 等人，2017）通过增加一个额外的损失——*循环一致性*损失——去除了这一要求。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这允许你将马翻译成斑马，夏天翻译成冬天，而无需成对示例。

在 2026 年，无配对图像到图像翻译主要通过扩散模型（ControlNet, IP-Adapter）完成，而非 CycleGAN，但循环一致性思想几乎存在于每一篇无配对领域自适应论文中。

## 构建它

`code/main.py` 实现了一个在一维数据上的小型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务是：为给定类别生成条件分布下的样本。

### 步骤1：将条件附加到生成器和判别器的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方法。更大的模型使用学习到的嵌入、FiLM 调制或交叉注意力。

### 步骤2：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*下的真实分布，而不是边缘分布。

### 步骤3：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** 生成器学会了边缘化，判别器从未惩罚，因为条件信号很弱。修复方法：更积极地条件化判别器（在早期层而非仅后期层），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重过低。** 生成器漂移到任意看起来真实的输出，而不忠实于输入。对于 Pix2Pix 风格的任务，从 λ≈100 开始。
- **L1 权重过高。** 生成器产生模糊输出，因为 L1 仍然是 L_p 范数。一旦训练稳定，逐步降低权重。
- **判别器中的数据泄漏。** 将 `(x, y)` 拼接作为判别器输入，而不仅仅是 `y`。没有这一点，判别器无法检查一致性。
- **每个类别的模式崩溃。** 每个类别可以独立崩溃。运行类别条件多样性检查。

## 使用它

2026 年图像到图像任务的现状：

| 任务 | 最佳方法 |
|------|---------------|
| 草图 → 照片，同域，成对数据 | Pix2Pix / Pix2PixHD（仍然快速，仍然清晰） |
| 草图 → 照片，无配对 | 使用涂鸦条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 使用 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已成遗产 |
| 深度图 → 照片 | 在 Stable Diffusion 上使用 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

Pix2Pix 仍然是正确的工具，当 (a) 你有数千个成对示例，(b) 任务狭窄且可重复，(c) 你需要快速推理时。对于通用的开放域任务，扩散模型胜出。

## 交付它

保存 `outputs/skill-img2img-chooser.md`。该技能接受任务描述、数据可用性（成对/无配对、样本数 N）以及延迟/质量预算，然后输出：方法（Pix2Pix, CycleGAN, ControlNet 变体, SDXL + IP-Adapter）、训练数据需求、推理成本以及评估协议（LPIPS, FID, 任务特定）。

## 练习

1. **简单。** 修改 `code/main.py`，添加第三个类别。确认生成器仍然将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设置中，将 L1 替换为感知风格损失（例如，使用一个小型冻结的判别器作为特征提取器）。它是否改变了条件分布的清晰度？
3. **困难。** 在一维设置中勾勒出 CycleGAN：两个分布、两个生成器、循环损失。证明它可以在没有成对数据的情况下学习它们之间的映射。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 条件 GAN | “带标签的 GAN” | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | “图像到图像 GAN” | 成对条件 GAN，带有 U-Net 生成器和 PatchGAN 判别器 + L1 损失。 |
| U-Net | “带跳跃连接的编码器-解码器” | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | “局部真实性分类器” | 判别器输出每个块的得分，而非全局得分。 |
| CycleGAN | “无配对图像翻译” | 两个生成器 + 循环一致性损失；无需成对数据。 |
| SPADE | “GauGAN” | 使用语义图对中间激活进行归一化；分割到图像。 |
| FiLM | “特征级线性调制” | 根据条件对每个特征进行仿射变换；廉价的条件化方法。 |

## 生产注意事项：Pix2Pix 作为延迟受限的基线

当你拥有成对数据和一个狭窄任务（草图→渲染、语义图→照片、白天→夜晚）时，Pix2Pix 的单次推理在延迟上比扩散模型快一个数量级。生产中的比较通常是：

| 路径 | 步数 | 单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批处理中（每个请求的 FLOPs 相同）在吞吐量上胜出。扩散模型在质量和泛化上胜出。现代的做法通常是：为狭窄任务部署一个 Pix2Pix 风格的蒸馏模型，并为尾部输入准备一个扩散回退方案。

## 进一步阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— 投影判别器。
