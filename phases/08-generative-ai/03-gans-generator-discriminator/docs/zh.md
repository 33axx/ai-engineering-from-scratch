# GANs — 生成器 vs 判别器

> Goodfellow 在 2014 年的妙招是直接跳过密度估计。两个网络。一个负责伪造，一个负责识破。它们相互对抗，直到伪造品与真实样本无法区分。理论上这不该成功，实践中也常常失败。但当它成功时，生成的样本在窄域任务中仍然是文献里最清晰的。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 08（优化器），Phase 8 · 02（VAE）
**预计时长：** ~75 分钟

## 问题所在

VAE 会产生模糊样本，因为其 MSE 解码器损失在贝叶斯意义下优化的是*平均*图像——而许多合理数字的平均值就是一个模糊数字。你需要一种损失函数，奖励的是*合理性*，而不是与某个目标在像素级别上的接近程度。合理性没有封闭形式的表达式，你必须自己去学习。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像与伪造图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为什么东西看起来真实的那个信号。当 `G` 改进时，这个信号也会更新，追逐一个移动的目标。如果两个网络收敛，那么 `G` 就学会了数据分布，而从未显式写出 `log p(x)`。

这就是对抗训练。其数学形式是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 的生成器（扩散模型和流匹配已经取代了这个王冠）。但 StyleGAN 2/3 仍然是有史以来最清晰的人脸模型；GAN 的判别器被用作扩散训练中的*感知损失*；对抗训练驱动着快速 1 步蒸馏（SDXL-Turbo, SD3-Turbo, LCM），让你能够实现实时的扩散推理。

## 概念

![GAN 训练：生成器与判别器进行极小极大博弈](../assets/gan.svg)

**生成器 `G(z)`**。将噪声向量 `z ~ N(0, I)` 映射到一个样本 `x̂`。一个解码器形状的网络（全连接或转置卷积）。

**判别器 `D(x)`**。将一个样本映射到一个标量概率（或分数）。真实 → 1，伪造 → 0。

**损失函数**。两种交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。真实标签为 1、伪造标签为 0 的二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 置信度高时会饱和，导致梯度消失）。

**训练循环**。对 `D` 做一个步骤，对 `G` 做一个步骤。重复。

**为什么有效**。如果 `G` 完美匹配 `p_data`，那么 `D` 只能做出随机猜测，处处输出 0.5；`G` 不再获得梯度。达到均衡。

**为什么崩溃**。模式坍缩（`G` 找到一个 `D` 无法分类的模式，然后无限生成）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批量大小、任何因素）。

## 使 GAN 起作用的变体

| 年份 | 创新 | 修复内容 |
|------|------|----------|
| 2015 | DCGAN | 卷积/转置卷积、批归一化、LeakyReLU——第一个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 将 BCE 替换为 Wasserstein 距离 + 梯度惩罚。解决了梯度消失问题。 |
| 2017 | 谱归一化 | 对判别器施加 Lipschitz 约束。2026 年的判别器中仍然使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再添加层。首个百万像素级结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域照片级真实感的 SOTA。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年仍然是人脸领域的黄金标准。 |
| 2022 | StyleGAN-XL | 条件化、类别感知、更大规模。 |
| 2024 | R3GAN | 用更强的正则化重新包装；无需技巧即可在 1024² 分辨率上工作。 |

## 构建它

`code/main.py` 在一个 1 维数据（两个高斯分布的混合）上训练一个微小的 GAN。生成器和判别器都是单隐藏层 MLP。我们手动实现前向、反向和极小极大循环。目标是观察训练中出现的两种关键失败模式（模式坍缩 + 梯度消失）。

### 第 1 步：非饱和损失

原始的 Goodfellow 损失 `log(1 - D(G(z)))` 在 `D` 以高置信度将 `G` 的伪造品判别为伪造时趋于 0。此时 `G` 的梯度几乎为零——`G` 无法改进。而非饱和形式 `-log D(G(z)))` 具有相反的渐近线：当 `D` 置信度高时它会爆炸，给 `G` 提供强烈的信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每个生成器步对应一个判别器步

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

每次为 `G` 生成新的伪造品，否则梯度会过时。

### 第 3 步：观察模式坍缩

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个不再被生成。判别器不再纠正它，因为它从未被当作伪造品看待。

## 陷阱

- **判别器过强。** 将 `D` 的学习率降低 2–5 倍，或者添加实例/层噪声。如果 `D` 的准确率达到 >95%，`G` 就死了。
- **生成器记住了某个模式。** 向 `D` 的输入添加噪声，使用小批量判别器层，或者切换到 WGAN-GP。
- **批归一化泄漏统计量。** 当真实批次和伪造批次流经同一个 BN 层时，它们的统计量会混合。改用实例归一化或谱归一化。
- **Inception 分数被人为拔高。** 在样本数量较少时，FID 和 IS 噪声很大。评估时使用 ≥10k 个样本。
- **条件任务中一次采样即可成功的说法不可靠。** 你仍然需要 CFG 缩放、截断技巧和重新采样才能得到可用的输出。

## 使用场景

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|------|------|
| 照片级真实人脸，固定姿势 | StyleGAN3（最清晰，体积最小） |
| 动漫/风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像的转换 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速的单步文生图 | 扩散模型的对抗蒸馏（SDXL-Turbo, SD3-Turbo） |
| 扩散训练器内部的感知损失 | 对图像小块使用小型 GAN 判别器 |
| 任何多模态、开放式任务 | 不要用——改用扩散模型或流匹配 |

GAN 很清晰但领域较窄。一旦您的领域变得开放——照片、任意文本提示、视频——请切换到扩散模型。对抗技巧作为一个组件（感知损失、蒸馏）存活下来，而非独立的生成器。

## 交付物

保存 `outputs/skill-gan-debugger.md`。该技能接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小），输出一个按可能性排序的原因列表、一行修复建议以及一份重新运行的协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后将 `D_LR = 5 * G_LR` 并重新运行。`G` 的损失需要多久才能坍缩到一个常数？
2. **中等。** 将 Goodfellow 的 BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 `D` 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较 wall-clock 收敛时间。
3. **困难。** 将 1 维示例扩展到 2 维数据（环上的 8 个高斯混合分布）。追踪生成器在步骤 1k、5k、10k 时捕获了 8 个模式中的多少个。实现小批量判别并重新测量。

## 关键术语

| 术语 | 常说的 | 实际含义 |
|------|--------|----------|
| 生成器 | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，真实 vs 伪造。 |
| 极小极大 | "那场博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "那个修补" | 对 `G` 使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
| 模式坍缩 | "G 只记住了一个东西" | 尽管数据多样，生成器只产生少量不同的输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；梯度更平滑。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 `D` 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个能用的" | 映射网络 + AdaIN；人脸领域最佳，2026 年仍是。 |

## 生产环境说明：单次推理是 GAN 的持久优势

GAN 在开放域生成上已不再赢得样本质量之争，但它们在推理成本上仍然胜出。在生产推理文献的术语中，一个 GAN 具有：

- **没有预填充、没有解码阶段。** 一次 `G(z)` 前向传播即可。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一的状态是权重。批量大小受激活内存限制，而非缓存。
- **简单的连续批处理。** 由于每个请求的 FLOPs 是固定的，通常在服务器目标占用率下采用静态批处理是最优的。不需要在线调度器。

这就是为什么在 2026 年，GAN 蒸馏（SDXL-Turbo, SD3-Turbo, ADD, LCM）是快速文生图的主导技术：它将 20–50 步的扩散流水线压缩成 1–4 步的 GAN 风格前向传播，同时保持扩散基座模型的分布。对抗损失作为训练时的调节旋钮存活下来，用于将慢生成器变成快生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定的架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
