# 潜在扩散与 Stable Diffusion

> 在 512×512 图像上进行像素空间扩散是计算战犯级别的行为。Rombach 等人（2022）注意到，生成一张图像不需要全部 786k 维特征——只需要足够的维度来捕获语义结构，其余部分交给单独的解码器。在 VAE 的潜在空间内运行扩散。这一个想法就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8·02（VAE），阶段 8·06（DDPM），阶段 7·09（ViT）
**时间：** 约 75 分钟

## 问题

在 512² 的像素空间进行扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。对于一个 500M 参数的 U-Net，每个采样步骤大约是 100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿张图像上训练，计算账单高得离谱。

这些 FLOPs 大部分花在了将感知上不重要的细节推过网络——那些有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：训练一次 VAE（*第一阶段*），冻结它，然后完全在 4 通道的 64×64 潜在空间（*第二阶段*）中运行扩散。同样的 U-Net。1/16 的像素。约为可比较质量所需 FLOPs 的 1/64。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 潜在空间上使用了 860M 的 U-Net，SDXL 在 `128×128×4` 上使用了 2.6B 的 U-Net，SD3 用带有流匹配的扩散 Transformer（DiT）替换了 U-Net。Flux.1-dev（Black Forest Labs，2024）搭载了一个 12B 参数的 DiT-MMDiT。所有这些都运行在相同的两阶段基底上。

## 概念

![潜在扩散：VAE 压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8 倍下采样 + 调整通道数，使总潜在大小约为像素数的 1/16。损失 = 重构（L1 + LPIPS 感知损失）+ KL（权重较小，这样 `z` 不会被强制太接近高斯分布，因为我们不需要从 `z` 精确采样）。通常还会加上对抗损失，使解码后的图像锐利。

2. **阶段 2 — 对 `z` 的扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）对 `z_t` 进行去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件控制。** 两个额外组件。一个冻结的文本编码器（SD 1.x 使用 CLIP-L，SD 2/XL 使用 CLIP-L+OpenCLIP-G，SD3 和 Flux 使用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块采用 `[Q = 图像特征, K = V = 文本 token]` 并进行混合。这些 token 是文本影响图像的唯一方式。

**损失函数与第 06 课相同。** 同样的 DDPM / 流匹配 MSE 作用于噪声。你只需要切换数据域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 潜在形状 | 文本编码器 | 参数 |
|------|------|----------|----------|------------|------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 token) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + 精炼器 | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 同上 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | 蒸馏 MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趋势：用 DiT（基于潜在块的 Transformer）替换 U-Net，扩大文本编码器（T5 在提示遵循方面优于 CLIP），增加潜在通道数（4→16 提供更多细节空间）。

## 动手构建

`code/main.py` 在第 06 课的 DDPM 之上堆叠了一个玩具 1-D “VAE”（编码器和解码器均为恒等映射，用于演示；真正的 VAE 会是卷积网络），并加入了类别条件控制和分类器无引导。它展示了同样的扩散损失无论是在原始 1-D 值上运行还是在编码后的值上运行都有效——这是关键洞见。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的 VAE 有训练好的权重。出于教学目的，这个线性映射足以说明扩散在 `z` 上运行而不关心原始数据空间。

### 步骤 2：在 `z` 空间中的扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：分类器无引导

训练期间，10% 的概率丢弃类别标签（替换为 null token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和/过锐化。

### 步骤 4：文本条件控制（概念，非代码）

将类别标签替换为冻结的文本编码器输出。通过交叉注意力将文本嵌入注入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一的实质性区别。

## 陷阱

- **VAE 尺度不匹配。** SD 1.x 的 VAE 在编码后应用了一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这个常数会使 U-Net 在方差严重错误的潜在值上进行训练。每个检查点都附带了该常数。
- **文本编码器功能不全但悄无声息。** SD3 需要 T5-XXL 并且 >=128 个 token，仅回退到 CLIP 会有损失。务必检查 `use_t5=True`，否则提示保真度会崩溃。
- **混合潜在空间。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 潜在空间上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的检查点。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，并过度拟合提示而牺牲多样性。最佳范围是 `w = 3-7`。
- **负面提示泄露。** 空的负面提示会成为 null token；填充了内容的负面提示成为 `ε_uncond`。这两者并不相同；某些流水线会静默地默认为 null。

## 应用

2026 年的生产栈：

| 目标 | 推荐骨干网络 |
|------|-------------|
| 窄领域、成对数据、从头训练模型 | SDXL 微调（LoRA / 全量）—— 最快上线 |
| 开放域文本到图像、开源权重 | Flux.1-dev（12B，Apache / 非商业）或 SD3.5-Large |
| 最快推理、开源权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示遵循、托管服务 | GPT-Image / DALL-E 3（仍然）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）—— 原生接受图像+文本 |
| 研究、基线 | SD 1.5 —— 古老但研究充分 |

## 交付

保存 `outputs/skill-sd-prompter.md`。技能接受文本提示 + 目标风格，输出：模型 + 检查点、CFG 尺度、采样器、负面提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及一个逐步骤的 QA 检查表。

## 练习

1. **简单。** 运行 `code/main.py`，设置引导值 `w ∈ {0, 1, 3, 7, 15}`。记录每个类别的平均样本。当 `w` 为多少时，类别均值偏离了真实数据均值？
2. **中等。** 将玩具线性编码器替换为带有重构损失的 tanh-MLP 编码器/解码器对。在新的潜在值上重新训练扩散。样本质量会改变吗？
3. **困难。** 使用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，运行 30 步 Euler 采样器，CFG=7，计时。然后切换到 `sdxl-turbo`，4 步，CFG=0。相同主题，不同质量——描述发生了什么变化以及原因。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| 第一阶段 | “VAE” | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | “U-Net” | 对潜在空间的扩散模型。 |
| CFG | “引导尺度” | `(1+w)·ε_cond - w·ε_uncond`；调节条件控制强度。 |
| Null token | “空提示嵌入” | 用于 `ε_uncond` 的无条件嵌入。 |
| 交叉注意力 | “文本如何进入” | 每个 U-Net 块以文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | “扩散 Transformer” | 用 Transformer 替换 U-Net，作用于潜在块；更好的可扩展性。 |
| MMDiT | “多模态 DiT” | SD3 的架构：文本和图像流带有联合注意力。 |
| VAE 缩放因子 | “魔法数字” | 将潜在值除以约 5.4，使扩散在单位方差空间中运行。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是“我有消费级 GPU，能把它部署上线吗？”这一问题的经典配方。诀窍与生产推理文献中列出的三旋钮配方相同，应用于扩散 DiT：

1. **交错加载。** Flux 有三个网络，它们永远不需要同时存在于显存中：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（小）、12B MMDiT 以及 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能容纳一个阶段。
2. **通过 bitsandbytes 进行 4 位量化。** 在 T5 编码器和 DiT 上都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，根据 Aritra 的基准测试（链接在 notebook 中），文本到图像的质量下降几乎不可感知。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 会在每次前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 的延迟，但使得流水线能够运行。

内存计算：`10 GB T5 / 8 = 1.25 GB` 量化后，`12 B 参数 × 0.5 bytes = ~6 GB` 量化 DiT，加上激活值。用 stas00 的话说，这是 TP=1 推理的极端情况——没有模型并行，最大程度的量化。生产环境你会使用 TP=2 或 TP=4 在 H100 上运行；对于单个开发者笔记本，这是就是配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述每个检查点的参考实现。
