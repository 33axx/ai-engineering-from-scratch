# 图像修复、扩展与编辑

> 文生图创造新事物，图像修复则修补旧有之物。在实际生产中，70% 的计费图像工作都是编辑——替换背景、移除Logo、扩展画布、重新生成手部。图像修复正是扩散模型大显身手的领域。

**类型：** 实战
**语言：** Python
**前置知识：** 阶段8·07（潜空间扩散），阶段8·08（ControlNet & LoRA）
**时长：** ~75分钟

## 问题

客户发来一张完美的产品照片，但背景中有一个干扰性的标志。你想擦除这个标志，同时保持其他像素完全一致。你不能从头开始运行文生图——结果会有不同的颜色、光照和产品角度。你只想重新生成*仅*被遮罩的区域，并且希望重新生成的结果能尊重周围的上下文。

这就是图像修复。变体包括：

- **图像修复（Inpainting）**：在遮罩内部重新生成，保持外部像素不变。
- **图像扩展（Outpainting）**：在遮罩外部（或画布之外）重新生成，保持内部像素不变。
- **图像编辑（Image Editing）**：重新生成整张图像，但保持与原始图像的语义或结构一致性（SDEdit, InstructPix2Pix）。

2026年的每个扩散管线都内置了图像修复模式：Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们都遵循相同的原理。

## 概念

![图像修复：带有遮罩感知去噪和上下文保持重注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它不对）

使用遮罩运行标准的文生图。在每个采样步骤中，将未遮罩区域的含噪潜变量替换为经过前向扩散的干净图像。这样做的效果……很糟糕。边界伪影会渗透出来，因为模型对遮罩区域内是什么完全没有信息。

### 正确的图像修复模型

训练一个经过修改的U-Net，输入9个通道而不是4个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是经过VAE编码的源图像的一个副本，加上一个单通道的遮罩。在训练时，你随机遮罩图像的部分区域，并训练模型仅对遮罩区域进行去噪，而未遮罩区域作为干净的 conditioning 信号给出。在推理时，模型可以“看到”遮罩区域周围的内容，从而生成一致的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种9通道（或类似）输入。Diffusers 中的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等人，2022）——免费编辑

向源图像添加噪声到某个中间步 `t`，然后从 `t` 向下运行反向链到 0，同时使用新的提示词。无需重新训练。起始 `t` 的选择在保真度和创作自由度之间进行权衡：

- `t/T = 0.3` → 与源图像几乎相同，微小的风格变化
- `t/T = 0.6` → 适度编辑，保持粗略结构
- `t/T = 0.9` → 从接近噪声生成，源图像保留最少

### InstructPix2Pix（Brooks 等人，2023）

在 `(输入图像, 指令, 输出图像)` 三元组上微调扩散模型。推理时，同时基于输入图像和文本指令（“使其变成日落”、“添加一条龙”）进行条件生成。有两个CFG尺度：图像尺度与文本尺度。

### RePaint（Lugmayr 等人，2022）

保留一个标准的无条件扩散模型。在每个反向步骤中，重新采样——偶尔跳回一个更嘈杂的状态并重新生成。避免边界伪影。在你没有经过训练的图像修复模型时使用。

## 实战构建

`code/main.py` 在5维数据上实现了一个玩具级别的1维图像修复方案。我们在5维混合数据上训练一个DDPM，每个样本是来自两个簇之一的5个浮点数。在推理时，我们“遮罩”5个维度中的2个，在每个步骤中注入未遮罩三个维度的加噪前向版本，并仅重新生成被遮罩的维度。

### 第一步：5维DDPM数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第二步：在所有5个维度上训练去噪器

标准DDPM。网络为5维含噪输入输出5维噪声预测。

### 第三步：推理时，遮罩感知的反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，并且在玩具1维数据上有效。真实的图像修复使用9通道输入，因为纹理连贯性更为重要。

### 第四步：图像扩展

图像扩展是遮罩反转后的图像修复：遮罩新的（先前不存在的）画布，其余部分用原始图像填充。训练目标相同。

## 陷阱

- **接缝。** 朴素方法会留下可见的边界，因为梯度信息无法穿过遮罩流动。解决方法：将遮罩膨胀8-16像素，或使用正确的图像修复模型。
- **遮罩泄漏。** 如果 conditioning 图像的未遮罩区域质量低或噪声大，它会污染遮罩内的生成结果。可进行轻微去噪或模糊处理。
- **CFG与遮罩大小相互作用。** 在小遮罩上使用高CFG会导致色块饱和。对于小编辑，降低CFG。
- **SDEdit保真度悬崖。** 从 `t/T = 0.5` 变为 `t/T = 0.6` 可能会丢失主体的身份。建议进行扫参并设置检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而不仅是新内容。“一只坐在椅子上的猫”而不是“一只猫”。

## 使用场景

| 任务 | 管线 |
|------|------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + “日落时的蓝天” |
| 扩展画布 | SDXL 扩展模式（8像素羽化）或带扩展遮罩的 Flux-Fill |
| 重新生成手部/面部 | SD-Inpaint + 重新描述主体的提示词 + ControlNet-Openpose |
| 改变某个区域的风格 | 对遮罩区域使用 SDEdit，`t/T=0.5` |
| “使其变成日落” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | 对于最困难的案例，使用 Flux-Fill 或 GPT-Image（托管） |

SAM（Meta 的 Segment Anything，2023）+ 扩散图像修复是2026年的背景移除管线。SAM 2（2024）可用于视频。

## 交付输出

保存 `outputs/skill-editing-pipeline.md`。技能要求：输入原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示），输出：遮罩生成方法、基础模型、CFG尺度（图像+文本）、SDEdit-t 或图像修复模式、以及QA检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将遮罩维度的比例从0.2变化到0.8。在什么比例下，图像修复质量（遮罩维度上的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每第10个反向步骤，跳回5步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 进行比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在20个面部重新生成任务上的表现。分别对姿态一致性和身份保持进行评分。

## 关键术语

| 术语 | 日常说法 | 实际含义 |
|------|----------|----------|
| Inpainting | “填满空洞” | 在遮罩内部重新生成；保持外部像素。 |
| Outpainting | “扩展画布” | 在画布外部重新生成；保持内部像素。 |
| 9-channel U-Net | “正确的图像修复模型” | 输入为 `含噪潜变量 \| 编码后的源图像 \| 遮罩` 的U-Net。 |
| SDEdit | “带噪声级别的图生图” | 加噪到时间步 `t`，用新提示词去噪。 |
| InstructPix2Pix | “仅文本编辑” | 在（图像，指令，输出）三元组上微调的扩散模型。 |
| RePaint | “无需重新训练” | 在反向过程中周期性地重新加噪以减少接缝。 |
| SAM | “Segment Anything” | 通过点击或框选生成遮罩；与图像修复配对使用。 |
| Flux-Kontext | “带上下文的编辑” | Flux 变体，接受参考图像+指令进行编辑。 |

## 生产注意事项：编辑管线对延迟敏感

用户编辑图像时，期望往返时间低于5秒。在 L4 上，30步的 SDXL-Inpaint（1024²）需要3-4秒，加上 SAM 遮罩生成（~200毫秒）和 VAE 编码/解码（合计~500毫秒）。在生产框架中，这更受TTFT（首令牌延迟）限制而非吞吐量限制——批次大小为1，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 下大约200毫秒；SAM-ViT-B 大约40毫秒，质量损失较小。SAM 2（视频）增加了时间开销；对于单图像编辑不要使用它。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 将图像编码为潜变量。如果你有前一次生成得到的潜变量（这在迭代编辑UI中很常见），可以直接通过 `latents=...` 传入，从而跳过一次 VAE 编码。
- **遮罩膨胀对吞吐量也很重要。** 小遮罩意味着大部分U-Net前向传递被浪费（未遮罩像素无论如何都会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论遮罩大小都会运行完整的U-Net；只有 9 通道的正确图像修复变体才能利用遮罩计算。
- **Flux-Kontext 是2025年的答案。** 对 `(源图像, 指令)` 进行单次前向传递——无需单独的遮罩，无需 SDEdit 噪声扫参。在 H100 上，它能在约1.5秒内完成一次编辑。架构上的教训：将多个阶段合并。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) —— 无需训练的图像修复。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) —— SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) —— 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) —— SAM，遮罩来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) —— 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) —— 注意力层面的编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) —— 2024年工具集。
