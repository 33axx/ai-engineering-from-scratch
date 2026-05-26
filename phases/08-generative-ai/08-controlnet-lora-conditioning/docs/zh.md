# ControlNet、LoRA 与条件控制

> 仅靠文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型，并用深度图、姿态骨架、手绘草图或边缘图像来引导它。LoRA 让你通过训练 1000 万个参数来微调一个 20 亿参数的模型。两者结合将 Stable Diffusion 从一个小玩具变成了 2026 年每家代理公司都在使用的图像管线。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 8 · 07（潜空间扩散）、阶段 10（从头构建 LLM——为 LoRA 打基础）  
**时间：** ~75 分钟

## 问题

像 "a woman in a red dress walking a dog on a busy street" 这样的提示词没有给模型提供关于狗在哪里、女人的姿态是什么、街道的视角的任何信息。文本大约只能指定你所需图像的 10%。剩下的都是视觉信息，无法用文字有效描述。为每一种信号（姿态、深度、Canny 边缘、分割）从头训练一个新的条件模型成本过高。你需要保留 2.6B 参数的 SDXL 主干冻结，附加一个小型侧网络来读取条件信号，并让它调整主干的中间特征。这就是 ControlNet。

你也想教导模型新概念（你的脸、你的产品、你的风格）而无需重训整个模型。你需要一个比完整模型小 100 倍的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年从业者的工具箱。大多数生产级图像管线在 SDXL / SD3 / Flux 基座上叠加了 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al.，2023）

取一个预训练的 SD。*克隆* U-Net 的编码器部分。冻结原始网络。训练克隆版本以接受额外的条件输入（边缘、深度、姿态）。使用*零卷积*跳跃连接（初始化为零的 1×1 卷积——开始时是无操作，学习增量）将克隆版本连接回原始网络的解码器部分。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 开始时为恒等映射——即便在训练前也不会造成损害。使用标准扩散损失在 1M（提示、条件、图像）三元组上训练。

每个模态的 ControlNet 以小型侧模型形式发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al.，2021）

对于模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常使用秩 4-16，重度微调使用秩 64-128。新增参数数量：`2 · d · r` 而非 `d²`。对于 SDXL 注意力层，`d=640`，`r=16`：每个适配器 2 万参数而非 41 万——减少了 20 倍。在整个模型上：LoRA 通常为 20-200MB，而基础模型为 5GB。

推理时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常范围。多个 LoRA 可以叠加相加（但需注意它们以非线性方式相互作用）。

### IP-Adapter（Ye et al.，2023）

一种小巧的适配器，接受*图像*作为条件（与文本并列）。它使用 CLIP 图像编码器生成图像 token，并将它们与文本 token 一起注入交叉注意力层。每个基础模型约 20MB。允许你"以这张参考图像的风格生成图像"而无需 LoRA。

## 组合矩阵

| 工具 | 控制什么 | 大小 | 使用时机 |
|------|----------|------|----------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格迁移 |
| IP-Adapter | 来自参考图像的风格或主体 | 20MB | 无法用文字描述外观时 |
| Textual Inversion | 单个概念作为新 token | 10KB | 遗留方法，基本被 LoRA 取代 |
| DreamBooth | 在主体上进行完整微调 | 2-5GB | 强身份特征，高计算需求 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备，推理预算受限 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

## 构建

`code/main.py` 在一维上模拟了两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 匹配一个目标线性层。展示 `r = 1` 足以完美学习一个秩 1 的修正。

2. **轻量版 ControlNet。** 一个"冻结的基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由一个学习到的标量（初始化为零）门控（这是我们版本的零卷积）。训练并观察该门控值逐渐增大。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化的侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步骤 0 时输出与基础网络相同。早期训练会缓慢更新 `gate`——不会发生灾难性漂移。

## 陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是一种常见的"让它更强"的 hack，会导致过度风格化 / 损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时使用权重 1.0 的姿态 ControlNet 和权重 1.0 的深度 ControlNet 通常会超出目标。权重之和 ≈ 1.0 是安全默认值。
- **LoRA 加载到错误的基础模型。** SDXL LoRA 静默地不生效于 SD 1.5，因为注意力层维度不匹配。Diffusers 在 0.30+ 版本会发出警告。
- **Textual Inversion 漂移。** 在一个检查点上训练的 token 在另一个检查点上严重漂移。LoRA 更便携。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中以获得更快推理（无需运行时加法），但会失去在推理时调整 `α` 的能力。保留两个版本。

## 使用场景

| 目标 | 2026 年管线 |
|------|-------------|
| 复现品牌的艺术风格 | 在约 30 张精选图像上以秩 32 训练 LoRA |
| 在生成图像中放入我的脸 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图像 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting（课程 09） |
| 快速单步风格 | SDXL-Turbo 上的 LCM-LoRA |

## 交付

保存 `outputs/skill-sd-toolkit-composer.md`。技能接收任务（输入资产：提示词、可选的参考图像、可选姿态、可选深度、可选手绘草图），并输出工具栈、权重以及可复现的种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 改变到 4。LoRA 在哪个秩时能精确匹配秩 2 的目标增量？
2. **中等。** 在两个目标变换上分别训练两个 LoRA。同时加载它们并展示其相加交互作用。交互作用在何时会偏离线性？
3. **困难。** 使用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随着堆叠权重变化，测量 FID 与提示对齐程度的权衡。

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|----------|
| ControlNet | "空间控制" | 克隆的编码器 + 零卷积跳跃连接；读取条件图像。 |
| 零卷积 | "从恒等开始" | 1×1 卷积初始化为零；ControlNet 从无操作开始。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数比完整微调少 100 倍。 |
| 秩 r | "旋钮" | LoRA 压缩程度；4-16 为典型，64+ 用于重度个性化。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 进行小型图像条件适配。 |
| DreamBooth | "完整主体微调" | 在大约 30 张主体图像上训练整个模型。 |
| Textual Inversion | "新 token" | 仅学习新的词嵌入；遗留方法，基本被取代。 |

## 生产注意事项：LoRA 热插拔、ControlNet 通道、多租户服务

一个真实的文本到图像 SaaS 通常在同一基础检查点上提供数百个 LoRA 和十几个 ControlNet。服务问题与 LLM 多租户非常相似（生产文献在连续批处理和 LoRAX / S-LoRA 中讨论了 LLM 案例）：

- **热插拔 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型会带来每步推理约 3-5% 的速度提升，但会冻结 `α` 和基础模型。将 LoRA 以秩 r 增量形式保留在 VRAM 中；diffusers 提供了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。交换成本是 `2 · d · r · num_layers` 级别的权重——MB 量级，亚秒级。
- **ControlNet 作为第二注意力通道。** 克隆的编码器与基础模型并行运行。两个权重各为 1.0 的 ControlNet 意味着每步两次额外前向传播，而非一次合并传播。批大小余量呈二次方下降。为每个活跃 ControlNet 预留约 1.5× 的步骤成本。
- **量化 LoRA 也可行。** 如果你量化了基础模型（参见课程 07，Flux 在 8GB 上），LoRA 增量也能干净地量化为 8 位或 4 位。QLoRA 风格的加载允许你在 4 位 Flux 基座上堆叠 5-10 个 LoRA 而不会超出内存。

Flux 特定：Niels 的 Flux-on-8GB 笔记本将基础模型量化为 4 位；在该量化基座上叠加一个风格 LoRA（`pipe.load_lora_weights("user/style-lora")`，`weight_name="pytorch_lora_weights.safetensors"`）仍然有效。这是大多数 SaaS 代理公司在 2026 年使用的配方。

## 进一步阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻量替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考管线。
