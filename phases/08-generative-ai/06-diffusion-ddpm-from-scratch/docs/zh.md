# 扩散模型——从零实现DDPM

> Ho、Jain、Abbeel（2020）为这一领域提供了令人难以舍弃的方法。通过上千个小步骤用噪声破坏数据，训练一个神经网络来预测噪声，在推理时逆向这一过程。如今，每一款主流的图像、视频、3D和音乐模型都基于这一循环，有时会在其上层叠加流匹配或一致性技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段3·02（反向传播），阶段8·02（VAE）
**时间：** 约75分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 玩的是一个极小极大博弈，往往发散。VAE 从高斯解码器产生模糊样本。你真正需要的是一个训练目标，它（a）是单一稳定的损失（没有鞍点，没有极小极大），（b）是 `log p(x)` 的下界（这样你就有了似然），并且（c）样本能达到 SOTA 质量。

Sohl-Dickstein 等人（2015）给出了一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 逐步加入高斯噪声，并训练一个逆向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）证明该损失可以简化为一行——预测噪声——并清理了数学表述。2020 年这还是一个新鲜事物。2021 年它生成了 SOTA 样本。2022 年它变成了 Stable Diffusion。2026 年它已成为基础组件。

## 概念

![DDPM: 前向加噪，逆向去噪](../assets/ddpm.svg)

**前向过程 `q`。** 以 `T` 个小步逐步加入高斯噪声。其闭合形式——也是数学易于处理的原因——在于累积步骤同样服从高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 按某种调度确定。将 `β_t` 从 1e-4 到 0.02 线性取值，共 T=1000 步，则 `x_T` 近似为 `N(0, I)`。

**逆向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测加入的噪声。给定 `x_t`，通过下式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或学习到的方差。这个表达式看起来复杂，但仅仅是代数运算——根据后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并用噪声预测的估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选取一个 `t`，采样 `ε ~ N(0, I)`，通过闭合形式一步计算出带噪的 `x_t`，然后对噪声进行回归。单一损失，没有极小极大，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始，从 `t = T` 到 `1` 迭代逆向步骤。完成。

## 为什么有效

三种直觉：

1. **去噪容易，生成困难。** 在 `t=T` 时，数据是纯噪声——网络只需解决一个简单问题。在 `t=0` 时，网络只需清理少数像素。在中间的 `t`，问题变难，但网络从每个噪声层级都有大量梯度流经相同的权重。

2. **伪装成评分匹配。** Vincent（2011）证明，预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*评分*。逆向 SDE 利用这个评分沿密度梯度上行走——一种导向高概率区域的随机游走。

3. **ELBO 简化为简单的 MSE。** 完整的变分下界在每个时间步都有一个 KL 项。在 DDPM 的参数化下，这些 KL 项简化为带特定系数的噪声预测 MSE；Ho 等人去掉了系数（称之为“简单”损失），而质量反而*提升了*。

## 动手实现

`code/main.py` 实现了一个一维 DDPM。数据是双峰混合。所谓的“网络”是一个小型 MLP，输入 `(x_t, t)`，输出预测噪声。训练就是那一行损失。采样则迭代逆向链。

### 步骤 1：前向调度（闭合形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一步训练

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：逆向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个有 40 个时间步和 24 单元 MLP 的一维问题，该方法约 200 个 epoch 即可学到双峰混合。

## 时间条件化

网络需要知道它正在去噪的是哪个时间步。两种标准选项：

- **正弦嵌入。** 类似 Transformer 的位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。经过一个 MLP，广播到网络中。
- **FiLM / 组归一化条件化。** 将嵌入投影到每个通道的缩放/偏置（FiLM），应用于每个模块。

我们的玩具代码使用正弦嵌入 → 拼接。生产环境的 U-Net 使用 FiLM。

## 陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 的默认选择，但余弦调度（Nichol & Dhariwal, 2021）在相同计算量下能获得更好的 FID。如果质量停滞不前，可以切换调度。
- **时间步嵌入很脆弱。** 将原始的 `t` 作为浮点数传递可用于玩具一维问题，但在图像上会失败；务必使用合适的嵌入。
- **V 预测 vs ε 预测。** 在狭窄区间（非常小或非常大的 t）内，`ε` 的信噪比很低。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导。** 在推理时，同时计算条件和无条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。该内容在第 08 课中涵盖。
- **1000 步很多。** 生产环境使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。参见第 12 课。

## 用途

| 角色 | 2026 年的典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜空间扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜空间扩散 | 时空 DiT（Sora, Veo, WAN） |
| 音频潜空间扩散 | Encodec + 扩散Transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM, RFdiffusion, AlphaFold3） |

扩散是通用的生成式主干。流匹配（第 13 课）是 2024-2026 年的竞争者，通常在相同质量下推理速度更快。

## 交付

保存 `outputs/skill-diffusion-trainer.md`。技能接受一个数据集 + 计算预算，输出：调度（线性/余弦/sigmoid）、预测目标（ε/v/x）、步数、引导尺度、采样器家族以及评估协议。

## 练习

1. **简单。** 将 `code/main.py` 中的 T 从 40 改为 10。样本质量（输出的可视化直方图）如何下降？在哪个 T 值时双峰结构崩溃？
2. **中等。** 从 ε 预测切换为 v 预测。重新推导逆向步骤。比较最终的样本质量。
3. **困难。** 添加无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，训练时 10% 的概率丢弃它，并在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 下测量条件模式命中率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 前向过程 | "加噪" | 固定的马尔可夫链 `q(x_t | x_{t-1})`，用于破坏数据。 |
| 逆向过程 | "去噪" | 学习到的链 `p_θ(x_{t-1} | x_t)`，用于重建数据。 |
| β 调度 | "噪声阶梯" | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；使得可以从 `x_0` 直接得到 `x_t` 的闭合形式。 |
| 简单损失 | "噪声的 MSE" | `||ε - ε_θ(x_t, t)||²`；所有变分推导都归结为此。 |
| ε 预测 | "预测噪声" | 输出是加入的噪声；标准 DDPM。 |
| V 预测 | "预测速度" | 输出是 `α·ε - σ·x`；在不同 t 上条件化更好。 |
| DDPM | "那篇论文" | Ho 等人 2020 年；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同的训练目标。 |
| 无分类器引导 | "CFG" | 混合条件和无条件噪声预测，以增强条件化效果。 |

## 生产环境提示：扩散推理是一个步数问题

DDPM 论文使用 T=1000 步逆向。没有哪个生产系统会直接部署这样。每个真实的推理栈都会选择以下三种策略之一——每种策略都清晰地映射到生产环境中关于“延迟从何而来”的讨论：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。直接替换逆向循环；训练好的 `ε_θ` 权重保持不变。延迟降低 20-50 倍。
2. **蒸馏。** 训练一个学生模型用更少的步骤匹配教师模型：渐进式蒸馏（2→1）、一致性模型（任意→1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降低 5-10 倍，需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟降低约 2 倍。可与 (1) 和 (2) 叠加使用。

对于生产环境的扩散服务器，预算讨论与生产文献中关于 LLM 的描述相同：延迟是 `步数 × 每步成本 + VAE 解码`，吞吐量是 `批量大小 × (步数 × 每步成本)^-1`。TTFT 很小（一步）；TPOT 等价物就是完整的响应时间，因为图像生成从用户视角来看是“一次性”的。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少的步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最清晰的配方。
