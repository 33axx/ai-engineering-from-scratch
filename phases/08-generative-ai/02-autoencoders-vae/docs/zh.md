# 自编码器与变分自编码器 (VAE)

> 一个普通的自编码器压缩然后重建。它记住了。它不会生成。加上一个技巧——强制编码看起来像高斯分布——你就得到了一个采样器。正是这个单一技巧，即 `z = μ + σ·ε` 的重参数化，才使得你在 2026 年使用的每一个潜在扩散和流匹配图像模型在输入端都有一个 VAE。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 3 · 02（反向传播），阶段 3 · 07（CNN），阶段 8 · 01（分类法）  
**时间：** 约 75 分钟

## 问题

将一个 784 像素的 MNIST 数字压缩成 16 个数字的编码，然后重建。一个普通的自编码器能很好地完成重建 MSE，但编码空间是一团乱麻。在编码空间中随机选取一个点，解码后只能得到噪声。它没有采样器。它只是一个伪装成生成模型的压缩模型。

你实际想要的是：(a) 编码空间是一个干净、平滑的分布，你可以从中采样——例如各向同性高斯分布 `N(0, I)`，(b) 解码任何样本都会产生一个合理的数字，(c) 编码器和解码器仍然压缩得很好。三个目标，一个架构，一个损失函数。

Kingma 在 2013 年提出的 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z` 来解决这个问题。在推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚正是强制编码空间结构化的原因。

到 2026 年，VAE 很少独立部署——在原始图像质量上它们已被扩散模型取代——但它们仍然是每个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了使用的每个图像管道中不可见的第一层。

## 概念

![自编码器与 VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`, `x̂ = decoder(z)`, 损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 中采样是不可微的。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的一个确定性函数，加上一个非参数噪声——梯度可以流过 `μ` 和 `σ`。

**损失函数。** 证据下界 (ELBO)，包含两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项将 `x̂` 推向 `x`。KL 项将 `q(z|x)` 推向先验。它们互相权衡。小的 β (<1) 产生更锐利的样本，编码空间不那么高斯。大的 β (>1) 产生更干净的编码空间，但样本更模糊。β-VAE (Higgins 2017) 使这个旋钮闻名，并开启了解耦表示学习的研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——不像扩散那样需要迭代采样。

## 构建它

`code/main.py` 实现了一个微型 VAE，不使用 numpy 或 torch。输入是 8 维合成数据，来自 8 维空间中的 2 分量高斯混合。编码器和解码器都是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失函数和手工编写的反向传播。这不是生产代码——而是教学。

### 第 1 步：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出不受约束（σ 的 softplus 是个陷阱——当 σ ≈ 0 时梯度会消失）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

由于两个分布都是高斯分布，KL 散度采用精确的闭式解。不要用数值积分。2026 年还有人用蒙特卡洛估计 KL 散度——这无故慢了 3 倍。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验坍塌。** KL 项将 `q(z|x) → N(0, I)` 驱动得过猛，以至于 `z` 不携带任何关于 `x` 的信息。解决方法：β 退火（从 β=0 开始，逐渐增加到 1）、自由位、或跳过不活跃维度上的 KL。
- **模糊样本。** 高斯解码器似然函数意味着 MSE 重建，这对于 L2 损失（均值）是贝叶斯最优的——一组合理的数字的均值就是一个模糊的数字。解决方法：离散解码器（VQ-VAE、NVAE），或者只将 VAE 作为编码器，在潜在变量上堆叠扩散模型（Stable Diffusion 就是这么做的）。
- **β 太大且开始得太早。** 见后验坍塌。从 β≈0.01 开始并逐渐增加。
- **潜在维度太小。** MNIST 用 16 维，ImageNet 256² 用 256 维，ImageNet 1024² 用 2048 维。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积 32 倍下采样，通道数 32 倍）。

## 使用它

2026 年 VAE 的堆叠：

| 情况 | 选择 |
|------|------|
| 用于扩散的图像潜在编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频潜在编码器 | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| 视频潜在变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 用于 Transformer 建模的离散潜在变量 | VQ-VAE、RVQ (ResidualVQ) |
| 用于生成的连续潜在变量 | 普通 VAE，然后在潜在空间上调节流/扩散模型 |

一个潜在扩散模型就是一个 VAE，在编码器和解码器之间加入一个扩散模型。VAE 负责粗略压缩，扩散模型负责繁重工作。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）也是同样的模式。

## 交付它

保存 `outputs/skill-vae-trainer.md`。

技能接受：数据集简介 + 潜在维度目标 + 下游用途（重建、采样或潜在扩散输入），并输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜在维度、解码器似然函数（高斯 vs 分类）、以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终的重建 MSE 和 KL。对于你的合成数据，哪个 β 是帕累托最优的？
2. **中等。** 用伯努利似然函数（交叉熵损失）替换高斯解码器似然函数。在同一合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为一个小型 VQ-VAE：用一个包含 K=32 个条目的码本中的最近邻查找替换连续的 `z`。比较重建 MSE，并报告有多少码本条目被使用（码本坍塌是真实存在的）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，学习 MSE。不是生成式的。 |
| VAE | 带采样器的自编码器 | 编码器输出一个分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ 重建 - KL[q(z|x) \|\| p(z)]`；当 `q = p(z|x)` 时紧致。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点重写为确定性 + 纯噪声。使采样可反向传播。 |
| 先验 | `p(z)` | 潜在变量的目标分布，通常是 `N(0, I)`。 |
| 后验坍塌 | “KL 项赢了” | 编码器忽略 `x`，输出先验；解码器必须幻想。 |
| β-VAE | 可调的 KL 权重 | `损失 = 重建 + β·KL`。β 越大解耦性越好但样本越模糊。 |
| VQ-VAE | 离散潜在变量 | 用最近码本向量替换连续的 `z`；支持 Transformer 建模。 |

## 生产注意：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 管道中，每次请求 VAE 被调用两次——一次用于编码（如果做 img2img / inpainting），一次用于解码。在 1024² 分辨率下，解码器传播往往是整个管道中最大的激活内存峰值，因为它将 `128×128×16` 的潜在变量上采样回 `1024×1024×3`。两个实际后果：

- **对解码进行分片或分块。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以小的接缝伪影换取 `O(tile²)` 内存而非 `O(H·W)`。对于消费级 GPU 上的 1024²+ 分辨率是必需的。
- **bf16 解码器，但最终缩放时用 fp32 数值。** SD 1.x VAE 以 fp32 发布，在 1024²+ 转换为 fp16 时*静默地产生 NaN*。SDXL 提供了 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体或使用 bf16。

## 进一步阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦的 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
