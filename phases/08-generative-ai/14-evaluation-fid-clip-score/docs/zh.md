# 评估——FID、CLIP 分数、人类偏好

> 每个生成模型排行榜都引用了 FID、CLIP 分数以及人类偏好擂台的胜率。每个数字都有一种失败模式，决意做手脚的研究人员可以钻空子。如果你不知道这些失败模式，你就无法区分真正的改进和作弊跑分。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 8 · 01（分类法），阶段 2 · 04（评估指标）
**时间：** 约 45 分钟

## 问题

生成模型的好坏取决于*样本质量*和*条件遵循度*。两者都没有闭式度量。你的模型需要渲染 10,000 张图像；有人必须给它们分配数字；你必须在不同模型家族、不同分辨率、不同架构间信任这些数字。三个指标在 2014–2026 年的淘汰赛中留存下来：

- **FID（Fréchet Inception Distance）。** 真实分布与生成分布在 Inception 网络特征空间中的距离。越低越好。
- **CLIP 分数。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。用于衡量提示遵循度。
- **人类偏好。** 让两个模型在相同提示词上直接对比，由人类（或 GPT-4 类模型）选出更好的结果，聚合为 Elo 分数。

你还会看到：IS（Inception 分数，基本已弃用）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个新指标都在修正前一指标的某个失败点。

## 概念

![FID、CLIP 和偏好：三个维度，不同的失败模式](../assets/evaluation.svg)

### FID——样本质量

Heusel 等人 (2017)。步骤：

1. 对 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 为每个池拟合一个高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯分布之间的 Fréchet 距离。越低表示分布越相似。

失败模式：
- **对小 N 有偏。** FID 是对特征分布求均方——小 N 会低估协方差，给出虚假的低 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。远离 ImageNet 的领域（人脸、艺术、文字图像）会产生无意义的 FID。使用领域特定的特征提取器。
- **作弊。** 过度拟合 Inception 先验会导致 FID 偏低，而视觉质量却没有提升。可用 CMMD（见下文）击败它。

### CLIP 分数——提示遵循度

Radford 等人 (2021)。对于一张生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 30k 张生成图像取平均 → 得到一个可在模型之间比较的标量。

失败模式：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（例如“蓝色球体上的红色立方体”常常失败）。模型可以在 CLIP 分数上排名靠前，但实际上并不真正遵循复杂提示。
- **短提示词偏倚。** 短提示词在现实中有更多的 CLIP-图像匹配。长提示词的 CLIP 分数会机械地更低。
- **提示词作弊。** 在提示词中加入“高画质、4k、杰作”会抬高 CLIP 分数，但并未改善图像-文本绑定。

CMMD（Jayasumana 等人，2024）修正了其中一些问题：使用 CLIP 特征代替 Inception，使用最大均值差异（MMD）代替 Fréchet。能更好地检测细微的质量差异。

### 人类偏好——事实真相

选择一个提示词池。用模型 A 和模型 B 生成图像。将成对结果展示给人类（或强大的 LLM 评判者）。汇总胜局为 Elo 或 Bradley-Terry 分数。常用基准：

- **PartiPrompts（Google）：** 1,600 个多样化提示词，12 个类别。
- **HPSv2：** 107k 条人类标注，广泛用作自动代理。
- **ImageReward：** 137k 条提示-图像偏好对，MIT 许可证。
- **PickScore：** 在 Pick-a-Pic 2.6M 偏好数据上训练。
- **Chatbot-Arena 形式的图像竞技场：** https://imagearena.ai/ 等。

失败模式：
- **评判者方差。** 非专家与专家有不同的偏好。两种都应使用。
- **提示词分布。** 精心挑选的提示词会偏向某一模型家族。务必记录。
- **LLM 评判者的奖励破解。** GPT-4 评判者会被好看但错误的输出所迷惑。与人类评判交叉验证。

## 结合使用

一份生产级评估报告应包含：

1. 在 10–30k 样本上与保留的真实分布计算的 FID（样本质量）。
2. 相同样本与其提示词之间的 CLIP 分数 / CMMD（遵循度）。
3. 在与前一个模型的盲测擂台中的胜率（整体偏好）。
4. 失败模式分析：随机抽取 50 个输出，标记已知问题（手部解剖、文字渲染、目标数量一致性）。

任何一个单独的指标都是谎言。三个相互印证的指标加上定性分析才构成一个论断。

## 动手构建

`code/main.py` 在合成的“特征向量”（我们使用 4 维向量代替 Inception 特征）上实现了 FID、类 CLIP 分数和 Elo 聚合。你会看到：

- 在小 N 和大 N 下计算 FID——偏倚的存在。
- “CLIP 分数”作为特征池之间的余弦相似度。
- 从合成的偏好流中更新 Elo 的规则。

### 步骤 1：四行代码的 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 时的 FID。** 在 N<10k 时启发式结果不可靠。报告小 N 下 FID 的论文是在作弊。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。只在匹配的分辨率下比较。
- **只报告一次随机种子。** 至少运行 3 个随机种子。报告标准差。
- **通过负提示词膨胀 CLIP 分数。** 某些流程通过过度拟合提示词来提升 CLIP。检查视觉饱和度。
- **因提示词重叠导致的 Elo 偏倚。** 如果两个模型在训练期间都见过某个基准提示词，Elo 就没有意义了。使用保留的提示词集。
- **人类评估付费人群偏倚。** Prolific、MTurk 的标注者偏向年轻/技术爱好者。应与招聘的艺术/设计专家混合。

## 投入使用

2026 年的生产级评估协议：

| 支柱 | 最低要求 | 推荐 |
|------|---------|-------------|
| 样本质量 | 针对保留的真实集，10k 样本上的 FID | + 5k 样本上的 CMMD + 按类别子集的 FID |
| 提示遵循度 | 30k 样本上的 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格的问答 |
| 偏好 | 200 个盲测对 vs 基线 | + 2000 个成对人类 + LLM 评判 + Chatbot Arena |
| 失败分析 | 50 个手动标记样本 | 500 个手动标记样本 + 自动安全分类器 |

报告中包含所有四个支柱 = 论断。只有其中一个 = 营销。

## 交付

保存 `outputs/skill-eval-report.md`。技能接收一个新模型检查点和基线，输出完整的评估计划：样本量、指标、失败模式探针、签署条件。

## 练习

1. **简单。** 运行 `code/main.py`。比较相同合成分布在 N=100 和 N=1000 时的 FID。报告偏倚大小。
2. **中等。** 从合成的 CLIP 风格特征实现 CMMD（公式参见 Jayasumana 等人，2024）。与 FID 相比，在对质量差异的敏感性上有何差异？
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的一个子集取 1000 个图像-提示词对，在一个小型基于 CLIP 的评分器上微调偏好，并测量其与保留集的一致性。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 真实与生成 Inception 特征的高斯拟合的 Fréchet 距离。 |
| CLIP 分数 | "文本-图像相似度" | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代品" | CLIP 特征 MMD；偏倚更小，无高斯假设。 |
| IS | "Inception 分数" | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已弃用。 |
| HPSv2 / ImageReward / PickScore | "学到的偏好代理" | 在人类偏好上训练的小型模型；用作自动评判者。 |
| Elo | "国际象棋评分" | 成对胜局的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示集" | Google 策划的 1,600 个提示词，12 个类别。 |
| FD-DINO | "自监督替代" | 使用 DINOv2 特征的 FD；更适合非 ImageNet 领域。 |

## 生产备注：评估也是一种推理负载

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于单个 L4 上 50 步的 SDXL base（1024²），这大约是 11 小时的单请求推理。评估预算真实存在，并且其框架正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **大批量，忽略延迟。** 离线评估 = 使用能装入内存的最大尺寸进行静态批处理。在 80GB H100 上，`pipe(...).images` 配合 `num_images_per_prompt=8` 的 wall-clock 速度比单请求快 4–6 倍。
- **缓存真实特征。** 对真实参考集执行的 Inception（FID）或 CLIP（CLIP 分数、CMMD）特征提取*只运行一次*，存储为 `.npz` 文件。不要在每次评估时重新计算。

对于 CI / 回归门控：在每个 PR 上对 500 样本子集运行 FID + CLIP 分数（约 30 分钟）；每晚运行完整的 10k FID + HPSv2 + Elo。

## 进一步阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失败模式综述。
