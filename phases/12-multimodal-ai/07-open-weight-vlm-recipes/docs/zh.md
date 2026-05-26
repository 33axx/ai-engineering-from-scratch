# 开放权重 VLM 配方：什么才是真正重要的

> 2024-2026 年的开放权重 VLM 文献是一片消融实验表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明了详细的人工标注描述优于 GPT-4V 蒸馏。Cambrian-1 进行了 20 多项编码器比较。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练配方。在这些噪音中，有少数结果在论文间保持一致：图像编码器比连接器架构更重要，数据混合比两者都重要，详细的人工标注描述优于蒸馏合成数据。本课将解读这些表格，让你无需亲自阅读。

**类型:** 学习 + 实验
**语言:** Python (标准库, 消融表格解析器 + 配方选择器)
**前置条件:** 阶段 12 · 05 (LLaVA 基线)
**时间:** ~180 分钟

## 学习目标

- 命名 VLM 的五轴设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 消融表格，并预测哪个旋钮会影响给定的基准。
- 根据计算预算和任务组合，为新 VLM 选择一个配方（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细的人工标注描述优于 GPT-4V 蒸馏。

## 问题

存在数百个开放权重 VLM。大多数“好”与“最先进”之间的差距并非来自架构，而是来自数据、分辨率调度和编码器选择。当你的模型表现不佳时，知道先该转动哪个旋钮，可以避免 500 万 GPU 小时的错误。

2023 年的浪潮（LLaVA-1.5, InstructBLIP, MiniGPT-4）依赖于描述对预训练 + LLaVA-Instruct-150k。良好的基线。在 MMMU 上最高约为 35%。

2024 年的浪潮（MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs）进行了详尽的消融实验。结果出人意料且实用。

## 概念

### 五轴设计空间

Idefics2 (Laurençon 等人, 2024) 明确了这些轴：

1. **图像编码器**：CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上各不相同。
2. **连接器**：MLP (2-4 层), Q-Former (32 查询 + 交叉注意力), Perceiver Resampler (64 查询), C-Abstractor (卷积 + 双线性池化)。
3. **语言模型**：Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5。LLM 大小是主要参数成本。
4. **训练数据**：描述对 (CC3M, LAION), 交错数据 (OBELICS, MMC4), 指令数据 (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron)。
5. **分辨率调度**：固定 224/336/448, AnyRes, 原生动态。在训练过程中渐增或恒定。

每个生产级 VLM 在每个轴上都有一个选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而不是你选择了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 切换到 SigLIP SO400m/14 在 MMMU 上增加了 3 个以上百分点。将连接器从 MLP 切换为 Perceiver Resampler 增加不到 1 个点。Idefics2 复现了这一结果：SigLIP > CLIP，在相同 token 数量下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的“Cambrian Vision Encoders Match-Up”（Tong 等人, 2024）在视觉中心基准 (CV-Bench) 上测试了 20 多种编码器。排行榜顶端是 DINOv2 和 SigLIP 的混合；CLIP 位于中游；ImageBind 和 ViT-MAE 较低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距约为 CV-Bench 上的 5-7 个点。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 密集特征，有时会与 DINOv2 ViT-g/14 特征拼接（Cambrian 的“Spatial Vision Aggregator”就是这样做的）。

### 轴 2：连接器设计无关紧要

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了相同结论：在固定视觉 token 数量的情况下，连接器架构几乎不重要。在相同的 token 预算下，对均值池化的 patch 使用两层 MLP 与 32 查询的 Q-Former 相比，性能差异在 1 个点以内。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 计算 = 性能更好，直到某个点后收益递减。每张图像 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开放 VLM 的最佳点。2048 以上仅对文档和图表有帮助。

Q-Former vs MLP 是成本问题，而非质量问题：Q-Former 无论图像分辨率如何，token 数上限为 32-64；MLP 则输出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异只是噪音。

### 轴 3：LLM 大小设定天花板

将 LLM 从 7B 加倍到 13B，在每篇 VLM 论文的 MMMU 上稳定增加 2-4 个点。在 70B 时，大多数基准都会饱和。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能提供输入，不能代替推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上表现优异：语言大脑很大。7B VLM 无法通过巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细的人工标注描述优于蒸馏

Molmo + PixMo（Deitke 等人, 2024）是每个人都应该读的 2024 年成果。Allen AI 让人类标注员通过 1-3 分钟的密集语音转文本方式描述图像，生成了 712K 张密集标注的图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败了 Llama-3.2-90B-Vision。差异不在于架构——而在于标注质量。详细的用人标注描述每张图像包含的信息量是简短网络描述的 5-10 倍，并且在事实基础上保持准确，而 GPT-4V 蒸馏则会幻觉。

ShareGPT4V（Chen 等人, 2023）和 Cauldron（Idefics2）遵循了同样的策略，混合了人工和 GPT-4V 标注。趋势很明确：对于 2026 年的前沿模型，标注密度 > 标注数量 > 蒸馏的便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融实验：384 -> 448 增加 1-2 个点。448 -> 980 配合图像分割（AnyRes）在 OCR 基准上再增加 3-5 个点。固定分辨率训练在中等精度处趋于平稳；分辨率渐增（从 224 开始，结束于 448 或原生分辨率）训练更快且最终效果更好。

Cambrian-1 进行了分辨率与 token 数量的权衡：在固定计算量下，你可以选择低分辨率更多 token 或高分辨率更少 token。高分辨率在 OCR 方面胜出；低分辨率更多 token 在通用场景理解方面胜出。

2026 年的生产配方：第一阶段在 384 固定分辨率下训练，第二阶段使用动态分辨率，对 OCR 密集型任务最高可达 1280。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等人, 2024）是一篇控制了所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只改变一个轴。结果：

- 每张图像的视觉 token 数量解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他一切（数据混合、调度器、学习率）约占 15%。

这是一个粗略的分解，但它是文献中对“我应该先消融哪个”最清晰的回答。

### 2026 年的选择器

根据现有证据，2026 年新项目的默认开放 VLM 配方是：

- **编码器**：SigLIP 2 SO400m/14，原生分辨率配合 NaFlex，如果需要分割/定位，则与 DINOv2 ViT-g/14 的密集特征拼接。
- **连接器**：patch token 上的两层 MLP。除非你受到 token 数量限制，否则跳过 Q-Former。
- **LLM**：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于成本控制，70B 用于质量，根据目标延迟选择。
- **数据**：PixMo + ShareGPT4V + Cauldron，并用任务特定的指令数据进行补充。
- **分辨率**：动态（每边长最小 256，最大 1280 像素）。
- **调度**：第一阶段对齐（仅投影器），第二阶段全微调，第三阶段任务特定微调。

每一项默认选择都可以追溯到本课末尾引用的论文中测量的消融实验。

## 使用它

`code/main.py` 是一个消融表格解析器和配方选择器。它编码了 MM1 和 Idefics2 的消融表格（压缩版），并允许你查询：

- “给定预算 X 和任务 Y，哪个配方获胜？”
- “如果在 7B Llama 上将 SigLIP 换成 CLIP，预期的 MMMU 差异是多少？”
- “为了 80% 置信度的答案，我应该先消融哪个轴？”

输出是一个排序的配方列表，包含预期的基准差异和“先消融”建议。

## 交付

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，它会生成一个完整的配方（编码器、连接器、LLM、数据混合、分辨率调度），并附有证明每个选择的消融实验引用。防止工程师每次启动新的 VLM 项目时重新发明 Idefics2 消融表格。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定 2B LLM 且预算为 5000 万张图像的情况下，哪个编码器胜出？如果换成 13B LLM，答案会改变吗？为什么？

2. Cambrian-1 发现，拼接 DINOv2 + SigLIP 在视觉中心基准上优于单独使用任意一个，但在 MMMU 上未增加信号。预测哪些基准会提升，哪些保持平坦。

3. 你的目标是在 2B LLM 上运行移动 UI Agent。选择编码器、连接器、分辨率和数据混合。用具体的消融表格证明每个选择。

4. Molmo 发布了 4B 和 72B 模型。4B 与封闭的 7B VLM 相当；72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。这对 LLM 大小平台假设有什么启示？

5. 设计一个消融表格，在 7B VLM 上隔离数据混合质量与编码器质量。最少需要多少次训练运行？提出四个轴的设置。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 消融 (Ablation) | “转动一个旋钮” | 训练多个运行，仅在设计空间的一个轴上不同，保持其他一切不变 |
| 连接器 (Connector) | “桥” / “投影器” | 可训练模块，将视觉编码器输出映射到 LLM 的 token 空间（MLP, Q-Former, Perceiver） |
| 详细人工标注描述 (Detailed human caption) | “密集标注” | 多句人工撰写的描述（通常 80-300 token），比网络替代文本更丰富 |
| 蒸馏 (Distillation) | “GPT-4V 标注” | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / 动态分辨率 | “高分辨率路径” | 通过分块或 M-RoPE 将大于编码器原生分辨率的图像输入的策略 |
| 分辨率渐增 (Resolution ramp) | “课程学习” | 训练调度从低分辨率开始，逐渐增加，加速对齐学习 |
| 视觉中心基准 (Vision-centric bench) | “CV-Bench / BLINK” | 强调细粒度视觉感知而非语言密集型推理的评估 |
| PixMo | “Molmo 的数据” | Allen AI 的 712K 张密集标注图像数据集；人工语音转录为密集标注 |

## 延伸阅读

- [McKinzie 等人 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等人 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等人 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等人 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
