# LLaVA 与视觉指令微调

> LLaVA（2023年4月）是地球上被复制最多的多模态架构。它用两层MLP取代了BLIP-2的Q-Former，用朴素的token拼接取代了Flamingo的门控交叉注意力，并在由GPT-4仅从文本描述生成的15.8万条视觉指令对话上进行了训练。任何在2023年至2026年间构建VLM的从业者都构建了LLaVA的某种变体。LLaVA-1.5加入了AnyRes。LLaVA-NeXT提高了分辨率。LLaVA-OneVision将单图、多图和视频统一到一个方案中。本课将解读该方案，实现投影器，并解释为什么“更简单的方案胜出”。

**类型：** 构建
**语言：** Python（标准库，投影器 + 指令模板构建器）
**前置要求：** 阶段12·02（CLIP），阶段11（LLM工程——指令微调）
**时间：** 约180分钟

## 学习目标

- 构建一个两层MLP投影器，将ViT块嵌入（维度1024）映射到LLM的嵌入维度（维度4096）。
- 走通LLaVA的两阶段方案：（1）在55.8万条描述对上对齐投影器，（2）在15.8万条GPT-4生成的对话上进行视觉指令微调。
- 构建LLaVA格式的提示，包含图像token占位符、系统提示以及用户/助手对话。
- 解释尽管Q-Former在token预算上占优，但社区为何从Q-Former转向MLP。

## 问题

BLIP-2的Q-Former（第12.03课）将图像压缩到32个token。简洁、高效、对基准测试友好。但它有两个问题。

第一，Q-Former是可训练的，但其损失函数并非最终任务。阶段1训练ITC+ITM+ITG。阶段2训练LM损失。查询学习到某种中间表示，然后LLM必须对其进行解码。信息在瓶颈中丢失。

第二，Q-Former有1.88亿参数，在2023年LLaVA的规模下，你需要与目标LLM共同设计它。更换LLM，重新训练Q-Former。更换视觉编码器，重新训练。每种组合都是一个独立的研发项目。

LLaVA的答案简单得令人尴尬：取ViT的576个块token，分别通过一个两层MLP（`1024 → 4096 → 4096`），然后将全部576个token送入LLM的输入序列。没有瓶颈。没有基于奇怪目标的预训练。只需在直接的LM损失上训练MLP。

数据从哪里来？LLaVA的第二个洞察：使用GPT-4（仅文本）生成指令数据。将图像的COCO描述和边界框数据喂给GPT-4，让它生成对话、描述和复杂推理问题。免费获得15.8万条指令-响应对。无需人工标注。

结果：一个在8张A100上运行一天的VLM，在MMMU上击败了Flamingo，并发布了社区可以扩展的开放检查点。到2023年底，它已衍生出50多个分支。

## 概念

### 架构

LLaVA-1.5 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段1冻结，阶段2可选解冻）。
- 投影器：两层MLP，带GELU激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是Llama-3.1-8B）。

图像+文本提示的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用LLM上下文的576个token。在2048上下文下，文本还剩1472个token。在32k上下文中，这只是一个舍入误差。

### 阶段1：投影器对齐

冻结ViT。冻结LLM。仅训练两层MLP。数据集：55.8万张图像-描述对（LAION-CC-SBU）。损失：对描述的语言建模，以投影后的图像token为条件。

在单轮epoch、批大小128的情况下，几小时即可完成。投影器学会将ViT空间映射到LLM空间。没有任务特定的监督。

### 阶段2：视觉指令微调

解冻投影器（仍可训练）。解冻LLM（通常完全解冻，有时用LoRA）。在15.8万条视觉指令对话上训练。

指令数据是诀窍。Liu等人通过以下方式生成：
1. 取一张COCO图像。
2. 提取文本描述（5条人工描述 + 边界框列表）。
3. 发送给GPT-4，使用三种提示模板：
   - 对话：“生成一段关于这张图像的用户和助手之间的来回对话。”
   - 详细描述：“给出关于这张图像的丰富、详细的描述。”
   - 复杂推理：“提出一个需要对图像进行推理的问题，然后回答它。”
4. 解析GPT-4的输出，得到（指令，响应）对。

所有这些都不直接接触图像——只接触文本描述。GPT-4会幻觉出看似合理的图像内容。有些噪声，但有效：15.8万条对话足以解锁对话能力。

### 社区为何复制它

- 无需调优阶段1特定的损失。全程使用LM损失。
- 投影器训练几小时而非几天。
- LLM可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据管线使用GPT-4，为新领域重新生成成本低。

### LLaVA-1.5 和 LLaVA-NeXT

LLaVA-1.5（2023年10月）加入了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混合进指令微调。
- 更好的系统提示。
- 2048 → 32k上下文。

LLaVA-NeXT（2024年1月）加入了：
- AnyRes：将高分辨率图像分割成2x2或1x3的336x336裁剪块网格，外加一个全局低分辨率缩略图。每个裁剪块变成576个token；每张图像总共约2880个视觉token。OCR和图表任务大幅提升。
- 使用ShareGPT4V（高质量的GPT-4V描述）改进指令数据混合。
- 更强的基座LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第12.08课详细介绍了OneVision。简而言之：相同的投影器，但以课程方式训练，涵盖单图像、多图像和视频，共享视觉token预算。

### 与Q-Former的比较

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每张图像的视觉token数 | 32 | 576（基础）或2880（AnyRes） |
| 可训练参数 | 1.88亿 + LLM | 4000万 + LLM |
| 阶段1损失 | ITC+ITM+ITG | 仅LM |
| LLM替换 | 需要重新训练 | 最小化重新训练即可替换 |
| 多图像 | 笨拙 | 自然（拼接） |
| 视频 | 笨拙 | 自然（逐帧拼接） |
| Token预算 | 小 | 大 |

MLP在简单性和token灵活性上胜出。Q-Former在token预算上胜出。到2023年底，token预算已不再是制约因素（LLM上下文增长到32k-128k+），简单性主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位符token。在分词之前，它被替换为576个视觉token（AnyRes下为2880个）。分词器看到的是一个比训练时稍长的序列，但LLM能够处理这种新颖输入，因为阶段1教会了它。

### 参数经济性

LLaVA-1.5-7B的分解：
- CLIP ViT-L/14 @ 336：3.03亿（阶段1冻结，阶段2通常解冻）。
- 投影器（2个线性层）：约2200万可训练参数。
- Llama-7B：70亿。
- 总计：73亿参数。阶段2可训练参数：全部70亿 + 2200万投影器。

阶段2的训练成本：在8xA100上约20小时。这是关键数字——一天，一个节点，可复现。这就是LLaVA传播的原因。

## 使用它

`code/main.py` 实现了：

1. 两层MLP投影器（玩具规模下维度16 → 32 → 32），纯Python实现。
2. 提示构建管线：系统提示 + `<image>` 替换为N个投影后的token + 用户对话 + 助手生成占位。
3. 可视化576个视觉token块在LLM上下文中占用的百分比（相对于2k / 32k / 128k上下文）。

## 交付

本课生成 `outputs/skill-llava-vibes-eval.md`。给定一个LLaVA系列检查点，它运行一个10提示的vibe评估套件（3个描述、3个VQA、2个推理、2个拒答），并报告一个人类可读的得分卡。这不是基准测试；而是确认投影器和LLM连接良好的冒烟测试。

## 练习

1. 计算在 `1024 → 4096 → 4096` 下两层MLP投影器的可训练参数数量。包括GELU和偏置，它占LLaVA-13B的多少比例？

2. 为“拒答”案例构建一个LLaVA提示——图像中包含个人隐私。写出期望的助手响应。为什么LLaVA应该零样本拒绝回答？需要什么样的训练数据来强化这种拒绝？

3. 阅读LLaVA-NeXT博客中的AnyRes部分。计算1344x672图像在AnyRes下的视觉token数量。与336x336的基础576个token进行比较。

4. LLaVA阶段1的投影器在描述上使用LM损失进行训练。如果跳过阶段1直接进入阶段2（视觉指令微调）会发生什么？引用Prismatic VLMs的消融实验（arXiv:2402.07865）来回答。

5. LLaVA-Instruct-150k使用GPT-4和COCO描述生成指令。对于一个新领域（医学X光、卫星图像），描述生成领域指令的四步数据管线。每一步可能出现什么问题？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|-----------|
| Projector | “MLP桥” | 两层带GELU的MLP，将ViT维度映射到LLM维度 |
| Image token | “<image>占位符” | 推理前被N个投影后的视觉token替换的提示标记 |
| Visual instruction tuning | “LLaVA阶段2” | 在GPT-4生成的（图像、指令、响应）三元组上训练 |
| Stage 1 alignment | “投影器预训练” | 冻结ViT和LLM，使用描述上的LM损失训练投影器 |
| AnyRes | “多裁剪平铺” | 将高分辨率图像分割成平铺网格，并拼接每个平铺的视觉token |
| LLaVA-Instruct | “GPT-4生成的” | 从COCO描述 + GPT-4综合得到的15.8万条指令-响应对 |
| Vision encoder freeze | “主干冻结” | 阶段1 CLIP权重不更新，阶段2有时也不更新 |
| ShareGPT4V | “更好的描述” | GPT-4V生成的100万条密集描述，用于更高质量的对齐 |
| VQA | “视觉问答” | 关于图像的自由形式问答任务 |
| Prismatic VLMs | “设计空间论文” | Karamcheti 2024消融实验，系统测试投影器和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。
