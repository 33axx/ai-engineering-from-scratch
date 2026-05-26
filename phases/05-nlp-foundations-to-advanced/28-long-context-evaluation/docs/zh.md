# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称拥有1000万 token 的上下文窗口。在100万 token 时，8针 MRCR 降至26.3%。宣称 ≠ 可用。长上下文评估能告诉你实际部署的模型真正具备的能力。

**类型：** 学习
**语言：** Python
**前置条件：** 阶段5·13（问答系统），阶段5·23（分块策略）
**时长：** 约60分钟

## 问题

你有一份200页的合同。模型声称拥有100万 token 的上下文窗口。你把合同粘贴进去并提问："终止条款是什么？"模型回答了——但回答的是封面页的内容，因为终止条款位于12万 token 深处，超出了模型实际能够关注的范围。

这就是2026年的上下文容量鸿沟。规格表上写着100万或1000万，但实际可用部分只有60-70%，而且"可用"程度取决于具体任务。

- **检索（单针大海捞针）：** 前沿模型在宣称的最大长度下几乎完美。
- **多跳/聚合：** 大多数模型在超过约12.8万 token 后性能急剧下降。
- **分散事实推理：** 这是最先失败的任务。

长上下文评估正是衡量这些维度的。本课程将介绍各项基准、它们实际衡量什么，以及如何为你自己的领域构建定制化的"针"测试。

## 概念

![NIAH 基线、RULER 多任务、LongBench 整体评估](../assets/long-context-eval.svg)

**大海捞针（NIAH，2023年）。** 在长上下文中，将一个事实（如"魔法词是 pineapple"）放置在受控深度位置。要求模型检索该事实。遍历深度×长度。这是最早的长上下文基准。前沿模型现在已能完全胜任此项测试；它是一个必要但非充分的基线。

**RULER（Nvidia，2024年）。** 涵盖4大类别的13种任务：检索（单键/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频）、问答。可配置上下文长度（4k 到 128k+）。能揭示那些通过 NIAH 测试但在多跳任务上失败的问题。在2024年的发布中，17个宣称拥有32k+上下文长度的模型中，只有一半能在32k长度下保持质量。

**LongBench v2（2024年）。** 503道选择题，上下文长度从8k到200万 token，涵盖六个任务类别：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。这是评估真实世界长上下文行为的生产级基准。

**MRCR（多轮指代消解）。** 大规模的多轮指代消解。有8针、24针、100针变体。揭示了模型在注意力退化前能同时处理多少个事实。

**NoLiMa。** "非词汇性针"。针和查询之间没有字面上的词汇重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 将多个文档拼接起来，然后从其中任意一个文档提问。测试选择性注意力能力。

**BABILong。** 将 bAbI 推理链嵌入到无关的"干草堆"文本中。测试的是"干草堆中的推理能力"，而不仅仅是检索能力。

### 实际应该报告的内容

- **宣称的上下文窗口。** 规格表上的数字。
- **有效检索长度。** 在某个阈值（例如90%）下通过 NIAH 测试的长度。
- **有效推理长度。** 在该阈值下通过多跳或聚合测试的长度。
- **性能衰减曲线。** 针对每种任务类型，绘制准确率与上下文长度的关系图。

你的规格表上应该有两个数字：检索有效长度和推理有效长度。通常推理有效长度只有宣称窗口的25-50%。

## 构建实践

### 第一步：为你自己的领域构建定制化 NIAH 测试

参见 `code/main.py`。骨架代码如下：

```python
import random
import tiktoken
from your_inference_setup import model_completion

def build_haystack(n_tokens: int, seed: int = 42) -> str:
    """Generate `n_tokens` of filler text (Wikipedia paragraphs)."""
    # Use your preferred filler document source
    return filler_text[:n_tokens]

def insert_needle(haystack: str, needle: str, depth_ratio: float) -> str:
    """Insert needle at `depth_ratio` fraction into the haystack."""
    split = int(len(haystack) * depth_ratio)
    return haystack[:split] + needle + haystack[split:]

def run_niah_pass(model: str, context_len: int, depth: float, needle: str) -> bool:
    haystack = build_haystack(context_len)
    prompt = insert_needle(haystack, needle, depth)
    response = model_completion(model, prompt, instructions="Answer concisely.")
    return needle in response

# Sweep
for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for length in [1_000, 4_000, 16_000, 64_000]:
        result = run_niah_pass("target-model", length, depth, "magic_word")
        print(f"depth={depth:.2f} len={length} pass={result}")
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你的目标模型的 NIAH 评估卡片。

### 第二步：多针变体

```python
def run_multi_needle(model: str, context_len: int, needles: list[str], depths: list[float]) -> bool:
    haystack = build_haystack(context_len)
    for needle, depth in zip(needles, depths):
        haystack = insert_needle(haystack, needle, depth)

    query = "List the magic words: " + ", ".join(needle[:5] for needle in needles)
    response = model_completion(model, haystack, query)
    return all(n in response for n in needles)

needles = ["pineapple", "umbrella", "saturn"]
depths = [0.2, 0.5, 0.8]
run_multi_needle("target-model", 32_000, needles, depths)
```

像"哪三个是魔法词？"这样的问题需要检索全部三个信息。单针测试成功并不能预测多针测试的成功。

### 第三步：多跳变量追踪（RULER 风格）

```python
def variable_tracing_test(model: str, context_len: int, n_hops: int = 3) -> bool:
    """Create a chain like: X1=5, X2=X1+3, X3=X2*2. Ask for X3."""
    haystack = build_haystack(context_len)
    assignments = []
    for i in range(n_hops):
        var = f"VAR{i+1}"
        if i == 0:
            value = random.randint(1, 10)
            assignments.append(f"{var} = {value}")
        else:
            prev = f"VAR{i}"
            assignments.append(f"{var} = {prev} + {random.randint(1, 5)}")

    # Insert assignments at staggered depths
    for j, assignment in enumerate(assignments):
        depth = 0.2 + 0.3 * j
        haystack = insert_needle(haystack, assignment, depth)

    query = f"What is the value of VAR{n_hops}?"
    response = model_completion(model, haystack, query)
    # Parse numeric answer
    return True  # placeholder
```

答案需要串联三次赋值才能得出。前沿模型在12.8万 token 长度下，此项准确率通常会降至50-70%。

### 第四步：在你自己的技术栈上运行 LongBench v2

```python
# Pseudocode — LongBench v2 is a fixed 503-item dataset
from longbench_v2 import load_dataset

dataset = load_dataset()
categories = dataset.groupby("category")

for cat, examples in categories:
    acc = 0
    for ex in examples:
        response = model_completion(model, ex["context"], ex["question"])
        if response == ex["correct_answer"]:
            acc += 1
    print(f"{cat}: {acc / len(examples):.2%}")
```

按类别报告准确率。汇总得分会掩盖任务层面巨大的差异。

## 常见误区

- **仅做 NIAH 评估。** 在100万 token 下通过 NIAH 并不能说明多跳能力。务必同时运行 RULER 或定制的多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。应测试 depth=0, 0.25, 0.5, 0.75, 1.0——"中间迷失"效应是真实存在的。
- **与填充文本存在词汇重叠。** 如果"针"的关键词与填充文本共享关键词，检索就会变得过于简单。应使用 NoLiMa 风格的无重叠"针"。
- **忽略延迟。** 100万 token 的提示需要30-120秒进行预填充。应将"首个 token 时间"与准确率一同衡量。
- **仅依赖供应商自报的评测数字。** OpenAI、Google、Anthropic 都会发布自己的分数。务必针对你自己的用例独立进行重测。

## 应用指南

2026年的技术栈选择：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速简单验证 | 定制化 NIAH（3深度 × 3长度） |
| 生产环境模型选型 | 目标长度下的 RULER（13项任务） |
| 真实世界问答质量 | LongBench v2 单文档QA子集 |
| 多跳推理 | BABILong 或定制变量追踪 |
| 对话/对话式 | 目标长度下的 MRCR 8针测试 |
| 模型升级回归测试 | 固定内部 NIAH + RULER 测试套件，每个新模型都运行 |

生产环境的经验法则：除非你在目标长度上通过了 NIAH 测试及一项推理任务测试，否则永远不要相信声称的上下文窗口。

## 产出交付

保存为 `outputs/skill-long-context-eval.md`：

```markdown
# 长上下文评估报告

## 被测模型
- 模型名称：`your-model-2026-v1`
- 宣称上下文窗口：1M tokens
- 评估日期：2026-01-15

## 测试结果摘要
- **有效检索长度（90% NIAH 通过率）：** 800k tokens
- **有效推理长度（RULER 多跳 70% 阈值）：** 320k tokens
- **性能衰减曲线：** 见下方

| 任务类型 | 长度=1k | 长度=32k | 长度=128k | 长度=512k | 长度=1M |
|-----------|---------|----------|-----------|-----------|---------|
| 单针 NIAH | 100%    | 100%     | 100%      | 95%       | 85%     |
| 3针检索   | 100%    | 95%      | 80%       | 55%       | 30%     |
| 多跳追踪  | 100%    | 85%      | 60%       | 35%       | 15%     |

## 关键发现
1. 检索在高达约80万 token 长度下保持良好；多跳功能在12.8万 token 处开始下降。
2. 对于任何需要跨文档聚合的任务，不要依赖超过32万 token 的上下文。
3. 多针测试暴露了单针测试未能发现的注意力饱和问题。

## 建议
- 合同分析：在12.8万 token 以下使用。对于更长的文档，使用分块 + 检索增强生成（RAG）。
- 代码库理解：在32万 token 以内使用；超出此范围使用摘要 + 分块。
```

## 练习

1. **简单。** 构建一个 NIAH 测试，包含3种深度（0.25, 0.5, 0.75）×3种长度（1k, 4k, 16k）。在任何模型上运行。将通过率绘制为3×3热力图。
2. **中等。** 添加一个3针变体。在每个长度下测量同时检索全部3针的成功率。与相同长度下的单针通过率进行比较。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，3跳），嵌入到6.4万 token 的填充文本中。在3个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | 大海捞针 | 在填充文本中植入一个事实，要求模型检索它。 |
| RULER | 加强版 NIAH | 涵盖检索/多跳/聚合/问答的13种任务类型。 |
| 有效上下文 | 真实容量 | 准确率仍能保持在阈值之上的长度。 |
| 中间迷失 | 深度偏差 | 模型对长输入中间部分的内容注意力不足。 |
| 多针 | 同时处理多个事实 | 植入多个目标；测试的是注意力分配能力，而非仅仅是检索。 |
| MRCR | 多轮指代消解 | 8、24或100针指代消解；揭示注意力饱和点。 |
| NoLiMa | 非词汇性针 | 针和查询之间没有字面上的 token 重叠；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) —— 原始的 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) —— 多任务基准论文。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) —— 现实世界的长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) —— 更难的针测试。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) —— 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) —— 深度偏差论文。
