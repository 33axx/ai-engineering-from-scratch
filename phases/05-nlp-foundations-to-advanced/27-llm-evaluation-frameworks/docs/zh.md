# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 会漏掉语义等价性。人工评审无法扩展。LLM 作为裁判是生产环境中的答案——经过足够的校准后，可以信赖那个数字。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 13（问答）、阶段 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题

你的 RAG 系统回答："June 29th, 2007."
标准参考答案是："June 29, 2007."
精确匹配得分为 0。F1 得分约 75%。人类会打 100 分。

现在将测试用例乘以 10,000 个。再乘以对检索器、分块、提示或模型的每一次改动。你需要一个评估器，它能够理解含义、以低成本大规模运行、不会对回归问题撒谎，并且能暴露出正确的失败模式。

2026 年，有三个框架掌控着这个问题。

- **RAGAS.** 检索增强生成评估（Retrieval-Augmented Generation ASsessment）。四个 RAG 指标（忠实度、答案相关性、上下文精确度、上下文召回率），后端采用 NLI + LLM 裁判。有研究支撑，轻量级。
- **DeepEval.** LLM 的 Pytest。提供 G-Eval、任务完成度、幻觉、偏见等指标。原生支持 CI/CD。
- **G-Eval.** 一种方法（也是 DeepEval 的一个指标）：LLM 作为裁判，配合思维链、自定义标准，输出 0-1 分数。

这三个框架都依赖 LLM 作为裁判。本课程将帮助你建立对这种方法及其信任层的直觉。

## 概念

![四个评估维度，LLM 作为裁判架构](../assets/llm-evaluation.svg)

**LLM 作为裁判（LLM-as-judge）。** 用静态指标替换为：使用一个 LLM 根据评分规则对输出进行打分。给定 `(查询, 上下文, 答案)`，向裁判 LLM 发送提示："请基于忠实度打分，0-1。" 返回分数。

为什么有效：LLM 能够以极低的成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，使得 1000 样本的回归评估运行成本低于 $5。

为何会静默失败：

1. **裁判偏见。** 裁判偏好更长的答案、来自自己模型家族的答案、与提示风格匹配的答案。
2. **JSON 解析失败。** 错误的 JSON → NaN 分数 → 静默地从聚合中排除。RAGAS 用户对此深有体会。使用 try/except 加上明确的失败模式来防护。
3. **跨模型版本的漂移。** 升级裁判模型会改变每一个指标。请锁定裁判模型及其版本。

**RAG 四大指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| 忠实度（Faithfulness） | 答案中的每个主张是否都来自检索到的上下文？ | 基于 NLI 的蕴涵判断 |
| 答案相关性（Answer relevance） | 答案是否针对问题？ | 从答案生成假设性问题，与真实问题比较 |
| 上下文精确度（Context precision） | 在检索到的分块中，有多大比例是相关的？ | LLM 裁判 |
| 上下文召回率（Context recall） | 检索是否返回了所有必要信息？ | LLM 裁判，与标准答案对比 |

**G-Eval。** 定义一个自定义标准："答案是否引用了正确的来源？" 框架会自动将其扩展为思维链评估步骤，然后给出 0-1 分数。适用于 RAGAS 未涵盖的领域特定质量维度。

**校准。** 在确认裁判分数与人工标签存在相关性之前，永远不要信任原始的裁判分数。对 100 个手工标注的示例进行测试。绘制裁判 vs 人工的散点图。计算斯皮尔曼相关系数（Spearman rho）。如果 rho < 0.7，你的裁判评分规则需要改进。

## 动手构建

### 第 1 步：使用 NLI 评估忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子主张。对每个主张，使用 NLI 检查其是否被检索到的上下文所蕴含。忠实度 = 被支持的比例。

### 第 2 步：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示了与所提问题不同的问题，相关性就会下降。

### 第 3 步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分规则。明确的步骤比隐式的 "打分 0-1" 提示更稳定。

### 第 4 步：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

将其作为一个 pytest 文件交付。在每个 PR 上运行。在出现回归时阻止合并。

### 第 5 步：从头构建的玩具评估

参见 `code/main.py`。仅使用标准库实现的近似忠实度（答案主张与上下文的重叠）和近似相关性（答案 token 与问题 token 的重叠）。非生产级。仅示意结构。

## 陷阱

- **未校准。** 一个与人工标签相关度只有 0.3 的裁判只是噪声。要求在上线前进行校准运行。
- **自我评估。** 使用同一个 LLM 生成答案并作为裁判，会使分数虚高 10-20%。请使用不同模型家族的裁判。
- **成对评估中的位置偏差。** 裁判偏好呈现的第一个选项。始终随机化顺序并进行两次评估。
- **原始聚合结果隐藏了失败。** 平均分数 0.85 常常掩盖了 5% 的灾难性失败。始终检查最低分位组。
- **黄金数据集腐化。** 未经版本管理的评估集随时间漂移会破坏纵向比较。每次改动都要给数据集打标签。
- **LLM 成本。** 在规模下，裁判调用会主导成本。使用能满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small 等。

## 如何使用

2026 年的技术栈：

| 使用场景 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | RAGAS 免参考模式 |
| 人在回路中的抽检 | LangSmith 或 Phoenix，带注释 UI |
| 红队/安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新的维度。三个都运行；它们之间的分歧是有益的。

## 交付成果

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 对 10 个已知有幻觉的 RAG 示例使用 RAGAS。验证忠实度指标是否捕获了每一个幻觉。
2. **中等。** 手工标注 50 个 QA 答案的正确性，分数 0-1。使用 G-Eval 打分。测量裁判与人工之间的斯皮尔曼相关系数。
3. **困难。** 使用 DeepEval 构建一个 pytest CI 门控。故意使检索器出现回归。验证门控是否失败。添加最低分位告警，通过检查最低 10% 的阈值。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 使用 LLM 打分 | 用一个裁判模型根据评分规则对输出打分 0-1。 |
| RAGAS | RAG 指标库 | 开源评估框架，提供 4 个免参考的 RAG 指标。 |
| Faithfulness | 答案是否基于事实？ | 答案中的主张被检索到的上下文所蕴含的比例。 |
| Context precision | 检索到的分块是否相关？ | top-K 分块中真正有影响的比例。 |
| Context recall | 检索是否找到了所有内容？ | 标准答案中的主张被检索到的分块所支持的比例。 |
| G-Eval | 自定义 LLM 裁判 | 评分规则 + 思维链评估步骤 + 0-1 分数。 |
| Calibration | 信任但验证 | 裁判分数与人工分数之间的斯皮尔曼相关系数。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准、局限性。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，集成 RAGAS、DeepEval、Phoenix。
