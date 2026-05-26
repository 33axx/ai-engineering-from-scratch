# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 11（机器翻译）  
**时间：** 约 75 分钟

## 问题

一篇 2000 字的新闻文章出现在你的信息流中。你需要 120 个字来概括它。你可以从文章中选出三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都叫做摘要，但它们是完全不同的问题。

抽取式摘要是排序问题。对每个句子评分，返回前 `k` 个。输出总是符合语法的，因为它是逐字摘取的。风险在于遗漏分布在整篇文章中的内容。

生成式摘要是生成问题。Transformer 根据输入条件生成新文本。输出流畅且压缩性强，但可能产生源文本中不存在的事实。风险在于自信地编造。

本课程将构建这两种方法，并介绍各自对应的失败模式。

## 概念

![抽取式 TextRank 与生成式 transformer 的对比](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法），根据句子与其他句子的连接程度对句子进行评分。得分最高的句子即为摘要。典型实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档并通过交叉注意力逐 token 生成摘要。Pegasus 特别使用了间隔句子预训练目标，使其在几乎不需要微调的情况下就能出色地完成摘要任务。

使用 **ROUGE**（面向摘要评估的召回率导向替身）进行评估。ROUGE-1 和 ROUGE-2 评估一元组和二元组重叠。ROUGE-L 评估最长公共子序列。分数越高越好，但 40 的 ROUGE-L 算“不错”，50 算“出色”。每篇论文都会报告这三种指标。使用 `rouge-score` 包。

## 构建

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两件事值得说明。相似度函数使用对数归一化的词重叠，这是原始 TextRank 的变体。TF-IDF 向量的余弦相似度也可以。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：使用 BART 的生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调。它开箱即用地生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），请使用相应的 Pegasus 检查点或根据目标数据自行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干还原。没有它，“running”和“run”会被计为不同的词，导致 ROUGE 低估。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 作为主要的摘要指标已有二十年，但在 2026 年它已不足以单独使用。一项关于 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年之前逐渐流行，现在大多数摘要论文中会与 ROUGE 一同报告。
- **BARTScore** 将评估视为生成任务：根据源文本评估 BART 模型赋予摘要的可能性。
- **MoverScore**（基于上下文嵌入的推土机距离）在 2025 年的摘要基准测试中达到最高位置，因为它能比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实性**在 2021-2023 年很常见，现在通常被 **G-Eval**（一种 GPT-4 提示链，通过思维链推理对连贯性、一致性、流畅性、相关性进行评分）所取代。
- **G-Eval** 及类似的 LLM 评判方法在评估标准设计良好时，与人类判断的一致性约为 80%。

生产环境建议：报告 ROUGE-L 用于传统比较，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。针对 50-100 个人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是逐字从源文本中摘取的，但如果源句子脱离上下文、过时或引用顺序不当，它们仍然可能产生误导。这是生产系统在处理合规相关内容时仍然偏好抽取式方法的唯一最大原因。

需要指出的幻觉类型：

- **实体替换。** 源文本说“John Smith”。摘要说“John Brown”。
- **数字偏差。** 源文本说“25,000”。摘要说“2500 万”。
- **极性翻转。** 源文本说“拒绝了该提议”。摘要说“接受了该提议”。
- **事实编造。** 源文本未提及 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个二元分类器，在源句子和摘要句子之间的蕴含关系上训练。预测事实/非事实。
- **基于 QA 的事实性。** 向 QA 模型提问，问题的答案在源文本中。如果摘要支持不同的答案，则标记。
- **实体级 F1。** 比较源文本和摘要中的命名实体。仅在摘要中出现的实体是可疑的。

对于任何面向用户且事实性至关重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要将事实性检查纳入流程。

## 使用

2026 年技术栈：

| 使用场景 | 推荐方案 |
|---------|-------------|
| 新闻，3-5 句摘要，英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优后的 T5 |
| 多文档，长文本 | 任何具有 32k+ 上下文长度的 LLM，配合提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，通过构造降低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

当计算资源不是限制时，具有长上下文的 LLM 通常在 2026 年胜过专用模型。权衡在于成本和可重复性；专用模型能给出更一致的输出。

## 提交

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将前 3 个句子与参考摘要进行比较。测量 ROUGE-L。对于 CNN/DailyMail 风格的文章，你应该看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性：从源文本和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率以及摘要实体相对于源文本的精确率。高精确率低召回率意味着安全但简洁；低精确率意味着出现了编造的实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（通过实体 F1 值）以及每篇摘要的成本。记录每种方法的优势所在。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 | 挑选句子 | 从源文本中逐字返回句子。永远不会产生幻觉。 |
| 生成式 | 重写 | 根据源文本条件生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考摘要之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行 PageRank。 |
| 事实性 | 是否正确 | 摘要中的主张是否得到源文本的支持。 |
| 幻觉 | 编造的内容 | 摘要中源文本不支持的内容。 |

## 进一步阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和间隔句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性概述论文。
