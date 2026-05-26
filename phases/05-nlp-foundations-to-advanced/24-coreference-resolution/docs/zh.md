# 共指消解

> “她打电话给他。他没有接。医生正在吃午饭。”三个指代对应两个人，但都没指名。共指消解就是要弄清楚谁是谁。

**类型：** 学习  
**语言：** Python  
**先修条件：** Phase 5 · 06 (NER)、Phase 5 · 07 (POS & Parsing)  
**时长：** ~60 分钟

## 问题

从一篇 300 词的新闻中提取所有提及 Apple Inc. 的表述。当文章说“Apple”时很容易。但当它说“the company”、“they”、“Cupertino's technology giant”或“Jobs's firm”时就难了。如果不将这些提及解析到同一个实体，你的 NER 流水线会遗漏 60%–80% 的提及。

共指消解将所有指向同一现实世界实体的表述链接成一个簇。它是表层 NLP（NER、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的黏合剂。

为什么它在 2026 年很重要：

- **摘要：** “The CEO announced...” vs “Tim Cook announced...”——摘要应该提及 CEO 的名字。
- **问答：** “Who did she call?” 需要解析“she”。
- **信息抽取：** 一个知识图谱中“PER1 创立了 Apple”和“Jobs 创立了 Apple”作为两条独立记录是错误的。
- **多文档信息抽取：** 合并关于同一事件的多篇文章中的提及是跨文档共指。

## 概念

![共指聚类：提及 → 实体](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：提及（文本跨度）的一个聚类，每个簇对应一个实体。

**提及类型。**

- **命名实体。** “Tim Cook”
- **名词性短语。** “the CEO”、“the company”
- **代词。** “he”、“she”、“they”、“it”
- **同位语。** “Tim Cook, Apple's CEO,”

**架构。**

1. **基于规则（Hobbs, 1978）。** 利用语法规则进行基于句法树的代词消解。不错的基线。在代词上出奇地难以超越。
2. **提及对分类器。** 对每一对提及 (m_i, m_j) 预测它们是否共指。通过传递闭包聚类。2016 年之前的标准做法。
3. **提及排序。** 对每个提及，对候选先行词（包括“无先行词”）进行排序，选取最高分者。
4. **基于跨度端到端（Lee 等, 2017）。** Transformer 编码器。枚举所有候选跨度（长度上限）。预测提及分数。为每个跨度预测先行词概率。贪心聚类。现代默认方法。
5. **生成式（2024+）。** 给 LLM 一个提示：“列出文本中的每个代词及其先行词。”在简单情形下表现良好，在长文档和罕见指代上力不从心。

**评价指标。** 五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕捉聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的最新水平：约 83 F1。

**已知困难情形。**

- 定指描述指向之前多页引入的实体。
- 搭桥回指（“the wheels” → 之前提到的一辆车）。
- 中文和日文等语言中的零代词。
- 预指（代词在指代之前出现）：“When **she** walked in, Mary smiled.”

## 动手实现

### 步骤 1：预训练神经共指消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长文档上，你会得到类似下面的结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 步骤 2：基于规则的代词消解器（教学用）

参见 `code/main.py` 中仅使用标准库的实现：

1. 提取提及：命名实体（大写开头段落）、代词（字典查找）、定指描述（“the X”）。
2. 对每个代词，查看前 K 个提及并按以下条件打分：
   - 性别/数的一致性（启发式）
   - 近期性（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争，但它展示了搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用大语言模型进行共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要注意两种失败模式。首先，LLM 倾向于过度合并（“him”和“her”指向两个不同的人）。其次，LLM 在长文档中会静默地丢弃提及。始终要用跨度的起始偏移量进行检查。

### 步骤 4：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，先从带标注的测试集上的跨度级别的精确率和召回率开始，然后再增加提及链接 F1。

## 常见陷阱

- **单例爆炸。** 有些系统把每个提及都报告为独立的簇。B³ 比较宽容，MUC 则会惩罚。始终检查全部三个指标。
- **长上下文中的代词。** 在超过 2000 token 的文档上性能下降约 15 F1。要仔细分块。
- **性别假设。** 硬编码的性别规则在非二元指代、组织、动物上会失效。使用学习模型或中性评分。
- **长文档上的 LLM 漂移。** 单次 API 调用无法可靠地对超过 50 段的提及进行聚类。使用滑动窗口 + 合并。

## 应用场景

2026 年的技术栈：

| 情形 | 选择 |
|-----|------|
| 英文、单文档 | `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP 神经共指 |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专门的端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude 使用结构化输出共指提示 |
| 生产对话系统 | 基于规则的后备 + 神经主模型 + 关键槽位人工审核 |

2026 年主流的集成模式：先运行 NER，再运行共指消解，将共指簇合并到 NER 实体中。下游任务每个簇看到一个实体，而不是每个提及一个实体。

## 交付产物

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 段人工编写的段落上运行 `code/main.py` 中的基于规则消解器。与人工标注对比提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练的神经共指模型。将自己的手动标注与模型聚类结果进行比较。它在哪些地方失败了？
3. **困难。** 构建一个共指增强的 NER 流水线：先做 NER，再通过共指簇进行合并。在 100 篇文章上测量与纯 NER 相比的实体覆盖率提升。

## 关键术语

| 术语 | 日常说法 | 实际含义 |
|------|----------|----------|
| 提及（Mention） | 一个指代 | 指向实体的文本跨度（名称、代词、名词短语）。 |
| 先行词（Antecedent） | “它”指什么 | 后面的提及与之共指的较早提及。 |
| 簇（Cluster） | 实体的提及集合 | 所有指向同一现实世界实体的提及的集合。 |
| 回指（Anaphora） | 向后指代 | 后面提及指向前面的（“他” → “John”）。 |
| 预指（Cataphora） | 向前指代 | 前面提及指向后面的（“当他到达时，John...”）。 |
| 搭桥指代（Bridging） | 隐含指代 | “我买了一辆车。轮子坏了。”（那辆车的轮子）。 |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第 26 章——共指消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf)——经典教科书章节。
- [Lee 等 (2017). 端到端神经共指消解](https://arxiv.org/abs/1707.07045)——基于跨度的端到端方法。
- [Joshi 等 (2020). SpanBERT](https://arxiv.org/abs/1907.10529)——改进共指消解的预训练。
- [Pradhan 等 (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/)——基准任务。
- [Hobbs (1978). 代词指代消解](https://www.sciencedirect.com/science/article/pii/0024384178900064)——基于规则的经典论文。
