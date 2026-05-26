# 关系抽取 & 知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系抽取找到它们之间的边。知识图谱是节点、边及其出处总和。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 5 · 06 (NER), 阶段 5 · 25 (实体链接)
**时间：** 约 60 分钟

## 问题

分析师读到一句话：“蒂姆·库克于 2011 年成为苹果公司 CEO。”其中包含四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取 (RE) 将自由文本转化为结构化三元组 `(subject, relation, object)`。在整个语料库上聚合后，就得到知识图谱。聚合后可以进行查询，为 RAG、分析或合规审计提供推理基础。

2026 年的问题：LLM 热情地抽取关系，但过于热情。它们会幻觉出源文本不支持的三元组。没有出处，就无法区分真实三元组与看似合理的虚构。2026 年的答案是 AEVS 式的锚定与验证流水线。

## 概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式** `(subject_entity, relation_type, object_entity)`。关系可以来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，什么都可以）。

**三种抽取方法**

1. **基于规则/模式** Hearst 模式：“X such as Y” → `(Y, isA, X)`。再加上手工构建的正则表达式。脆弱、精确、可解释。
2. **监督分类器** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年标准方法。
3. **生成式 LLM** 提示模型输出三元组。开箱即用。需要出处，否则会幻觉出看似合理的垃圾。

**AEVS（锚定-抽取-验证-补充，2026）** 当前的幻觉缓解框架：

- **锚定** 精确识别每个实体跨度与关系短语跨度及其位置。
- **抽取** 生成与锚定跨度相关联的三元组。
- **验证** 将每个三元组元素匹配回源文本；拒绝任何不支持的部分。
- **补充** 覆盖性检查确保没有锚定跨度被遗漏。

幻觉显著下降。需要更多计算资源，但可审计。

**开放 vs 封闭权衡**

- **封闭本体** 固定的属性列表（例如 Wikidata 的 11000+ 个属性）。可预测、可查询、难以杜撰。
- **开放 IE** 任何动词短语都成为关系。高召回率、低精确率、查询杂乱。

生产级知识图谱通常混合使用：先用开放 IE 发现，然后将关系规范化到封闭本体上，再合并到主图中。

## 构建它

### 第 1 步：基于模式的关系抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst 模式仍用于特定领域的流水线，因为它们易于调试。

### 第 2 步：有监督的关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个序列到序列的关系抽取器：输入文本，输出三元组，且已使用 Wikidata 属性 ID。在远程监督数据上微调。标准的开源基线模型。

### 第 3 步：LLM 提示抽取 + 锚定

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回的跨度是否与源文本匹配。拒绝 `text[start:end] != triple_entity` 的情况。这是 AEVS “验证”步骤的简化形式。

### 第 4 步：规范化为封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作量的 60–80%。请为此预留预算。

### 第 5 步：构建一个小型图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于知识图谱的 RAG 系统的原子操作。可通过 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强的图存储进行扩展。

## 陷阱

- **关系抽取前的指代消解** “他创立了苹果”——关系抽取需要知道“他”是谁。先运行指代消解（第 24 课）。
- **实体规范化** “Apple Inc”和“Apple”必须解析为同一个节点。先进行实体链接（第 25 课）。
- **幻觉三元组** LLM 会输出文本不支持的三元组。强制进行跨度验证。
- **关系规范化漂移** 开放 IE 关系不一致（“was born in”、“came from”、“is a native of”）。必须将其折叠为规范 ID，否则图将无法查询。
- **时间错误** “蒂姆·库克是苹果公司 CEO”——现在正确，2005 年错误。许多关系是有时间限制的。使用限定符（Wikidata 中的 P580 开始时间、P582 结束时间）。
- **领域不匹配** REBEL 在维基百科上训练。法律、医学和科学文本通常需要领域微调的关系抽取模型。

## 使用它

2026 年的技术选型：

| 场景 | 选择 |
|------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred 配合 Wikidata 规范化 |
| 特定领域（生物医学、法律） | SciREX 风格的领域微调 + 自定义本体 |
| LLM 提示 + 可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 高容量新闻信息抽取 | 基于模式 + 有监督混合 |
| 从头构建知识图谱 | 开放 IE + 人工规范化处理 |
| 时态知识图谱 | 使用限定符（开始/结束时间、时间点）抽取 |

集成模式：NER → 指代消解 → 实体链接 → 关系抽取 → 本体映射 → 图加载。每个阶段都是潜在的质量关口。

## 交付它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单** 在 5 条新闻句子上运行 `code/main.py` 中的模式抽取器。手工检查精确率。
2. **中等** 对相同句子使用 REBEL（或一个小型 LLM）抽取三元组。比较两者结果。哪个抽取器精确率更高？哪个召回率更高？
3. **困难** 构建 AEVS 流水线：使用 LLM 抽取 + 验证跨度是否与源文本匹配。在 50 条维基百科风格的句子上，测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 三元组 | 主体-关系-客体 | `(s, r, o)` 元组，知识图谱的原子单元。 |
| 开放 IE | 抽取一切 | 开放词汇的关系短语；高召回、低精确。 |
| 封闭本体 | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化 | 统一一切 | 将表面名称/关系映射到规范 ID。 |
| AEVS | 基于证据的抽取 | 锚定-抽取-验证-补充流水线（2026 年）。 |
| 出处 | 真实来源链接 | 每个三元组携带一个文档 ID + 字符跨度以指向其来源。 |
| 远程监督 | 廉价标签 | 将文本与现有知识图谱对齐以创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq 关系抽取主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计方案。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 标准图查询教程。
