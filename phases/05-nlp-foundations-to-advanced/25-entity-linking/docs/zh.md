# 实体链接与消歧

> NER 识别出“巴黎”。实体链接决定：是法国巴黎？帕丽斯·希尔顿？德克萨斯州巴黎？还是特洛伊王子帕里斯？若不链接，你的知识图谱将一直充满歧义。

**类型：** 构建  
**语言：** Python  
**前置要求：** 阶段 5 · 06（命名实体识别），阶段 5 · 24（指代消解）  
**时间：** 约 60 分钟  

## 问题

一句原文是：“Jordan beat the press.” 你的 NER 将“Jordan”标注为人物。很好。但是**哪一个** Jordan？

- 迈克尔·乔丹（篮球）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利机器学习教授 —— 没错，机器学习论文中确实存在这种混淆）？
- 约旦（国家）？
- 乔丹（希伯来语名字）？

实体链接（EL）将每个提及项解析为知识库中的唯一条目：维基数据、维基百科、DBpedia 或你的领域知识库。包含两个子任务：

1. **候选生成。** 给定“Jordan”，知识库中有哪些条目是合理的？
2. **消歧。** 给定上下文，哪一个候选是正确的？

两个步骤都可学习，都有基准测试。整个流水线已稳定了十年 —— 变化的是消歧器的质量。

## 概念

![实体链接流水线：提及 → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的字符串形式（“Jordan”），在别名索引中查找候选。维基百科的别名词典涵盖了大多数命名实体：“JFK” → 约翰·F·肯尼迪、杰奎琳·肯尼迪、JFK 机场、JFK（电影）。典型的索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(实体 | 提及) × 上下文相似度(实体, 文本)`。效果好、速度快、无需训练。
2. **基于嵌入（ESS / REL / BLINK）。** 编码提及 + 上下文。编码每个候选的描述。取最大余弦值。2020–2024 年的默认方法。
3. **生成式（GENRE, 2021; 基于 LLM, 2023+）。** 逐 token 解码实体的规范名。约束到有效实体名称的 trie 上，确保输出是有效的 KB ID。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一个 pass 中同时完成 NER + 候选生成 + 消歧。流水线系统仍主导生产环境，因为你可以替换组件。

### 两个度量指标

- **提及召回率（候选生成）。** 正确的 KB 条目出现在候选列表中的黄金提及比例。是整个流水线的下限。
- **消歧准确率 / F1。** 在候选正确的前提下，top-1 正确的频率。

始终同时报告两者。一个在 80% 候选召回率上达到 99% 消歧的系统，最终流水线性能是 80%。

## 构建

### 步骤 1：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约 1800 万（别名, 实体）对。从维基数据转储下载。存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是玩具示例。替换为基于嵌入的余弦相似度（参见 `code/main.py` 步骤 2 的 transformer 版本）。

### 步骤 3：基于嵌入（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，为每个 KB 实体嵌入一次。在查询时，为提及 + 上下文嵌入一次，与候选池做点积，取最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的维基百科标题。约束解码（见第 20 课）确保只能输出有效标题。与基于 KB 的 trie 紧密集成。现代后代是 REL-GEN 和带有结构化输出的 LLM 提示式 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最易部署的 EL 流水线。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准：1393 篇路透社文章，3.4 万个提及，维基百科实体。报告知识库内准确率（`P@1`）和知识库外 NIL 检测率。

## 陷阱

- **NIL 处理。** 某些提及不在知识库中（新兴实体、冷门人物）。系统必须预测 NIL 而非猜错实体。单独测量。
- **提及边界错误。** 上游 NER 缺失部分跨度（“Bank of America” 只标记为 “Bank”）。EL 召回率下降。
- **流行度偏差。** 训练过的系统过度预测高频实体。一篇机器学习论文中提及“Michael I. Jordan”常被链接到篮球乔丹。
- **跨语言 EL。** 将中文文本中的提及映射到英文维基百科实体。需要多语言编码器或翻译步骤。
- **知识库过时。** 新公司、事件、人物不在去年的维基百科转储中。生产流水线需要更新循环。

## 使用

2026 年的技术栈：

| 情况 | 选择 |
|-----------|------|
| 通用英语 + 维基百科 | BLINK 或 REL |
| 跨语言，知识库 = 维基百科 | mGENRE |
| 适合 LLM，每天少量提及 | 用候选列表 + 约束 JSON 提示 Claude/GPT-4 |
| 领域特定知识库（医疗、法律） | 定制 BERT + 知识库感知检索 + 在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可上线的生产模式：NER → 指代消解 → 每个提及做 EL → 将聚类坍缩为每个聚类一个规范实体。输出：文档中每个实体一个 KB ID，而非每个提及一个。

## 交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简易。** 在 `code/main.py` 中实现先验+上下文消歧器，用于 10 个有歧义的提及（Paris、Jordan、Apple）。手工标注正确实体。测量准确率。
2. **中等。** 使用句子 transformer 编码 50 个有歧义的提及。嵌入每个候选的描述。将基于嵌入的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个 1000 实体的领域知识库（例如你公司内的员工 + 产品）。实现端到端的 NER + EL。在 100 个保留句子上测量精确率和召回率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到维基百科 | 将提及映射到知识库中的唯一条目。 |
| 候选生成 | 可能是谁？ | 为提及返回一个合理的知识库条目短列表。 |
| 消歧 | 选出正确的那个 | 使用上下文对候选评分，选出胜者。 |
| 别名索引 | 查找表 | 从字符串形式到候选实体的映射。 |
| NIL | 不在知识库中 | 明确预测没有知识库条目匹配。 |
| KB | 知识库 | 维基数据、维基百科、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准 | 1393 篇带有黄金实体链接的路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) —— 基础的先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) —— 基于嵌入的主力模型。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) —— 带有约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) —— 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) —— 开放的生产栈。
