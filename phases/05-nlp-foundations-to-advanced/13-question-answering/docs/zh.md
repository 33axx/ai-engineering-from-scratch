# 问答系统

> 三种系统塑造了现代问答系统。抽取式系统定位答案片段。检索增强式系统将答案锚定在文档中。生成式系统直接产生答案。每个现代 AI 助手都是这三者的混合体。

**类型:** 构建  
**语言:** Python  
**先决条件:** 阶段 5 · 11（机器翻译），阶段 5 · 10（注意力机制）  
**时间:** ~75 分钟

## 问题

用户输入“第一代 iPhone 是何时发布的？”并期望得到“2007 年 6 月 29 日”。而不是“苹果的历史悠久而多样”。也不是孤立的“2007 年”没有句子的支撑。一个直接、有依据、正确的答案。

在过去十年中，三种架构主导了问答系统。

- **抽取式问答。** 给定一个问题和一个已知包含答案的段落，在段落中找到答案片段的起始和结束索引。SQuAD 是标准的基准数据集。
- **开放域问答。** 系统不提供段落。首先检索相关段落，然后抽取或生成答案。这是当今所有 RAG 管道的基石。
- **生成式 / 闭书问答。** 大型语言模型从其参数化记忆中回答。无需检索。推理速度最快，事实可靠性最低。

2026 年的趋势是混合式：检索最好的几个段落，然后使用生成式模型在这些段落的基础上回答。这就是 RAG，第 14 课详细介绍了检索部分。本课构建问答部分。

## 概念

![QA 架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 使用一个变换器（BERT 系列）将问题和段落编码在一起。训练两个头，分别预测答案的起始和结束 token 索引。损失函数是有效位置上的交叉熵。输出是段落中的一个片段。从架构上讲永远不会产生幻觉，也从未处理过段落无法回答的问题（从架构上讲）。

**检索增强式（RAG）。** 两个阶段。首先，检索器从语料库中找到 top-`k` 个段落。其次，阅读器（抽取式或生成式）利用这些段落产生答案。检索器-阅读器分离使得每个部分可以独立训练和评估。现代 RAG 通常会在两者之间添加一个重排序器。

**生成式。** 一个仅解码器的 LLM（GPT、Claude、Llama）从学习到的权重中回答。没有检索步骤。在常见知识上表现优异，在罕见或近期事实上表现糟糕。幻觉率与预训练数据中的事实频率呈负相关。

## 构建

### 第 1 步：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，该数据集包含不可回答的问题。默认情况下，`question-answering` 管道返回得分最高的片段，即使模型的空值得分最高——它*不会*自动返回空答案。要获得显式的“无答案”行为，请在管道调用时传递 `handle_impossible_answer=True`：此时仅在空值得分超过所有片段得分时才返回空答案。无论哪种情况，都请检查 `score` 字段。

### 第 2 步：检索增强式管道（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的 top 段落中提取答案片段。适用于小型语料库。对于百万文档级别的语料库，请使用 FAISS 或向量数据库。

### 第 3 步：基于 RAG 的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词模式很重要。明确告诉模型要依据上下文回答，并在上下文不足时返回“我不知道”，与朴素提示相比，可将幻觉率降低 40-60%。更复杂的模式会增加引用、置信度分数和结构化提取。

### 第 4 步：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**词元级 F1**。EM 是归一化后（小写、去除标点、移除冠词）的严格匹配——要么预测完全匹配，要么得 0 分。F1 基于预测和参考之间的词元重叠计算，并给予部分分数。两者都对同义改写得分不足：“June 29, 2007”与“June 29th, 2007”通常得到 0 EM（序数破坏了归一化），但由于重叠词元仍然获得较高的 F1。

对于生产环境中的问答：

- **答案正确性**（由 LLM 或人工判断，因为指标无法捕捉语义等价性）。
- **引用准确性。** 引用的段落是否确实支持答案？通过生成引用与检索到的段落之间的字符串匹配即可自动检查。
- **拒答校准。** 当答案不在检索到的段落中时，系统是否正确地回答“我不知道”？测量虚假置信率。
- **检索召回率。** 在评估阅读器之前，检查检索器是否能将正确的段落包含在 top-`k` 中。阅读器无法修复缺失的段落。

### RAGAS：2026 年生产环境评估框架

`RAGAS` 是专为 RAG 系统设计的，并且是 2026 年的出厂默认选择。它无需黄金参考即可评估四个维度：

- **忠实度。** 答案中的每个主张是否都来自检索到的上下文？通过基于 NLI 的蕴含关系测量。您的主要幻觉指标。
- **答案相关性。** 答案是否针对问题？通过从答案生成假设性问题并与真实问题进行比较来测量。
- **上下文精确度。** 在检索到的块中，实际相关的比例是多少？低精确度 = 提示词中的噪声。
- **上下文召回率。** 检索到的集合是否包含所有必要信息？低召回率 = 阅读器无法成功。

无参考评估让您可以在没有精心策划的黄金答案的情况下，对实时生产流量进行评估。在开放式问题上叠加 LLM 作为评判者，因为精确匹配指标在这些问题上毫无用处。

`pip install ragas`。接入您的检索器 + 阅读器。每次查询获得四个标量。对回归进行告警。

## 使用

2026 年的技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找到答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料库之上，不允许闭书 | RAG：密集检索器 + LLM 阅读器 |
| 在文档存储上实时问答 | RAG + 混合检索（BM25 + 密集） + 重排序器（第 14 课） |
| 对话式问答（后续问题） | 带有对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管的领域 | 基于权威语料库的抽取式问答；绝不单独使用生成式 |

抽取式问答在 2026 年不流行，因为带有 LLM 的 RAG 可以处理更多情况。但在需要逐字引用的场景中，它仍然被使用：法律研究、法规遵从、审计工具。

## 交付

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 篇维基百科段落上搭建上述 SQuAD 抽取式管道。手工制作 10 个问题。测量答案正确的次数。如果段落和问题清晰，你应该看到 7-9 次正确。
2. **中等。** 添加一个拒答分类器。当最高检索得分低于某个阈值（例如余弦相似度 0.3）时，返回“我不知道”而不是调用阅读器。在留出集上调整阈值。
3. **困难。** 在你选择的 10,000 文档语料库上搭建 RAG 管道。实现混合检索（BM25 + 密集），并使用 RRF 融合（参见第 14 课）。分别测量有无混合步骤时的答案正确性。记录哪些问题类型受益最多。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答 | 找到答案片段 | 在给定段落中预测答案的起始和结束索引。 |
| 开放域问答 | 基于语料库的问答 | 不提供段落；必须先检索再回答。 |
| RAG | 检索然后生成 | 检索增强式生成。检索器 + 阅读器管道。 |
| SQuAD | 标准基准数据集 | 斯坦福问答数据集。EM + F1 指标。 |
| 幻觉 | 编造答案 | 阅读器输出不基于检索到的上下文。 |
| 拒答校准 | 知道何时该闭嘴 | 系统在无法回答时正确地说“我不知道”。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答中标准的密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
