# 信息检索与搜索

> BM25 精准但脆弱。Dense 覆盖面广但会漏掉关键词。Hybrid 是 2026 年的默认方案。其他都是调参。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 04（GloVe、FastText、子词）
**用时：** ~75 分钟

## 问题

用户输入 “what happens if someone lies to get money”，期望找到实际对应的法条：“Section 420 IPC。” 纯关键词搜索会完全漏掉（没有共享词汇）。纯语义搜索在嵌入向量未经法律文本训练时也会漏掉。真正的搜索必须同时处理这两种情况。

IR 是所有 RAG 系统、每个搜索栏、每个文档网站模糊查找背后的流水线。2026 年在生产中有效的架构并非单一方法，而是一系列互补方法的链条，每种方法能够捕捉前一种方法失败的情况。

本节课将构建每个组件，并说明每个组件处理哪些失败场景。

## 概念

![混合检索：BM25 + 稠密 + RRF + 交叉编码器重排](../assets/retrieval.svg)

四层。按需选择。

1. **稀疏检索（BM25）。** 速度快，精确匹配上精准，但语义能力差。基于倒排索引运行。百万级文档上每个查询耗时低于10ms。能正确处理法条引用、产品代码、错误信息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。能捕获同义改写和语义相似性。但会漏掉仅相差一个字的关键词精确匹配。使用 FAISS 或向量数据库时每个查询 50-200ms。
3. **融合。** 合并来自稀疏和稠密的排序列表。倒数排序融合（RRF）是简单的默认方案，因为它忽略原始分数（分数处于不同尺度），只使用排序位置。当你确信某一信号在你的领域占主导时，可以选择加权融合。
4. **交叉编码器重排。** 取融合的前 30 个结果，用交叉编码器（将查询和文档一起输入，给每一对打分），保留前 5 个。交叉编码器每对比双编码器慢，但准确度高得多。你只需在前 30 个结果上运行，以此分摊成本。

三路检索（BM25 + 稠密 + 学习型稀疏如 SPLADE）在 2026 年的基准测试中优于两路，但需要为学习型稀疏索引提供基础设施。对大多数团队而言，两路加交叉编码器重排是甜蜜点。

## 动手构建

### 第一步：从头实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得关注的参数。`k1=1.5` 控制词频饱和，值越大词频重复权重越高。`b=0.75` 控制长度归一化，0 忽略文档长度，1 完全归一化。这两个默认值来自原始论文中 Robertson 的推荐，通常无需调整。

### 第二步：使用双编码器进行稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入进行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索来说足够强大。对于多语言任务，使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度时，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第三步：倒数排序融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 这个常量来自 RRF 原始论文。更高的 `k` 会拉平排序差异的贡献，更低的 `k` 会使前排排序占主导。60 是已发表的默认值，通常无需调整。

### 第四步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段的组合。BM25 找到词汇匹配。稠密检索找到语义匹配。RRF 合并两个排序，无需分数校准。交叉编码器对前 30 个结果重新打分，使用查询-文档对一起输入，捕获双编码器遗漏的细粒度相关性。保留前 5 个结果。

### 第五步：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在查询中，当正确的文档存在时，它出现在前 k 个结果中的比例 |
| MRR（平均倒数排名） | 第一个相关文档的排名倒数的平均值 |
| nDCG@k | 考虑了相关性的层级，不仅仅是非相关/相关 |

对于 RAG 而言，检索器的 **Recall@k** 是最重要的数字。如果正确的段落不在检索结果中，阅读器就无法回答。

调试技巧：对于失败的查询，比较稀疏检索和稠密检索的排序。如果其中一个找到了正确的文档而另一个没有，那么要么是词汇不匹配（解决方法：补充缺失的那一半），要么是语义歧义（解决方法：更好的嵌入或重排器）。

## 使用

2026 年的技术栈：

| 规模 | 技术栈 |
|------|--------|
| 1k-100k 个文档 | 内存中的 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独的数据库。 |
| 100k-10M 个文档 | FAISS 或 pgvector 用于稠密检索 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 个文档 | Qdrant / Weaviate / Vespa / Milvus（支持混合检索）。对前 30 个结果进行交叉编码器重排。 |
| 最佳质量前沿 | 三路（BM25 + 稠密 + SPLADE）+ ColBERT 后期交互重排 |

无论你选择什么，都要预留评估预算。在基准测试端到端 RAG 准确率之前，先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产级 RAG 的硬核经验

- **80% 的 RAG 失败可追溯到数据摄入和分块，而非模型。** 团队花费数周时间切换 LLM 和调整提示词，而检索器却默默地在每三次查询中返回一次错误的上下文。首先修复分块。
- **分块策略比分块大小更重要。** 固定大小切分会破坏表格、代码和嵌套标题。句子感知是默认方案；语义或基于 LLM 的分块对技术文档和产品手册效果显著。
- **父-子文档模式。** 检索精确的“子”块以保持准确性。当多个子块来自同一父段落时，用父块替换以保留上下文。这能在无需重新训练的情况下持续提升答案质量。
- **k_rerank=3 通常是最优的。** 每多一个块都会增加 token 成本和生成延迟，且不会提升答案质量。如果 k=8 对你来说仍然优于 k=3，说明重排器性能不足。
- **HyDE / 查询扩展。** 从查询生成假设性答案，对假设答案进行嵌入，然后检索。弥合短问题与长文档之间的措辞差距。无需训练即可获得免费的精度提升。
- **上下文预算保持在 8K token 以下。** 如果在该限制下持续命中的结果说明重排器阈值太宽松。
- **对所有内容进行版本管理。** 提示词、分块规则、嵌入模型、重排器。任何漂移都会悄然破坏答案质量。CI 门禁检查忠实性、上下文精度和未回答问题率，在用户看到之前阻止回归。
- **三路检索（BM25 + 稠密 + 学习型稀疏如 SPLADE）优于两路**，尤其是在混合了专有名词和语义的查询上。当基础设施支持 SPLADE 索引时部署它。

根据 2026 年行业测量，合理的检索设计可将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在包含 500 个文档的语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25 单独、稠密单独和混合模式在 top-5 上的召回率。
2. **中等。** 添加 MRR 计算。对于每个有已知正确文档的测试查询，找出正确文档在 BM25、稠密和混合排序中的排名。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域微调稠密编码器。从 500 对查询-文档对构建训练集。比较微调前后召回率。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|--------------|----------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、逆文档频率和文档长度对文档打分。 |
| 稠密检索 | 向量搜索 | 将查询和文档编码为向量，寻找最近邻。 |
| 双编码器 | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器 | 重排器模型 | 将查询和文档一起编码。速度慢但准确。 |
| RRF | 排序融合 | 通过求和 `1/(k + rank)` 来合并两个排序。 |
| Recall@k | 检索指标 | 查询中相关的文档出现在 top-k 中的比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 学习型稀疏检索器，缩小了与稠密检索的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索。
