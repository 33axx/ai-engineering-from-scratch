# 高级 RAG（分块、重排序、混合搜索）

> 基础 RAG 检索与查询最相似的前 k 个块。对于简单问题，这已经足够。但在多跳推理、模糊查询以及大规模语料库面前，它就会失效。高级 RAG 是区分“能在 10 个文档上运行的 demo”与“能在 1000 万个文档上运行的系统”的关键。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11，第 06 课（RAG）
**时长：** ~90 分钟
**相关：** 阶段 5 · 23（RAG 的分块策略）涵盖了全部六种分块算法 —— 递归、语义、句子、父子文档、延迟分块、上下文检索，并附有 Vectara/Anthropic 基准测试数据。本课程在此基础上进一步深入：混合搜索、重排序、查询转换。

## 学习目标

- 实现高级分块策略（语义、递归、父子），以保留文档结构和上下文
- 构建结合 BM25 关键词匹配与语义向量搜索，以及交叉编码重排序器的混合搜索流水线
- 应用查询转换技术（HyDE、多查询、后退一步）来改善模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 故障：检索到错误的块、答案不在上下文中、多跳推理失败

## 问题所在

你在第 06 课构建了一个基础 RAG 流水线。对于小规模语料库上的直白问题，它表现良好。现在试试这些：

**模糊查询**："上个季度营收是多少？" 语义搜索返回的是关于营收策略、营收预测以及 CFO 对营收增长的看法等块。这些块在语义上都与"营收"这个词相似，但没有一个包含实际数字。正确的块写着"2025 年第三季度营收为 4720 万美元"，但用的是"收益"一词而非"营收"。嵌入模型认为"营收策略"比"第三季度收益为 4720 万美元"与查询更相似。

**多跳问题**："哪个团队的客户满意度得分提升最高？" 这需要找出每个团队的满意度得分，进行比较，然后找出最大值。没有一个块包含完整答案。信息分散在各团队报告中。

**大规模语料库问题**：你有 200 万个块。正确答案在第 1,847,293 号块中。你的前 5 个检索结果拉出了第 #14、#89,201、#1,200,000、#44 和 #901,333 号块。在嵌入空间中它们很接近，但没有一个包含答案。在这种规模下，近似最近邻搜索引入了足够大的误差，导致相关结果被挤出了前 k 名。

基础 RAG 之所以失败，是因为向量相似性并不等同于相关性。一个块可能在语义上与查询相似，但对于回答该查询却毫无用处。高级 RAG 通过四种技术来解决这个问题：混合搜索（增加关键词匹配）、重排序（更仔细地对候选结果评分）、查询转换（在搜索前修正查询）以及更好的分块（以合适的粒度进行检索）。

## 概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似性）擅长理解含义。"我如何取消订阅？" 可以匹配"终止你的计划的步骤"，即使两者没有共享任何词语。但它在精确匹配上表现不佳。"错误代码 E-4021" 可能无法匹配包含"E-4021"的块，如果嵌入模型将其视为噪音的话。

关键词搜索（BM25）则相反。它擅长精确匹配。"E-4021" 完美匹配。但如果文档中说"终止你的计划"，那么"取消我的订阅"返回的结果数为零。

混合搜索同时运行这两种搜索，然后合并结果。

**BM25**（最佳匹配 25）是标准的关键词搜索算法。自 1990 年代以来，它一直是搜索引擎的支柱。其公式为：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词条 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和程度（默认值为 1.2），b 控制长度归一化（默认值为 0.75）。

通俗来说：BM25 给那些包含查询词条（尤其是稀有词条）的文档分配更高的分数，但重复词条带来的收益递减。一篇出现"收入"一词 50 次的文档，其相关性并非只出现一次文档的 50 倍。

### 倒数排序融合 (RRF)

你得到了两个排序列表：一个来自向量搜索，一个来自 BM25。如何将它们合并？倒数排序融合是标准方法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常为 60），用于防止排名第一的结果占据主导地位。

在向量搜索中排名第 1、在 BM25 中排名第 5 的文档得分为：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

在向量搜索中排名第 3、在 BM25 中排名第 2 的文档得分为：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然平衡了两种信号。在两个列表中排名都靠前的文档获得最佳分数。在一个列表中排名第一但在另一个列表中缺失的文档获得中等分数。这种方法很稳健，因为它使用排名而非原始分数，因此两个系统之间分数分布的差异无关紧要。

### 重排序

检索（无论是向量、关键词还是混合检索）速度很快但不够精确。它使用双编码器：查询和每个文档分别独立嵌入，然后进行比较。嵌入计算一次后缓存。这可以扩展到数百万个文档。

重排序使用交叉编码器：将查询和候选文档一起输入模型，输出一个相关性分数。该模型同时看到两个文本，能够捕捉它们之间细粒度的交互。交叉编码器可以理解"第三季度收益是多少？" 与包含"第三季度收入 4720 万美元"的块高度相关，即使双编码器错过了这种联系。

权衡：交叉编码器比双编码器慢 100-1000 倍，因为它需要联合处理查询-文档对。你无法为一百万个文档预先计算交叉编码器分数。解决方案：先检索一个更大的候选集（混合搜索的前 50 个结果），然后使用交叉编码器重新排序，得到最终的前 5 个结果。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

2026 年常见的重排序模型：
- Cohere Rerank 3.5：托管 API，多语言，对混合语料库召回率提升最大
- Voyage rerank-2.5：托管 API，托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开源权重，支持 100+ 种语言
- bge-reranker-v2-m3：开源权重，强大的基线模型
- cross-encoder/ms-marco-MiniLM-L-6-v2：开源权重，可在 CPU 上运行用于原型开发
- ColBERTv2 / Jina-ColBERT-v2：后期交互多向量重排序器——评分时复杂度为 O(词元数) 而非 O(文档数)

### 查询转换

有时问题不在于检索，而在于查询本身。"那个关于新政变化的东西是什么？" 是一个糟糕的搜索查询。它不包含任何具体词条。嵌入模糊不清。没有任何检索系统能够据此找到正确的文档。

**查询重写**：将用户的查询改写为更好的搜索查询。LLM 可以做到：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（假设性文档嵌入）**：不直接使用查询进行搜索，而是先生成一个假设性的答案，对其进行嵌入，然后搜索与之相似的、真实的文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

对假设性答案进行嵌入，然后搜索与它相似的真实文档。直觉上：假设性答案在嵌入空间中比原始问题更接近真实答案。问题和答案具有不同的语言结构。通过生成假设性答案，你在嵌入空间中弥合了"问题空间"和"答案空间"之间的差距。

HyDE 在检索前增加了一次 LLM 调用。这会使延迟增加 500-2000 毫秒。当原始查询的检索质量较差时，这是值得的。

### 父子分块

标准分块迫使你在以下两者之间做出权衡：小分块用于精确检索，大分块用于提供足够上下文。父子分块消除了这种权衡。

对小分块（128 个词元）进行索引用于检索。当一个小分块被检索到时，返回其父分块（512 个词元）用于提示。小分块精确匹配查询。父分块为 LLM 生成良好答案提供了足够的上下文。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询"企业退款？" 精确匹配子分块 C2。但提示接收的是完整的父分块 P，其中包含了关于处理时间和提交流程的周围上下文。

### 元数据过滤

在运行向量搜索之前，根据元数据（日期、来源、类别、作者、语言）过滤语料库。这减少了搜索空间，并防止不相关的结果出现。

"上个月安全策略有什么变化？" 应只搜索过去 30 天内属于安全类别的文档。如果没有元数据过滤，你会搜索整个语料库，可能会检索到一份因语义相似而恰好匹配的、两年前的安全文档。

生产级 RAG 系统会为每个分块存储元数据：源文档、创建日期、类别、作者、版本。向量数据库支持在相似性搜索之前根据元数据进行预过滤，这对于大规模性能至关重要。

### 评估

你构建了一个 RAG 系统。如何知道它是否有效？三个指标：

**检索相关性 (Recall@k)**：对于一组已知相关文档的测试问题，在 top-k 结果中出现了多少百分比的相关文档？如果问题的答案在第 #47 分块中，该分块是否出现在 top-5 里？

**忠实性**：生成的答案是否基于检索到的文档？如果检索到的分块说"60 天退款窗口"，而模型说"90 天退款窗口"，那就是忠实性失败。尽管有正确的上下文，模型还是产生了幻觉。

**答案正确性**：生成的答案是否与预期答案一致？这是端到端指标。它结合了检索质量和生成质量。

一个简单的忠实性检查：提取生成答案中的每个主张，验证它是否（在实质上）出现在检索到的分块中。如果答案包含未在任何检索分块中出现的事实，则很可能是幻觉。

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

## 构建它

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数排序融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简单的重排序器

在生产环境中，你可能会使用交叉编码模型。这里我们构建一个重排序器，通过词重叠、词条重要性和短语匹配来对查询-文档相关性进行评分。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设性文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实性评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 使用它

使用真正的交叉编码器进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真实 LLM 进行 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

使用 Weaviate 进行生产级混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha 值。

## 交付

本课程产出：
- `outputs/prompt-advanced-rag-debugger.md` —— 用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md` —— 用于构建带有混合搜索和重排序的生产级 RAG 的技能

## 练习

1. 在示例文档上比较 BM25、向量搜索和混合搜索。针对 5 个测试查询中的每一个，记录哪种方法在第 #1 位置返回了最相关的分块。混合搜索应至少在 3 个查询上胜出。

2. 实现一个元数据过滤器。为每个文档添加"category"字段（security, billing, api, product）。在运行向量搜索之前，仅筛选出相关类别的分块。使用"使用了什么加密？"进行测试，并验证它只搜索安全类别的分块。

3. 使用第 06 课的简单生成函数构建一个完整的 HyDE 流水线。比较在所有 5 个测试查询上，直接查询搜索和 HyDE 搜索的检索质量（top-3 相关性）。HyDE 应改善模糊查询的结果。

4. 在示例文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。使用子分块进行搜索，但在提示中返回父分块。将生成的答案与使用 chunk_size=50 的标准分块进行比较。

5. 创建一个评估数据集：10 个问题，附带已知的答案分块。测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合搜索 + 重排序的 Recall@3、Recall@5 和 Recall@10。绘制结果，并指出重排序在哪些方面帮助最大。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|--------------|----------|
| BM25 | “关键词搜索” | 一种概率排序算法，根据词频、逆文档频率和文档长度归一化对文档进行评分 |
| 混合搜索 | “两全其美” | 同时运行语义（向量）和关键词（BM25）搜索，然后通过排序融合合并结果 |
| 倒数排序融合 | “合并排序列表” | 通过求和每个文档在所有列表中的 1/(k + rank) 来组合多个排序列表 |
| 重排序 | “二次评分” | 使用更昂贵的交叉编码模型对初始检索的候选集进行重新评分 |
| 交叉编码器 | “联合查询-文档模型” | 将查询和文档作为单个输入，产生相关性分数；比双编码器更准确，但用于全语料库搜索时太慢 |
| 双编码器 | “独立嵌入模型” | 独立嵌入查询和文档的模型；由于嵌入可预计算，速度很快，但准确性不如交叉编码器 |
| HyDE | “用假答案搜索” | 生成一个对查询的假设性答案，对其嵌入，然后搜索与之相似的真实文档 |
| 父子分块 | “小搜索，大上下文” | 为精确检索索引小块，但返回较大的父块以提供足够上下文 |
| 元数据过滤 | “先缩小范围再搜索” | 在运行向量搜索之前，根据文档属性（日期、来源、类别）进行过滤，以减小搜索空间 |
| 忠实性 | “是否保持基于事实” | 生成的答案是否得到检索文档的支持，而非从模型的训练数据中幻觉出来 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) —— BM25 的权威参考，解释了公式背后的概率基础
- Cormack 等, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) —— 原始 RRF 论文，证明其优于更复杂的融合方法
- Gao 等, "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) —— HyDE 论文，展示了假设性文档嵌入无需任何训练数据即可改善检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) —— 表明在 BM25 之上使用交叉编码重排序可显著提升检索质量
- [Khattab 等, "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) —— 将提示构造和权重选择视为检索流水线上的优化问题；阅读此文以理解“编程 LLM”而非“提示 LLM”
- [Edge 等, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) —— GraphRAG 论文：实体关系抽取 + Leiden 社区检测用于查询聚焦摘要；区分全局与局部检索
- [Asai 等, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) —— 使用反思词元进行自我评估的 RAG；超越了静态的“检索-然后-生成”的智能体前沿
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) —— 如何将自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher）作为检索前步骤
