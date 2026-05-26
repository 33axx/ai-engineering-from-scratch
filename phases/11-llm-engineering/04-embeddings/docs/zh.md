# 嵌入与向量表示

> 文本是离散的，数学是连续的。每次你让大语言模型查找“相似”文档、比较含义或超越关键词进行搜索时，你依赖的都是连接这两个世界的桥梁。这座桥梁就是嵌入。不理解嵌入，你便不理解现代AI——你只是在用它。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11，第 01 课（提示工程）
**时间：** 约75分钟
**关联：** 阶段 5 · 22（嵌入模型深度剖析）涵盖了密集vs稀疏vs多向量、套娃截断、按轴模型选择等主题。本课聚焦于生产管道（向量数据库、HNSW、相似度数学）。在选择模型前，请先阅读阶段 5 · 22。

## 学习目标

- 使用 API 提供商和开源模型生成文本嵌入，并计算它们之间的余弦相似度
- 解释为什么嵌入能解决关键词搜索无法处理的词汇不匹配问题
- 构建一个基于语义的搜索索引，从而按含义而非精确关键词匹配来检索文档
- 使用检索基准（precision@k, recall）评估嵌入质量，并为你的任务选择正确的嵌入模型

## 问题所在

你有10,000个支持工单。一位客户写道“我的付款没成功”。你需要找到相似的过往工单。关键词搜索能找到包含“付款”和“没成功”的工单，却会漏掉“交易失败”“扣款被拒绝”和“账单错误”。这些工单描述的是完全相同的问题，但用词完全不同。

这就是词汇不匹配问题。人类语言有几十种方式表达同一件事。关键词搜索将每个词视为独立的符号，不包含含义。它无法知道“被拒绝”和“没成功”指的是同一概念。

你需要一种文本表示方式，让含义而非拼写决定相似性。你需要一种方法，将“我的付款没成功”和“交易被拒绝”在某个数学空间中放置得很近，同时将“我的付款按时到达”推得很远——尽管它共享了“付款”这个词。

这种表示就是嵌入。

## 概念理解

### 什么是嵌入？

嵌入是一个表示文本含义的稠密浮点数向量。“稠密”这个词很关键——每个维度都携带信息，这与稀疏表示（词袋、TF-IDF）不同，后者大部分维度都是零。

“The cat sat on the mat” 会变成像 `[0.023, -0.041, 0.087, ..., 0.012]` 这样的东西——根据模型不同，这是一列 768 到 3072 个数字。这些数字编码了含义。你永远不会直接检视它们，而是比较它们。

### Word2Vec 的突破

2013年，Tomas Mikolov 及其在谷歌的同事发表了 Word2Vec。核心思想：训练一个神经网络，根据一个词的邻居来预测该词（或根据一个词预测邻居），隐藏层的权重就成为了有意义的向量表示。

著名的结果：

```python
# 向量运算：国王 - 男人 + 女人 ≈ 女王
vector("king") - vector("man") + vector("woman") ≈ vector("queen")

词嵌入上的向量算术捕捉到了语义关系。从“男人”到“女人”的方向大致与从“国王”到“女王”的方向相同。这一刻，该领域意识到几何可以编码含义。

Word2Vec 生成了300维的向量。每个词无论上下文如何都只有一个向量。“river bank”中的“bank”和“bank account”中的“bank”拥有相同的嵌入。这个局限推动了接下来十年的研究。

### 从词到句子

词嵌入表示单个标记。生产系统需要嵌入整个句子、段落或文档。出现了四种方法：

**平均法**：取句子中所有词向量的均值。代价低，有信息损失，但对于短文本效果出奇地好。完全丢失了词序——“狗咬人”和“人咬狗”会得到相同的嵌入。

**CLS 标记**：Transformer 模型（BERT, 2018）会输出一个特殊的 [CLS] 标记嵌入，代表整个输入。比平均法好，但 [CLS] 标记是为下一句预测任务训练的，不适用于相似性。

**对比学习**：明确训练模型，将相似对推近，不相似对拉远。Sentence-BERT（Reimers & Gurevych, 2019）使用了这种方法，并成为现代嵌入模型的基础。给定“How do I reset my password?”和“I need to change my password”，模型学习到这两个句子应具有几乎相同的向量。

**指令微调嵌入**：最新方法。像 E5 和 GTE 这样的模型接受一个任务前缀（“search_query:”、“search_document:”），告诉模型要生成哪种类型的嵌入。这使得一个模型可以服务于多个任务。

```python
# 指令微调嵌入的典型前缀
prefix = "search_query: "  # 用于查询
text = "how to reset my password"
embedding = model.encode(prefix + text)
```

### 现代嵌入模型

市场已经收敛到少数几个生产级选项（截至2026年初的MTEB分数，MTEB v2）：

| 模型 | 提供商 | 维度 | MTEB | 上下文长度 | 每百万tokens成本 |
|-------|----------|-----------|------|---------|------------------|
| Gemini Embedding 2 | Google | 3072（套娃） | 67.7（检索） | 8192 | $0.15 |
| embed-v4 | Cohere | 1024（套娃） | 65.2 | 128K | $0.12 |
| voyage-4 | Voyage AI | 1024/2048（套娃） | 66.8 | 32K | $0.12 |
| text-embedding-3-large | OpenAI | 3072（套娃） | 64.6 | 8192 | $0.13 |
| text-embedding-3-small | OpenAI | 1536（套娃） | 62.3 | 8192 | $0.02 |
| BGE-M3 | BAAI | 1024（稠密+稀疏+ColBERT） | 63.0 多语言 | 8192 | 开放权重 |
| Qwen3-Embedding | Alibaba | 4096（套娃） | 66.9 | 32K | 开放权重 |
| Nomic-embed-v2 | Nomic | 768（套娃） | 63.1 | 8192 | 开放权重 |

MTEB（大规模文本嵌入基准）v2涵盖了100多个任务，包括检索、分类、聚类、重排序和摘要。分数越高越好。到2026年，开放权重模型（Qwen3-Embedding, BGE-M3）在大多数指标上匹敌甚至超越封闭托管的模型。Gemini Embedding 2 在纯检索上领先；Voyage/Cohere 在特定领域（金融、法律、代码）领先。在确定使用前，始终要在你自己的查询上做基准测试。

### 相似度度量

给定两个嵌入向量，有三种方法可以衡量它们有多相似：

**余弦相似度**：两个向量之间夹角的余弦值。范围从 -1（相反）到 1（方向相同）。忽略模长——一个10个词的句子和一个500词的文档如果方向相同，得分可以是1.0。这是90%用例的默认选择。

```python
import numpy as np

def cosine_similarity(v1, v2):
    """返回两个向量之间的余弦相似度。"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
```

**点积**：两个向量的原始内积。当向量已归一化（单位长度）时，与余弦相似度相同。计算更快。OpenAI 的嵌入已归一化，因此点积和余弦给出相同的排名。

```python
def dot_product(v1, v2):
    """返回两个向量之间的点积。假设向量已归一化。"""
    return np.dot(v1, v2)
```

**欧几里得（L2）距离**：向量空间中的直线距离。越小 = 越相似。对模长差异敏感。当空间中绝对位置（而非仅方向）重要时使用。

```python
def euclidean_distance(v1, v2):
    """返回两个向量之间的欧几里得距离。"""
    return np.linalg.norm(v1 - v2)
```

何时使用哪种：

| 度量 | 何时使用 | 避免使用 |
|--------|----------|------------|
| 余弦相似度 | 比较不同长度的文本；大多数检索任务 | 模长携带信息时 |
| 点积 | 嵌入已归一化；追求最大速度 | 向量模长不同时 |
| 欧几里得距离 | 聚类；空间最近邻问题 | 比较长度差异巨大的文档时 |

### 向量数据库与 HNSW

暴力相似性搜索会将查询与每个存储的向量进行比较。对于100万个1536维的向量，每次查询需要15亿次乘加运算。太慢了。

向量数据库通过近似最近邻（ANN）算法解决这个问题。主流的算法是 HNSW（层级可导航小世界）：

1. 构建一个多层的向量图
2. 顶层稀疏——远距离簇之间的长程连接
3. 底层稠密——附近向量之间的精细连接
4. 搜索从顶层开始，贪婪地向下细化
5. 在 O(log n) 时间内返回近似 top-k 结果，而不是 O(n)

HNSW 用很小的精度损失（通常95-99%召回率）换取了巨大的速度提升。对于1000万个向量，暴力搜索需要数秒，HNSW 只需毫秒。

```python
# 概念性 HNSW搜索示例（伪代码）
def hnsw_search(query, layers, top_k=10):
    # 从最顶层开始
    current = random_entry_point(layers[-1])
    for layer in reversed(layers):
        current = greedy_search_layer(query, current, layer, top_k)
    return current[:top_k]
```

生产选项：

| 数据库 | 类型 | 最适合 | 最大规模 |
|----------|------|----------|-----------|
| Pinecone | 托管SaaS | 零运维生产 | 数十亿 |
| Weaviate | 开源 | 自托管，混合搜索 | 1亿+ |
| Qdrant | 开源 | 高性能，带过滤 | 1亿+ |
| ChromaDB | 嵌入式 | 原型开发，本地开发 | 100万 |
| pgvector | Postgres扩展 | 已在使用Postgres | 1000万 |
| FAISS | 库 | 进程内，研究 | 10亿+ |

### 分块策略

文档太长，不能作为单个向量嵌入。一份50页的PDF涵盖十几个主题——它的嵌入会变成所有内容的平均值，对任何具体内容都不相似。你需要将文档拆分成块，然后对每个块进行嵌入。

**固定大小分块**：每 N 个 token 切分一次，带 M 个 token 的重叠。简单且可预测。当文档没有清晰结构时效果很好。512 token 的块带50 token 重叠：块1是 tokens 0-511，块2是 tokens 462-973。

**基于句子的分块**：在句子边界切分，将句子分组直到达到 token 限制。每个块至少包含一个完整句子。比固定大小好，因为你永远不会把一个想法切成两半。

**递归分块**：先尝试在最大的边界（章节标题）切分。如果仍然太大，则尝试段落边界。然后是句子边界。最后是字符限制。这就是 LangChain 的 `RecursiveCharacterTextSplitter`，对于混合格式的语料库效果很好。

**语义分块**：嵌入每个句子，然后将嵌入相似度高的连续句子分组。当嵌入相似度低于阈值时，开始新的块。成本高（需要对每个句子单独嵌入），但能产生最连贯的块。

| 策略 | 复杂度 | 质量 | 最适合 |
|----------|-----------|---------|----------|
| 固定大小 | 低 | 一般 | 非结构化文本、日志 |
| 基于句子 | 低 | 好 | 文章、邮件 |
| 递归 | 中 | 好 | Markdown、HTML、混合文档 |
| 语义 | 高 | 最佳 | 关键检索质量 |

大多数系统的理想点：256-512 token 的块，带50 token 重叠。

### 双编码器与交叉编码器

双编码器独立地将查询和文档编码成向量，然后比较向量。速度快——你只需将查询嵌入一次，然后与预先计算好的文档嵌入进行比较。这就是你用于检索的方式。

交叉编码器将查询和文档作为一个输入，输出一个相关性分数。速度慢——它需要将每个查询-文档对通过完整模型处理一次。但准确性高得多，因为它可以同时关注查询和文档 token 之间的交互。

生产模式：双编码器检索出 top-100 候选，交叉编码器将其重排序为 top-10。这就是“检索-然后-重排序”管道。

```python
# 检索-然后-重排序管道
def retrieve_and_rerank(query, documents, bi_encoder, cross_encoder, top_k=100, rerank_top=10):
    # 第一步：双编码器快速检索
    query_emb = bi_encoder.encode(query)
    doc_embs = [bi_encoder.encode(doc) for doc in documents]
    scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    # 第二步：交叉编码器精确重排序
    pairs = [(query, documents[i]) for i in top_indices]
    rerank_scores = cross_encoder.predict(pairs)
    final_indices = [top_indices[i] for i in np.argsort(rerank_scores)[-rerank_top:][::-1]]
    return [(documents[i], rerank_scores[i]) for i in final_indices]
```

重排序模型：Cohere Rerank 3.5（每1000次查询 $2）、BGE-reranker-v2（免费，开源）、Jina Reranker v2（免费，开源）。

### 套娃嵌入

传统嵌入是全有或全无的。一个1536维向量使用1536个浮点数。你不能在不重新训练的情况下将其截断到256维。

套娃表示学习（Kusupati 等，2022）解决了这个问题。模型经过训练，使得前 N 个维度捕获了最重要的信息，就像俄罗斯套娃一样。将一个1536维的套娃嵌入截断到256维会损失一些准确性，但仍然可用。

OpenAI 的 text-embedding-3-small 和 text-embedding-3-large 通过 `dimensions` 参数支持套娃截断。请求256维而不是1536维可以将存储减少6倍，同时在 MTEB 基准上大约只损失3-5%的准确率。

### 二值量化

一个1536维的嵌入以 float32 存储需要6,144字节。乘以1000万个文档：仅向量就需要61 GB。

二值量化将每个浮点转换为一个比特：正值变为1，负值变为0。存储从6,144字节降至192字节——减少了32倍。相似度通过汉明距离（计算不同的位数）计算，CPU 可以用一条指令完成。

准确率损失大约在检索召回率上为5-10%。常见模式：在首次扫描数百万个向量时使用二值量化，然后使用全精度向量对 top-1000 重新评分。这样可以在内存减少32倍的情况下获得95%+的全精度准确率。

## 构建部分

我们从零开始构建一个语义搜索引擎。不使用向量数据库。不使用外部嵌入 API。仅使用 Python 和 numpy 进行数学计算。

### 第一步：文本分块

```python
import re
from typing import List, Dict, Any

def chunk_by_tokens(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    将文本分割成固定大小的 token 块，带重叠。
    注意：这是一个简化的基于空格的 token 计数。
    生产环境应使用真正的 tokenizer。
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_by_sentences(text: str, max_chars: int = 1000) -> List[str]:
    """
    将文本分割成句子块。每个块至少包含一个完整的句子。
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
```

### 第二步：从头构建嵌入

我们实现一个简单的稠密嵌入，使用带 L2 归一化的 TF-IDF。这不是神经嵌入，但它遵循相同的约定：文本输入，固定大小向量输出，相似的文本产生相似的向量。

```python
import numpy as np
from collections import Counter

class SimpleEmbedder:
    """
    一个基于 TF-IDF 的简单嵌入器。
    这仅用于教学——生产环境请使用 Sentence-BERT 或类似的模型。
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab = {}  # 词 -> 索引
        self.idf = {}    # 词 -> IDF 分数
        self.vocab_size = vocab_size
        self.fitted = False
        
    def fit(self, documents: List[str]):
        """根据文档语料库拟合 TF-IDF 权重。"""
        # 构建词汇表（频率最高的前 N 个词）
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc.lower().split())
        
        most_common = word_counts.most_common(self.vocab_size)
        for i, (word, _) in enumerate(most_common):
            self.vocab[word] = i
        
        # 计算 IDF
        N = len(documents)
        for word in self.vocab:
            df = sum(1 for doc in documents if word in doc.lower().split())
            self.idf[word] = np.log((N + 1) / (df + 1)) + 1
        
        self.fitted = True
    
    def encode(self, text: str) -> np.ndarray:
        """将文本转换为稠密向量。"""
        if not self.fitted:
            raise ValueError("请先调用 fit()")
        
        vector = np.zeros(self.vocab_size)
        words = text.lower().split()
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if word in self.vocab:
                tf = count / len(words)
                vector[self.vocab[word]] = tf * self.idf[word]
        
        # L2 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
```

### 第三步：相似度函数

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """返回两个向量之间的余弦相似度。"""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0.0

def dot_product_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """返回两个向量之间的点积。假设向量已归一化。"""
    return np.dot(v1, v2)

def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    将欧几里得距离转换为相似度分数。
    较小的距离 -> 较高的相似度。
    """
    dist = np.linalg.norm(v1 - v2)
    return 1 / (1 + dist)  # 映射到 (0, 1]
```

### 第四步：带暴力搜索的向量索引

```python
from typing import List, Tuple, Callable
import numpy as np

class VectorIndex:
    """
    一个用于存储和搜索向量的简单索引。
    使用暴力搜索——对于生产环境，请使用 HNSW 或类似的算法。
    """
    def __init__(self, similarity_fn: Callable = cosine_similarity):
        self.vectors = []
        self.metadata = []
        self.similarity_fn = similarity_fn
    
    def add(self, vector: np.ndarray, metadata: dict = None):
        """向索引中添加一个向量及其可选的元数据。"""
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        搜索与查询向量最相似的 top_k 个向量。
        返回（相似度分数，元数据）元组的列表。
        """
        scores = []
        for vec in self.vectors:
            score = self.similarity_fn(query_vector, vec)
            scores.append(score)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(scores[i], self.metadata[i]) for i in top_indices]
```

### 第五步：语义搜索引擎

```python
class SemanticSearchEngine:
    """
    一个完整的语义搜索引擎，包含分块、嵌入和搜索功能。
    """
    def __init__(self, embedder, chunk_size: int = 200, overlap: int = 50):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.index = VectorIndex()
        self.document_id = 0
    
    def index_document(self, text: str, metadata: dict = None):
        """将文档分块、嵌入并添加到索引中。"""
        chunks = chunk_by_tokens(text, self.chunk_size, self.overlap)
        metadata = metadata or {}
        
        for chunk in chunks:
            vector = self.embedder.encode(chunk)
            chunk_metadata = {
                "document_id": self.document_id,
                "chunk": chunk,
                **metadata
            }
            self.index.add(vector, chunk_metadata)
        
        self.document_id += 1
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        """搜索相似文本并返回结果。"""
        query_vector = self.embedder.encode(query)
        return self.index.search(query_vector, top_k)
```

### 第六步：比较相似度度量

```python
# 使用相同的数据和查询比较不同的相似度度量
def compare_metrics(engine, query, top_k=3):
    """比较同一查询在不同度量下的结果。"""
    q_vec = engine.embedder.encode(query)
    
    # 余弦相似度
    engine.index.similarity_fn = cosine_similarity
    cosine_results = engine.index.search(q_vec, top_k)
    
    # 点积
    engine.index.similarity_fn = dot_product_similarity
    dot_results = engine.index.search(q_vec, top_k)
    
    # 欧几里得距离
    engine.index.similarity_fn = euclidean_similarity
    euclidean_results = engine.index.search(q_vec, top_k)
    
    print(f"查询: '{query}'\n")
    print("余弦相似度结果:")
    for score, meta in cosine_results:
        print(f"  分数: {score:.4f} | {meta['chunk'][:80]}...")
    
    print("\n点积结果:")
    for score, meta in dot_results:
        print(f"  分数: {score:.4f} | {meta['chunk'][:80]}...")
    
    print("\n欧几里得相似度结果:")
    for score, meta in euclidean_results:
        print(f"  分数: {score:.4f} | {meta['chunk'][:80]}...")
    print("-" * 60)
```

## 使用部分

当你使用生产级嵌入 API 时，架构保持不变。只有嵌入器发生变化：

```python
# 使用 OpenAI 的 text-embedding-3-small
import openai

class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        openai.api_key = "your-api-key"  # 使用环境变量
    
    def encode(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)

# 其余代码不变
# engine = SemanticSearchEngine(OpenAIEmbedder())
```

使用 OpenAI 进行套娃截断——同一模型，更低维度，更低存储：

```python
class MatryoshkaEmbedder:
    def __init__(self, model="text-embedding-3-small", dimensions=256):
        self.model = model
        self.dimensions = dimensions
        openai.api_key = "your-api-key"
    
    def encode(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            input=text,
            model=self.model,
            dimensions=self.dimensions  # 套娃截断
        )
        return np.array(response.data[0].embedding)
```

256维向量使用的存储减少6倍。对于1000万个文档，这是10 GB 对比 61 GB。标准基准上的准确率损失大约为3-5%。

使用 Cohere 进行重排序：

```python
import cohere

co = cohere.Client("your-api-key")

def rerank(query, documents, top_k=10):
    results = co.rerank(
        query=query,
        documents=documents,
        model="rerank-english-v3.0",
        top_n=top_k
    )
    return [(result.document, result.relevance_score) for result in results]
```

使用本地嵌入，无 API 依赖：

```python
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)
```

我们构建的 `VectorIndex` 类适用于上述任何一种。只需更换嵌入函数，保留搜索逻辑。

## 交付部分

本课产出：
- `outputs/prompt-embedding-advisor.md` —— 一个用于为特定用例选择嵌入模型和策略的提示
- `outputs/skill-embedding-patterns.md` —— 一个技能，教导智能体如何在生产环境中有效使用嵌入

## 练习

1. **度量比较**：使用余弦相似度、点积和欧几里得距离，对样本文档运行相同的5个查询。记录每种度量的 top-3 结果。哪些查询的度量结果不一致？为什么？

2. **块大小实验**：分别使用块大小 50、100、200 和 500 个词对样本文档进行索引。对每种大小运行5个查询，记录 top-1 相似度分数。绘制块大小与检索质量之间的关系图。找出块大小开始损害质量的那个点。

3. **套娃模拟**：构建一个能生成500维向量的 `SimpleEmbedder`。将其截断到50、100、200和500维。测量每次截断时检索召回率的下降程度。这模拟了套娃行为，但不需要真正的训练技巧。

4. **二值量化**：从搜索引擎中取出嵌入，将其转换为二进制（1 如果为正，0 如果为负），并实现汉明距离搜索。将 top-10 结果与全精度余弦相似度进行比较。测量重叠百分比。

5. **基于句子的分块**：用 `chunk_by_sentences` 替换固定大小的分块。运行相同的查询并比较检索分数。尊重句子边界是否能改善结果？

## 关键术语

| 术语 | 人们说它是什么 | 它实际的意思 |
|------|----------------|----------------------|
| Embedding | “文本转数字” | 一种稠密向量，其中几何上的邻近编码了语义相似性 |
| Word2Vec | “嵌入老祖宗” | 2013年的模型，通过预测上下文词来学习词向量；证明了向量算术能编码含义 |
| Cosine similarity | “两个向量有多相似” | 向量之间夹角的余弦；1 = 方向相同，0 = 正交，-1 = 相反 |
| HNSW | “快速向量搜索” | 层级可导航小世界图——多层结构，可实现 O(log n) 近似最近邻搜索 |
| Bi-encoder | “分开嵌入，快速比较” | 将查询和文档独立编码成向量；支持预计算和快速检索 |
| Cross-encoder | “慢但准确的重排序器” | 将查询-文档对联合通过完整模型处理；准确率更高，无法预计算 |
| Matryoshka embeddings | “可截断的向量” | 经过训练的嵌入，使得前 N 个维度捕获最重要的信息，从而实现可变大小存储 |
| Binary quantization | “1比特嵌入” | 将浮点向量转换为二进制（仅符号位），实现32倍存储压缩，配合汉明距离搜索 |
| Chunking | “为嵌入拆分文档” | 将文档分割成256-512 token 的片段，使每个片段可以独立嵌入和检索 |
| Vector database | “嵌入的搜索引擎” | 针对存储向量和执行大规模近似最近邻搜索进行了优化的数据存储 |
| Contrastive learning | “通过比较训练” | 训练方法，将相似对的嵌入推近，不相似对的嵌入拉远 |
| MTEB | “嵌入基准” | 大规模文本嵌入基准——涵盖8个任务的56个数据集；比较嵌入模型的标准 |

## 延伸阅读

- Mikolov 等，“Efficient Estimation of Word Representations in Vector Space”（2013）—— Word2Vec 论文，以国王-女王类比开启了嵌入革命
- Reimers & Gurevych，“Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks”（2019）—— 如何训练双编码器用于句子级相似度，现代嵌入模型的基础
- Kusupati 等，“Matryoshka Representation Learning”（2022）—— OpenAI 在 text-embedding-3 中采用的变维嵌入技术
- Malkov & Yashunin，“Efficient and Robust Approximate Nearest Neighbor using Hierarchical Navigable Small World Graphs”（2018）—— HNSW 论文，大多数生产级向量搜索背后的算法
- OpenAI Embeddings Guide（platform.openai.com/docs/guides/embeddings）—— text-embedding-3 模型的实用参考，包括套娃维度缩减
- MTEB Leaderboard（huggingface.co/spaces/mteb/leaderboard）—— 实时基准，比较所有嵌入模型在任务和语言上的表现
- [Muennighoff 等，“MTEB: Massive Text Embedding Benchmark”（EACL 2023）](https://arxiv.org/abs/2210.07316) —— 定义了基准报告的8个任务类别（分类、聚类、对分类、重排序、检索、STS、摘要、双文本挖掘）；在信任任何单一 MTEB 分数之前阅读此文
- [Sentence Transformers 文档](https://www.sbert.net/) —— 双编码器与交叉编码器、池化策略以及本课实现的“摄取-分割-嵌入-存储”RAG 管道的权威参考
```
