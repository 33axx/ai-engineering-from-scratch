# 嵌入模型 — 2026 深度指南

> Word2Vec 给你每个词一个向量。现代嵌入模型给你每个段落一个向量，支持跨语言，有稀疏、稠密和多向量三种视图，尺寸可根据你的索引调整。选错了，你的 RAG 就会检索到错误内容。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 5 · 03（Word2Vec），阶段 5 · 14（信息检索）
**时间：** ~60 分钟

## 问题

你的 RAG 系统在 40% 的情况下检索到了错误的段落。罪魁祸首通常不是向量数据库，也不是提示词，而是嵌入模型。

在 2026 年选择嵌入模型意味着从五个维度进行抉择：

1. **稠密 vs 稀疏 vs 多向量。** 每个段落一个向量，或每个 token 一个向量，或一个稀疏加权的词袋。
2. **语言覆盖。** 单语言英文模型在纯英文任务上仍然获胜。当语料混合时，多语言模型胜出。
3. **上下文长度。** 512 token vs 8,192 vs 32,768 —— 实际有效容量通常只有标称最大值的 60-70%。
4. **维度预算。** 3,072 个浮点数（全精度）= 每个向量 12 KB。对于 1 亿个向量，存储成本为 $1,300/月。马特罗什卡截断可将其降低 4 倍。
5. **开源 vs 托管。** 开源意味着你控制整个堆栈和数据。托管意味着你以控制权换取永远最新的技术。

本课将明确这些权衡，让你能够基于证据而不是上个季度的流行趋势做出选择。

## 概念

![稠密、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义相近度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏嵌入。** SPLADE 风格。一个 transformer 为每个词汇 token 预测一个权重，然后将大部分权重置零。结果是一个大小为 |vocab| 的稀疏向量。捕捉词法匹配（类似 BM25），但使用学习得到的词项权重。在关键词密集的查询上表现强劲。

**多向量（后交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。使用 MaxSim 评分：对于每个查询 token，找到最相似的文档 token，汇总分数。存储和评分成本更高，但在长查询和特定领域语料库上胜出。

**BGE-M3：三者合一。** 单个模型同时输出稠密、稀疏和多向量表示。每种表示可以独立查询；分数通过加权求和融合。当你想用一个检查点获得灵活性时，是 2026 年的默认选择。

**马特罗什卡表示学习。** 经过训练，使得向量的前 N 维成为一个可用的独立嵌入。将 1,536 维向量截断为 256 维，准确率损失约 1%，存储节省 6 倍。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 支持。

### MTEB 排行榜只反映了部分情况

大规模文本嵌入基准（Massive Text Embedding Benchmark）—— 启动时（2022 年）包含 8 个任务类型的 56 个任务，在 MTEB v2 中扩展到 100+ 个任务。2026 年初，Gemini Embedding 2 在检索任务上排名第一（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语言任务上领先（63.0）。排行榜是必要的但不充分 —— 始终在你的领域上进行基准测试。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速初筛 | 稠密双编码器（BGE-M3, text-3-small） |
| 召回提升 | 稀疏（SPLADE, BGE-M3 稀疏）+ RRF 融合 |
| 前 50 名精排 | 多向量（ColBERTv2）或交叉编码器重排序器 |

大多数生产堆栈使用全部三层。

## 动手构建

### 第一步：基准 —— 使用 Sentence-BERT 的稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使得点积等于余弦相似度。始终设置它。

### 第二步：马特罗什卡截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练，使得前几层截断是无损的。非马特罗什卡模型（原始 Sentence-BERT）在截断后性能急剧下降。

### 第三步：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次推理调用，三个索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调整权重。

### 第四步：在自定义任务上进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在你的 *代表性* 子集上运行候选模型。不要仅仅相信排行榜排名 —— 你的领域很重要。

### 第五步：手工实现余弦相似度

参见 `code/main.py`。平均哈希技巧嵌入（仅标准库）。无法与 transformer 嵌入竞争，但展示了流程：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **查询和文档使用相同模型。** 某些模型（Voyage、Jina-ColBERT）使用非对称编码 —— 查询和文档经过不同路径。始终查看模型卡片。
- **缺少前缀。** `bge-*` 模型需要在查询前添加 `"Represent this sentence for searching relevant passages: "`。如果忘记，召回率会下降 3-5 个点。
- **过度截断马特罗什卡。** 1,536 → 256 通常是安全的。1,536 → 64 不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（参见第 23 课）。
- **忽略延迟尾部分布。** MTEB 分数隐藏了 p99 延迟。一个 600M 的模型可能比 335M 的模型高 2 个点，但每次查询成本高出 3 倍。

## 使用建议

2026 年堆栈：

| 场景 | 选择 |
|-----------|------|
| 纯英文，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源，英文 | `BAAI/bge-large-en-v1.5` |
| 开源，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | 马特罗什卡截断 + int8 量化 |
| 关键词密集查询 | 添加 SPLADE 稀疏，与稠密进行 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果某个领域特定模型胜出超过 3 个点则切换。

## 交付物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子进行全维度（384）编码，然后在马特罗什卡 128 维下进行。测量在 10 个查询上的 MRR 下降。
2. **中等。** 比较 BGE-M3 稠密、稀疏和 colbert 在你领域的 500 个段落上的表现。哪个在 recall@10 上胜出？RRF 融合是否优于最佳单一模式？
3. **困难。** 在三个候选模型上，针对你的前两个领域任务运行 MTEB。报告 MTEB 分数、100 个查询批次的 p99 延迟以及每 100 万查询的美元成本。选择帕累托最优的那个。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 稠密嵌入 | 那个向量 | 每个文本一个固定大小的向量。余弦相似度用于排序。 |
| 稀疏嵌入 | 学习得到的 BM25 | 每个词汇 token 一个权重；大部分为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每个 token 一个向量；MaxSim 评分；索引更大，召回更好。 |
| 马特罗什卡 | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的更小嵌入。 |
| MTEB | 那个基准 | 大规模文本嵌入基准 —— 启动时 56 个任务，v2 中 100+ 个。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常被引用用于跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 进一步阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三种模式的模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产环境中的后交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
