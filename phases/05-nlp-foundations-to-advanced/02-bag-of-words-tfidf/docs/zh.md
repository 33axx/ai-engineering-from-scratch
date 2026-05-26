# Bag of Words、TF-IDF 与文本表示

> 先计数，后思考。2026 年的今天，TF-IDF 在定义明确的任务上依然比嵌入模型表现更好。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 01（文本处理），阶段 2 · 02（从零实现线性回归）
**时间：** 约 75 分钟

## 问题

模型需要数字。你手里只有字符串。

每一个 NLP 管道都必须回答同一个问题：如何将变长的 token 序列转换成一个固定大小的向量，使得分类器能够消费它？这个领域最早给出的答案，也是最简单却有效的方案：统计单词出现次数，构造向量。

这个向量承载的生产级 NLP 任务比任何嵌入模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准的头十年。2026 年的从业者在面对窄分类任务时仍然会首先想到它。它快速、可解释，并且在单词出现与否决定一切的场景下，效果往往与 4 亿参数的嵌入模型不相上下。

本课将从零实现词袋（Bag of Words）模型，然后是 TF-IDF。接着展示 scikit-learn 如何用三行代码做同样的事。最后指出什么时候该用嵌入模型。

## 概念

**词袋（Bag of Words, BoW）** 丢弃了顺序。对于每个文档，统计每个词汇表单词出现了多少次。向量的长度等于词汇表大小。位置 `i` 是单词 `i` 的计数。

**TF-IDF** 对词袋进行重加权。出现在每个文档中的单词不包含信息量，因此拉低它的权重。在语料库中罕见但在单个文档中频繁出现的单词是信号，因此提高它的权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是文档总数。`log` 使得常见词的权重保持有界。

关键性质：两者都产生稀疏向量，且轴是可解释的。你可以查看训练好的分类器的权重，读出哪些词将文档推向哪个类别。这在 768 维的 BERT 嵌入上是做不到的。

## 动手构建

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课中的 `code/main.py` 使用简单的转小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着第一个文档中第一个看到的词的索引为 0。惯例各有不同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇表索引。元素 `[i][j]` 表示“单词 `j` 在文档 `i` 中出现了多少次”。文档 1 中 `cat` 出现了两次，因为确实如此。文档 0 中 `ran` 出现零次，因为确实没有出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个值得提及的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保出现在每个文档中的词仍然有 IDF 为 1（而不是 0），这与 scikit-learn 的默认行为一致。其他实现使用原始的 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇表单词（`the`, `cat`, `sat`, `dog`, `ran`）。`the` 出现在所有三个文档中，因此它的 IDF 很低。`dog` 只出现在一个文档中，因此它的 IDF 很高。向量是稀疏的（大多数元素很小），具有区分性的单词凸显出来。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不经归一化，较长的文档会得到更大的向量，从而主导相似度分数。L2 归一化将每个文档放在单位超球面上。行之间的余弦相似度现在就是点积。

## 使用它

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一步调用中完成分词、词汇表和词袋。`TfidfVectorizer` 增加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万文档，密集版本无法放入内存；保持稀疏直到分类器要求密集数据。

影响巨大的参数：

| 参数 | 效果 |
|------|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在噪声数据上缩减词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似于无硬编码列表的停用词移除。 |
| `stop_words="english"` | scikit-learn 内置的停用词列表。任务相关——情感分析不应丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。当某个术语在一个文档中重复多次时很有用。 |

### 何时 TF-IDF 仍然胜出（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。单词出现与否决定一切；语义细微差别不重要。
- 低数据量场景（几百个标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何对延迟敏感的场景。TF-IDF 加线性模型在微秒级响应。通过 Transformer 嵌入一个文档需要 10-100 毫秒。
- 系统需要解释其预测。检查分类器的系数。权重最高的正类词就是原因。

### 何时 TF-IDF 失败

语义盲点失败。考虑以下两个文档：

- “这部电影一点也不好看。”
- “这部电影棒极了。”

一个是负面评论，一个是正面评论。它们的 TF-IDF 重叠恰好是 `{电影, 这, 部}`。基于词袋的分类器必须记住单词 `不` 靠近 `好看` 会翻转标签。在足够的数据上它可以学习这个规律，但远远不如理解句法的模型那样优雅。

另一个失败：推理时遇到词汇表外单词。在 IMDb 评论上训练的词袋模型，如果从未在训练中出现过 `Zoomer-approved` 这个 token，则完全不知道如何处理它。子词嵌入（第 04 课）可以处理这种情况。TF-IDF 不行。

### 混合方法：TF-IDF 加权嵌入

2026 年中等数据量分类的实际默认选择：使用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你得到嵌入的语义能力，以及 TF-IDF 对罕见词的强调。分类器在汇聚后的向量上训练。在情感、主题和意图分类上，当标注样本少于约 5 万时，这种方法优于单独使用其中任何一种。

## 提交

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单题。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇完全不相交的文档得分为 0.0。
2. **中等题。** 在 `bag_of_words` 中添加 `n-gram` 支持。参数 `n` 产生 `n` 元组的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难题。** 使用 GloVe 100 维向量（下载一次后缓存）构建上述 TF-IDF 加权嵌入混合模型。在 20 Newsgroups 数据集上比较分类准确率：纯 TF-IDF、纯均值池化嵌入、混合模型。报告哪种方法在哪些情况下胜出。

## 关键术语

| 术语 | 大家说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一个文档中词汇表单词的计数。丢弃顺序。 |
| TF | 词频 | 一个单词在文档中出现的次数，可选地按文档长度归一化。 |
| DF | 文档频率 | 包含该词至少一次的文档数。 |
| IDF | 逆文档频率 | `log(N / df)` 的平滑版本。降低出现在所有文档中的词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常有 1 万到 10 万个词；大多数词在给定文档中均未出现。 |
| 余弦相似度 | 向量角度 | L2 归一化向量的点积。1 表示完全相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — 从文本中提取特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，以及每个参数的注释。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为十年默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法何时胜出以及原因的观点。
