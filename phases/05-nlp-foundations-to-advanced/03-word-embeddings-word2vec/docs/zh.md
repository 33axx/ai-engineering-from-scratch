# Word Embeddings — Word2Vec从零实现

> 词由它的同伴决定。基于这个想法训练一个浅层网络，几何特性就会涌现。

**类型：** 动手实现
**语言：** Python
**前置知识：** 第5阶段·02 (词袋 + TF-IDF)，第3阶段·03 (从零实现反向传播)
**时间：** ~75分钟

## 问题

TF-IDF知道`dog`和`puppy`是不同的词。但它不知道它们意思几乎相同。一个在`dog`上训练的分类器无法泛化到关于`puppy`的评论。你可以通过列出同义词来掩盖这个问题，但这在罕见术语、领域术语以及你未预料到的每一种语言上都会失败。

你希望得到一种表示，使得`dog`和`puppy`在空间中距离很近。使得`king - man + woman`接近`queen`。使得在`dog`上训练的模型可以免费地将一些信号迁移到`puppy`上。

Word2Vec提供了这样的空间。一个两层神经网络，在数万亿token上训练，于2013年发表。这个架构简单得令人尴尬。其结果重塑了NLP十年之久。

## 概念

**分布假说** (Firth, 1957)："由词的同伴可得知其义。"如果两个词在相似的上下文中出现，它们可能意思相近。

Word2Vec有两种变体，都利用了这一思想：
- **Skip-gram.** 给定中心词，预测其周围的词。`cat -> (the, sat, on)`，窗口大小为2。
- **CBOW (连续词袋).** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram训练较慢，但能更好地处理罕见词。它成为了默认选择。

网络具有一个隐藏层，没有非线性激活。输入是词汇表上的one-hot向量。输出是词汇表上的softmax。训练后，丢弃输出层。隐藏层权重即为嵌入向量。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧：对10万个词进行softmax计算代价过高。Word2Vec使用**负采样**将其转化为二分类任务。预测"这个上下文词是否出现在此中心词附近，是或否"。对每个训练对采样几个负例（非共现）词，而不是对整个词汇表计算softmax。

## 动手实现

### 步骤1：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每个(中心词, 上下文词)对都是一个正训练样本。

### 步骤2：嵌入表

两个矩阵。`W`是中心词嵌入表（你需要保留的那个）。`W'`是上下文词表（通常被丢弃，有时与`W`取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇量1万和维度100是实际场景；教学中，词汇量50和维度16足以看到几何特性。

### 步骤3：负采样目标

对于每个正对`(center, context)`，从词汇表中随机采样`k`个词作为负例。训练模型使得正对的点积`W[center] · W'[context]`较大，负对的点积较小。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：对正对使用逻辑损失（希望sigmoid接近1），对负对使用逻辑损失（希望sigmoid接近0）。梯度流向两个表。完整推导见原始论文；如果想牢固掌握，建议用纸笔推导一遍。

### 步骤4：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大规模语料上经过足够多的epoch后，共享上下文的词具有相似的中心嵌入。在玩具语料上，效果微弱。在上十亿token上，效果显著。

### 步骤5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的300维Google News向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道皇室是什么。而是因为向量`(king - man)`捕捉到了类似"皇室"的东西，将其加到`woman`上，就落在了皇室-女性区域附近。

## 使用它

从零实现Word2Vec是为了教学。实际生产NLP使用`gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

在实际工作中，几乎不会自己训练Word2Vec。可以下载预训练向量：
- **GloVe** — 斯坦福的共现矩阵分解方法。有50d、100d、200d、300d的检查点。通用覆盖良好。第04课专门讨论GloVe。
- **fastText** — Facebook的Word2Vec扩展，嵌入字符n-gram。通过组合子词处理OOV词。第04课。
- **Google News上的预训练Word2Vec** — 300维，300万词汇量，发表于2013年。至今仍被每日下载。

### 2026年Word2Vec仍然胜出的场景

- 轻量级领域特定检索。在笔记本电脑上花一小时在医学摘要上训练，获得通用模型无法捕捉的专门向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。将其从其他词中减去得到中性轴。仍在公平性研究中使用。
- 可解释性。100维足够通过PCA或t-SNE可视化，并实际看到聚类形成。
- 任何推理必须在设备上运行且无GPU的场景。Word2Vec查找只是单行取回。

### Word2Vec的失效之处

多义词困境。`bank`只有一个向量。`river bank`和`financial bank`共享它。`table`（电子表格 vs. 家具）共享它。下游分类器无法从向量中区分词义。

上下文嵌入（ELMo、BERT以及之后的每个transformer）通过基于周围上下文为每个词的出现产生不同向量解决了这个问题。这是从Word2Vec到BERT的跨越：从静态到上下文。第7阶段涵盖transformer部分。

OOV问题是另一个失败。如果未在训练数据中出现，Word2Vec从未见过`Zoomer-approved`。无后备方案。fastText通过子词组合（第04课）修复了这个问题。

## 交付

保存为`outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单.** 在玩具语料（20句关于猫狗的句子）上运行训练循环。200个epoch后，验证`nearest(vocab, W, W[vocab["cat"]])`返回的前3中包含`dog`。如果没有，增加epoch或词汇量。
2. **中等.** 添加频繁词下采样。频率高于`10^-5`的词按照与其频率成比例的概率从训练对中丢弃。测量对罕见词相似度的影响。
3. **困难.** 在20 Newsgroups语料上训练模型。计算两个偏差轴：`he - she`和`doctor - nurse`。将职业词投影到两个轴上。报告哪些职业具有最大的偏差差距。这是公平性研究者使用的那种探测方法。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-------------|---------|
| Word embedding | 词作为向量 | 从上下文中学习到的密集、低维（通常100-300）表示。 |
| Skip-gram | Word2Vec技巧 | 从中心词预测上下文词。比CBOW慢，但对罕见词更好。 |
| Negative sampling | 训练捷径 | 用对`k`个随机词的二分类替代对整个词汇表的softmax。 |
| Static embedding | 每个词一个向量 | 无论上下文都相同向量。在多义词上失败。 |
| Contextual embedding | 上下文敏感向量 | 基于周围词为每个出现产生不同向量。transformer产生的结果。 |
| OOV | 词汇外 | 训练中未见的词。Word2Vec无法为这些词生成向量。 |

## 进一步阅读

- [Mikolov等人 (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果原始论文数学较密集。
- [gensim Word2Vec教程](https://radimrehurek.com/gensim/models/word2vec.html) — 实际有效的生产训练设置。
