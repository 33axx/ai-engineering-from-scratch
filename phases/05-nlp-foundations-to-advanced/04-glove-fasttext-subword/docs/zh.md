# GloVe、FastText 与子词嵌入

> Word2Vec 为每个词学习一个嵌入。GloVe 对共现矩阵进行分解。FastText 嵌入词的组成部分。BPE 为 Transformer 架起桥梁。

**类型：** 动手实践  
**语言：** Python  
**前置知识：** 阶段 5 · 03（从零实现 Word2Vec）  
**时长：** 约 45 分钟  

## 问题

Word2Vec 留下了两个待解决的问题。

首先，存在一条并行研究方向，它直接对共现矩阵进行分解（如 LSA、HAL），而不是进行在线 skip-gram 更新。Word2Vec 的迭代方法是否从根本上更好？还是两种方法处理计数方式的差异导致了结果不同？**GloVe** 回答了这个问题：使用精心设计的损失函数进行矩阵分解，其效果可以与 Word2Vec 相媲美甚至更优，并且训练成本更低。

其次，这两种方法都无法处理从未见过的词。比如 `Zoomer-approved`、`dogecoin`、上周才创造出来的专有名词，以及任何罕见词根的屈折变化形式。**FastText** 通过嵌入字符 n-gram 解决了这个问题：一个词是其组成部分（包括词素）的总和，因此即使是词表外的词也能获得合理的向量。

第三，Transformer 出现后，问题又发生了变化。词级别的词表最多只能包含一百万个条目，而实际语言远不止如此。**字节对编码（BPE）** 及其相关方法通过学习一个能覆盖所有内容的常见子词单元词表解决了这个问题。如今所有现代大语言模型的 tokenizer 都是子词 tokenizer。

本节课将讲解这三种方法，然后说明何时应该选择哪种方法。

## 核心概念

**GloVe（全局向量）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现的频率。训练向量，使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失进行加权，使得高频词对不会过分主导。完成。

**FastText。** 一个词是其所有字符 n-gram 加上词本身的向量之和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些成分向量的和。以 Word2Vec 的方式进行训练。好处：未见过的词（如 `whereupon`）可以通过已知的 n-gram 组合得到表示。

**BPE（字节对编码）。** 从一个包含单个字节（或字符）的词表开始。统计语料中每一对相邻单元的出现次数。将出现次数最多的相邻对合并为一个新 token。重复 `k` 次。结果：得到一个包含 `k + 256` 个 token 的词表，其中频繁出现的序列（如 `ing`、`tion`、`the`）成为单个 token，而罕见词则被拆分为熟悉的片段。每个句子都能被 token 化。

## 动手实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得提及的重要组件。加权函数 `f(x) = (x/x_max)^alpha` 降低了极高频率词对（如 `(the, and)`）的权重，使它们不会主导损失。最终的嵌入是 `W`（中心词表）和 `W_tilde`（上下文词表）的和。将两者相加是一种公开的技巧，通常比仅使用其中之一效果更好。

### FastText：子词感知的嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合（通常为 3 到 6 个字符）表示。词嵌入是其所有 n-gram 嵌入的总和。在 skip-gram 训练中，可以用它替代 Word2Vec 中使用的单个向量。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要其部分 n-gram 是已知的，你仍然能得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，因此两者会在向量空间中彼此靠近。

### BPE：学习子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。经过足够多次迭代后，频繁的子字符串（如 `low`、`est`、`tion`）会成为单个 token，而罕见词则被清晰拆分。

真实的 GPT / BERT / T5 tokenizer 会学习 30k-100k 次合并。结果：任何文本都能被 token 化成一个长度受限的已知 ID 序列，永远不会出现 OOV。

## 使用它们

在实际应用中，你很少需要自己训练这些模型。你会直接加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 Transformer 时代的 BPE 风格子词 tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀用于标记词边界（这是 GPT-2 的约定）。所有现代 tokenizer 都是 BPE、WordPiece（BERT）或 SentencePiece（T5、LLaMA）的变体。

### 何时选择哪种方法

| 场景 | 选择 |
|-------|------|
| 预训练的通用词向量，无需处理 OOV | GloVe 300d |
| 预训练的通用词向量，必须处理拼写错误/新词/形态丰富的语言 | FastText |
| 任何要输入 Transformer 的情况（训练或推理） | 模型自带的任何 tokenizer。切勿更换。 |
| 从头训练自己的语言模型 | 首先在语料库上训练 BPE 或 SentencePiece tokenizer |
| 使用线性模型进行生产文本分类 | 仍使用 TF-IDF。参见第 02 课。 |

## 交付

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠度。应该能看到大量的共享片段（`pla`、`lay`、`play`），这就是 FastText 能良好处理形态变体的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词表增长情况。绘制每个语料字符对应的 token 数随合并次数的变化图。起初你会看到快速压缩，然后渐近于每 token 约 2-3 个字符。
3. **困难。** 在莎士比亚全集中训练一个 1000 次合并的 BPE。比较常用词与罕见专有名词的 token 化结果。测量每个词在 token 化前后的平均 token 数。写下让你感到惊讶的地方。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|----------|
| 共现矩阵 | 词-词频次表 | `X[i][j]` = 词 `j` 在词 `i` 周围窗口内出现的频率。 |
| 子词 | 词的一部分 | 一个字符 n-gram（FastText）或学习到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻对，直到词表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 无法处理。FastText 和 BPE 可以处理。 |
| 字节级 BPE | 在原始字节上进行的 BPE | GPT-2 的方案。词表以 256 个字节开始，因此没有任何东西是 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍是对其损失函数的最佳推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — 说明 BPE、WordPiece 和 SentencePiece 在实际中如何不同。

--- 

注意：所有 ````python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
