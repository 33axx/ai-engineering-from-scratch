# 变换器之前的文本生成——N-gram 语言模型

> 如果某个词令人意外，说明模型差。困惑度把"意外"变成一个数值。平滑让这个数值保持有限。

**类型：** 动手实践  
**语言：** Python  
**前置知识：** 阶段 5 · 01（文本处理），阶段 2 · 14（朴素贝叶斯）  
**所需时间：** ~45 分钟

## 问题

在变换器之前，在循环神经网络之前，在词嵌入之前，语言模型是这样预测下一个词的：统计它前面 `n-1` 个词之后出现的次数。统计 "the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化后得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，每一个语音识别器、每一个拼写检查器、每一个基于短语的机器翻译系统都运行着它。当你需要廉价设备端语言建模时，它仍然在运行。

有趣的问题是如何处理未见过的 n-gram。一个基于原始计数的模型会为任何没见过的情况赋予零概率，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是其中的成果，现代深度学习继承了它的经验主义传统。

## 概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组取 3，四元组取 4）。通过计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未见过的 n-gram 都会被赋予零概率。2007 年一项针对 Brown 语料库的研究发现，即使是一个四元组模型，也有 30% 的保留四元组在训练中未见。没有平滑，你就无法在真实文本上评估模型。

**平滑方法，按复杂程度排序：**

1. **Laplace（加一平滑）。** 在所有计数上加 1。简单，但在罕见事件上表现很差。
2. **Good-Turing。** 根据频率的频率，将概率质量从高频事件重新分配给未见事件。
3. **插值。** 结合 n-gram、(n-1)-gram 等估计值，并带有可调权重。
4. **回退。** 如果 n-gram 计数为零，则回退到 (n-1)-gram。Katz 回退对此进行了归一化。
5. **绝对折扣。** 从所有计数中减去一个固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率*（一个词出现在多少种上下文中）而不是原始频率。

Kneser-Ney 的洞见很深刻。"San Francisco" 是一个常见的二元组。一元组 "Francisco" 主要出现在 "San" 之后。朴素的绝对折扣会给 "Francisco" 很高的单字概率（因为计数高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，因此降低了它的延续概率。结果：一个以 "Francisco" 结尾的新颖二元组获得了适当的低概率。

**评估：困惑度。** 测试集上每个词的平均负对数似然的指数。越小越好。困惑度为 100 意味着模型像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### 第 1 步：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是一个列表，其中每个元素是一个经过分词处理的句子。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界。

### 第 2 步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

在每个计数上加 1。平滑了，但将过多质量分配给了未见事件，反而损害了已知的罕见事件。

### 第 3 步：Kneser-Ney（二元组，插值版）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分。`continuation_prob` 捕捉了"这个词出现在多少种不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的质量，用于加权回退项。最终概率是折扣后的主项加上加权的延续项。

### 第 4 步：通过采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次以不同种子运行都会得到不同输出。若要类似束搜索的输出，每一步选 argmax（贪心），再加一个小的随机性旋钮（温度）。

### 第 5 步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越小越好。对于 Brown 语料库，一个调好的四元组 Kneser-Ney 模型的困惑度大约为 140。一个变换器语言模型在相同测试集上的困惑度为 15-30。差距约为 10 倍。正是这个差距推动了这个领域向前发展。

## 应用

- **经典 NLP 教学。** 接触平滑、最大似然估计和困惑度的最清晰途径。
- **KenLM。** 生产级 n-gram 库。用于对延迟敏感的语音和机器翻译系统中的重打分。
- **设备端自动补全。** 键盘上的三元组模型。现在还在用。
- **基线。** 在宣称你的神经语言模型好之前，始终先计算 n-gram 语言模型的困惑度。如果你的变换器没有大幅超过 Kneser-Ney，那就有问题。

## 交付

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在 1000 句莎士比亚语料上训练一个三元组语言模型。生成 20 个句子。这些句子局部合理但全局不连贯。这是经典 demo。
2. **中等。** 在你的 Kneser-Ney 模型上对保留的莎士比亚分割实现困惑度计算，并与 Laplace 比较。你应该会看到 Kneser-Ney 的困惑度低 30-50%。
3. **困难。** 构建一个三元组拼写纠正器：给定一个拼写错误的单词及其上下文，生成纠正候选，并按照语言模型下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|---------------|----------|
| N-gram | 词序列 | 由 `n` 个连续标记组成的序列。 |
| 平滑 | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| 困惑度 | 语言模型质量指标 | 在保留数据上计算 `exp(-平均对数概率)`。越小越好。 |
| 回退 | 回退到更短的上下文 | 如果三元组计数为零，则使用二元组。Katz 回退将其形式化。 |
| Kneser-Ney | n-gram 的最佳平滑方法 | 绝对折扣 + 低阶模型的延续概率。 |
| 延续概率 | Kneser-Ney 特有 | `P(w)` 按 `w` 出现的上下文数量加权，而不是原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram 语言模型和平滑的经典论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确定了 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — Kneser-Ney 原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的生产级 n-gram 语言模型，2026 年仍用于延迟敏感的应用。
