# 情感分析

> 经典的 NLP 任务。关于经典文本分类的大部分知识都体现在这里。

**类型：** 构建  
**语言：** Python  
**先修知识：** 阶段 5 · 02（词袋模型 + TF-IDF），阶段 2 · 14（朴素贝叶斯）  
**用时：** ~75 分钟  

## 问题

“食物不太好。” 正面还是负面？

情感听起来很简单。评论者说他们喜欢或不喜欢某样东西。给句子打上标签。它之所以成为经典的 NLP 任务，是因为每个看似简单的案例背后都隐藏着一个难题。否定翻转含义。讽刺使其颠倒。“一点也不差” 尽管有两个负面编码的词，却是正面的。表情符号比周围文本携带更多信号。领域词汇很重要（音乐评论中的 `tight` 与时尚评论中的 `tight` 含义不同）。

情感是经典 NLP 的实战实验室。如果你理解为什么每个朴素基线都有特定的失败模式，你就理解为什么每个更丰富的模型被发明出来。本课程从头构建了一个朴素贝叶斯基线，添加了逻辑回归，并指出了那些使生产级情感分析成为合规级问题的陷阱。

## 概念

经典情感分析是一个两步配方。

1. **表示**。将文本转换为特征向量。词袋模型、TF-IDF 或 n-gram。
2. **分类**。在标注样本上拟合线性模型（朴素贝叶斯、逻辑回归、支持向量机）。

朴素贝叶斯是能工作的最简单模型。假设在给定标签的条件下，每个特征相互独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推断时，将概率相乘。“朴素”的独立性假设荒唐地错误，但结果却惊人地强大。原因在于：当文本特征稀疏且数据量适中时，分类器更关心每个词倾向哪一边，而不是倾向多少。

逻辑回归修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个二元组特征会得到一个负权重。朴素贝叶斯无法为它从未标记过的二元组做到这一点。

## 构建

### 第一步：一个真正的小数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意小规模。实际工作使用数万个样本（IMDb、SST-2、Yelp 极性）。数学是一样的。

### 第二步：从头实现多项朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）即拉普拉斯平滑。没有它，一个在某个类别中未见过的词概率为零，对数会爆炸。`alpha=0.01` 在实践中常见。`alpha=1.0` 是教学默认值。

### 第三步：从头实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。从 `0.01` 开始调优。

### 第四步：处理否定（失败模式）

考虑“不好”和“不坏”。词袋模型分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中出现较多的那个中学习。二元组分类器看到 `not_good` 和 `not_bad`，并将它们作为不同的特征学习。这通常就足够了。

在没有二元组时也有效的更粗暴的修复方法：**否定作用域**。在否定词之后的 tokens 前缀 `NOT_`，直到下一个标点符号。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们相反的权重。三行预处理，在情感基准测试上可测量的准确率提升。

### 第五步：真正重要的评估指标

如果类别不平衡，单独准确率可能会误导。真实的情感语料库通常是 70-80% 正面或 70-80% 负面；一个常数多数类分类器能获得 80% 的准确率，但毫无价值。报告以下每一项：

- **每个类别的精确率和召回率**。每个类别一对。对它们进行宏平均，得到一个尊重类别平衡的单一数值。
- **宏平均 F1（不平衡数据的主要指标）** 。每个类别 F1 分数的平均值，等权重。当类别不平衡时，用它代替准确率。
- **加权平均 F1（替代方案）** 。与宏平均相同，但按类别频率加权。当不平衡本身具有商业意义时，与宏平均 F1 一起报告。
- **混淆矩阵**。原始计数。在信任任何标量指标之前总是先检查它；它能揭示模型混淆了哪些类别对。
- **每个类别的错误样本**。每个类别抽取 5 个错误预测。阅读它们。没有什么能替代阅读实际错误。

对于严重不平衡的数据（> 95-5 比率），报告 **AUROC** 和 **AUPRC** 而不是准确率。AUPRC 对少数类更敏感，而少数类通常是你关心的（垃圾邮件、欺诈、稀有情感）。

**要避免的常见错误。** 在不平衡数据上报告微平均 F1 而不是宏平均 F1，会得到一个看起来很高的数字，因为它被多数类主导。宏平均 F1 迫使你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用

scikit-learn 用六行正确完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

需要注意三件事。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元组，使 `not_good` 成为一个特征。`sublinear_tf=True` 抑制重复单词。这三个标志是 SST-2 上 75% 准确率基线与 85% 准确率基线之间的区别。

### 何时转向 Transformer

- 讽刺检测。经典模型在这里会失败。句号。
- 情感在文档中间发生变化的较长评论。
- 基于方面的情感分析。“相机很好，但电池很糟糕。” 你需要将情感归因于方面。只能用 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 免费为你提供一个零样本基线。

如果你需要上述任何一项，请跳到阶段 7（Transformer 深入探讨）。否则，TF-IDF 加上二元组加上否定处理的朴素贝叶斯或逻辑回归就是你的 2026 年生产基线。

### 可重复性陷阱（再次）

重新训练情感模型是常规操作。重新评估它们则不然。论文中报告的准确率使用了特定的划分、特定的预处理、特定的分词器。如果你将新模型与基线进行比较而不使用完全相同的流水线，你会得到误导性的差异。始终在你的流水线上重新生成基线，而不是使用论文中的数字。

## 交付

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单**。将 `apply_negation` 作为预处理步骤添加到 scikit-learn 流水线中，并在小情感数据集上测量 F1 的差值。
2. **中等**。实现类别加权逻辑回归（将 `class_weight="balanced"` 传给 scikit-learn，或者自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **困难**。通过训练第二个分类器来拟合情感模型的残差，构建一个讽刺检测器。记录你的实验设置。当你的准确率低于随机水平时警告读者（二分类讽刺的随机水平约为 50%，大多数初次尝试都落在那里）。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|----------|
| 极性 | 正面或负面 | 二元标签；有时扩展到中性或细粒度（五星）。 |
| 基于方面的情感分析 | 按方面的极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定作用域 | 反转附近的 tokens | 在“not”之后的 tokens 前缀 `NOT_` 直到标点符号。 |
| 拉普拉斯平滑 | 给计数加 1 | 防止朴素贝叶斯中出现概率为零的特征。 |
| L2 正则化 | 缩小权重 | 在损失中加入 `lambda * sum(w^2)`。对于稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础性综述。很长，但前四节涵盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 这篇论文表明，在短文本上二元组加朴素贝叶斯难以被超越。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及你会调优的每个参数的参考文档。
