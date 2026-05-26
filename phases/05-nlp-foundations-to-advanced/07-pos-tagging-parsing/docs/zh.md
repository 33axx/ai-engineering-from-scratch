# 词性标注与句法分析

> 语法曾一度不受重视。后来每个大语言模型管道都需要验证结构化抽取，语法便卷土重来。

**类型：** 实战  
**语言：** Python  
**前置知识：** 阶段 5 · 第 01 课（文本处理）、阶段 2 · 第 14 课（朴素贝叶斯）  
**时长：** ~45 分钟

## 问题描述

第 01 课曾承诺：词形还原需要知道词性标签。如果不清楚 `running` 是动词，词形还原器就无法将其还原为 `run`；如果不清楚 `better` 是形容词，就无法还原为 `good`。

这个承诺背后隐藏着一整个子领域。词性标注负责分配语法类别，句法分析则恢复句子的树状结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年的时间来精炼这两个任务。后来深度学习把它们简化为基于预训练 Transformer 的词元分类任务，研究社区便转向了其他方向。

但应用社区没有转向。每个结构化抽取管道仍然在底层使用 POS 和依存树。LLM 生成的 JSON 需要依据语法约束进行验证。问答系统通过依存分析来分解查询。机器翻译质量评估器会检查解析树的对齐情况。

值得了解。本课程将介绍标签集、基线方法，以及你什么时候该停止从头实现而直接调用 spaCy。

## 概念

**词性标注**为每个词元赋予一个语法类别。**宾州树库**标签集是英语的默认标签集。总共 36 个标签，有些区分在随意的读者看来有些吹毛求疵：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**通用依存关系**标签集则更粗糙（17 个标签），且与语言无关；它已成为跨语言工作的默认选择。

```python
# 词性标注示例
# "The quick brown fox jumps over the lazy dog."
# Penn Treebank 标签: DT JJ JJ NN VBZ IN DT JJ NN .
# 通用依存关系 (UD) 标签: DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN PUNCT
```

**句法分析**生成一棵树。主要有两种风格：

- **成分句法分析**。名词短语、动词短语、介词短语层层嵌套。输出是一棵由非终端类别（NP、VP、PP）构成的树，单词作为叶节点。
- **依存句法分析**。每个单词有一个它所依存的中心词，并标有语法关系。输出是一棵树，每条边是一个（中心词、依存词、关系）三元组。

依存句法分析在 2010 年代胜出，因为它能干净地泛化到不同语言，尤其是自由语序的语言。

```python
# 依存分析示例（spaCy 风格）
# "The quick brown fox jumps over the lazy dog."
# 依存弧:
#   jumps -> fox (nsubj)
#   jumps -> over (prep)
#   over -> dog (pobj)
#   fox -> The (det)
#   fox -> quick (amod)
#   fox -> brown (amod)
#   dog -> the (det)
#   dog -> lazy (amod)
#   jumps -> . (punct)
```

## 动手构建

### 第 1 步：最常见标签基线

最笨但有效的词性标注器。对于每个单词，预测它在训练集中出现次数最多的标签。

```python
from collections import defaultdict, Counter
import nltk
from nltk.corpus import brown

# 加载带标签的数据
tagged_sents = brown.tagged_sents(categories='news')
train_sents = tagged_sents[:400]
test_sents = tagged_sents[400:500]

# 为每个单词累计标签计数
word_tag_counts = defaultdict(Counter)
for sent in train_sents:
    for word, tag in sent:
        word_tag_counts[word][tag] += 1

# 为每个单词选择最常见标签，未登录词默认为 'NN'
most_frequent_tag = {}
for word, tag_counter in word_tag_counts.items():
    most_frequent_tag[word] = tag_counter.most_common(1)[0][0]

def baseline_tagger(sent):
    """对句子中的每个单词使用最常见标签进行标注"""
    return [most_frequent_tag.get(word, 'NN') for word in sent]

# 评估
def accuracy(tagged_sents, tagger):
    correct = 0
    total = 0
    for sent in tagged_sents:
        words = [word for word, tag in sent]
        gold_tags = [tag for word, tag in sent]
        pred_tags = tagger(words)
        for gold, pred in zip(gold_tags, pred_tags):
            if gold == pred:
                correct += 1
            total += 1
    return correct / total

print(f"Baseline accuracy: {accuracy(test_sents, baseline_tagger):.3f}")
```

在布朗语料库上，这个基线能达到约 85% 的准确率。不算好，但这已经是任何正经模型都不该低于的下限。

### 第 2 步：二元 HMM 标注器

对序列的联合概率进行建模：

```
P(W, T) = P(T1) * P(T2 | T1) * ... * P(Tn | Tn-1) * P(W1 | T1) * ... * P(Wn | Tn)
```

两个表：转移概率（给定前一个标签时的当前标签）、发射概率（给定标签时的单词）。从计数中估算两者并应用拉普拉斯平滑。使用维特比算法（在标签格上的动态规划）进行解码。

```python
import numpy as np
from collections import defaultdict, Counter

# 收集计数（简化版——实际应用中需要处理未知词和 OOV 标签）
tag_counter = Counter()
transition_counts = defaultdict(lambda: defaultdict(int))
emission_counts = defaultdict(lambda: defaultdict(int))

for sent in train_sents:
    tags = [tag for word, tag in sent]
    for i, tag in enumerate(tags):
        tag_counter[tag] += 1
        emission_counts[tag][sent[i][0]] += 1
        if i > 0:
            transition_counts[tags[i-1]][tag] += 1

# 带平滑的概率（拉普拉斯 +1）
tags = list(tag_counter.keys())
V = len(tags)  # 标签数

# 转移矩阵 A[t_i][t_j] = P(t_j | t_i)
# 发射矩阵 B[t][w] = P(w | t)
# 未知词：为每个标签分配一个小的无符号计数（这里简化为 1/总计数）

# 维特比算法（伪代码）
def viterbi(words, tags, A, B, start_prob, unknown_prob):
    """
    words: 输入句子
    tags: 所有可能的标签
    A: 转移概率 dict[t][t]
    B: 发射概率 dict[t][w]
    start_prob: 初始标签概率 dict[t]
    unknown_prob: 未知词发射概率（标量）
    """
    n = len(words)
    T = len(tags)
    # 使用对数概率以避免下溢
    viterbi_matrix = np.full((n, T), -np.inf)
    backpointer = np.zeros((n, T), dtype=int)
    
    # 初始化第一列
    for i, tag in enumerate(tags):
        emission = B[tag].get(words[0], unknown_prob)
        viterbi_matrix[0][i] = start_prob[tag] + np.log(emission) if emission > 0 else -np.inf
    
    # 递归
    for t in range(1, n):
        for i, tag in enumerate(tags):
            # 找到前一个标签 j 使得概率最大
            best_score = -np.inf
            best_j = 0
            for j, prev_tag in enumerate(tags):
                trans_prob = A[prev_tag].get(tag, 1e-10)  # 平滑
                emission_prob = B[tag].get(words[t], unknown_prob)
                if emission_prob == 0:
                    emission_prob = 1e-10
                score = viterbi_matrix[t-1][j] + np.log(trans_prob) + np.log(emission_prob)
                if score > best_score:
                    best_score = score
                    best_j = j
            viterbi_matrix[t][i] = best_score
            backpointer[t][i] = best_j
    
    # 回溯
    best_path = [np.argmax(viterbi_matrix[n-1])]
    for t in range(n-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])
    return [tags[i] for i in best_path]
```

在布朗语料库上，二元 HMM 准确率约为 93%。从 85% 提升到 93% 主要归功于转移概率——模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很罕见。

### 第 3 步：为什么现代标注器能超越这个水平

转移概率 + 发射概率是局部的。它们无法捕捉到 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词这样的差异。一个带有任意特征（后缀、词形、前后单词、单词本身）的 CRF 能达到约 97%。BiLSTM-CRF 或 Transformer 可以达到 98% 以上。

这个任务的天花板由标注者之间的不一致性决定。人类标注者在宾州树库上的意见一致性约为 97%。超过 98% 的模型很可能是在过拟合测试集。

### 第 4 步：依存句法分析概要

从头实现完整的依存句法分析超出了本课程的范围；经典的教科书讲解请参见 Jurafsky 和 Martin。需要了解两个经典流派：

- **基于转换的**句法分析器（arc-eager、arc-standard）类似于移位-归约分析器：它们读取词元，将其移至栈上，然后应用创建弧的归约动作。贪婪解码速度快。经典实现是 MaltParser。现代神经网络版：Chen 和 Manning 的基于转换分析器。
- **基于图的**句法分析器（Eisner 算法、Dozat-Manning 双仿射）对每一对可能的中心词-依存词进行打分，并选出最大生成树。速度较慢但准确率更高。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

for token in doc:
    print(f"{token.text:10} -> head: {token.head.text:10} dep: {token.dep_:10}")
```

```text
The        -> head: fox        dep: det      
quick      -> head: fox        dep: amod     
brown      -> head: fox        dep: amod     
fox        -> head: jumps      dep: nsubj    
jumps      -> head: jumps      dep: ROOT     
over       -> head: jumps      dep: prep     
the        -> head: dog        dep: det      
lazy       -> head: dog        dep: amod     
dog        -> head: over       dep: pobj     
.          -> head: jumps      dep: punct    
```

从下往上读 `dep` 列，句子的语法结构便一目了然。

## 如何使用

每个生产级 NLP 库都将 POS 和依存句法分析器作为标准管道的一部分。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。速度快、准确率高，与分词、NER、词形还原集成在一起。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 的 CoreNLP 后继者。在 60 多种语言上达到最先进水平。
- **trankit**。基于 Transformer，UD 准确率不错。
- **NLTK**。`pos_tag`。可用但速度慢且版本旧。适合教学。

### 2026 年这个领域仍然重要的原因

- **词形还原。** 第 01 课需要 POS 才能正确还原词形。这一点始终不变。
- **从 LLM 输出中结构化抽取。** 验证生成句子是否遵循语法约束（例如主谓一致、必要修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析被分解为结构化约束条件。
- **跨语言迁移。** UD 标签和依存关系与语言无关，使得对新兴语言进行零样本结构化分析成为可能。
- **低计算量管道。** 如果你无法部署 Transformer，那么 POS + 依存分析 + 查找表就已经能走得很远。

## 交付物

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
# Grammar Pipeline Report

## 词性标注基线准确率
- 最常见标签基线: 0.851 (布朗语料库，新闻类别，100 句测试集)
- 二元 HMM: 0.928 (同一测试集)

## 错误分析（二元 HMM 在布朗语料库上）
最常见的混淆对:
- NN ↔ NNS (复数 vs 单数名词混淆)
- VBD ↔ VBN (过去式 vs 过去分词)
- JJ ↔ NN (形容词 vs 名词，例如 "running" 在 "running water" 中)

## spaCy 依存分析检查
- 在 "The quick brown fox jumps over the lazy dog." 上测试
- 所有弧均正确标注 (使用 en_core_web_sm)
- 注意: spaCy 3.x 默认使用基于转换的分析器

## 生产用例清单
- [x] 词形还原需要 POS
- [ ] 从 LLM 输出中验证结构化抽取
- [ ] 基于方面的情感（依存关系）
- [ ] 查询理解与分解
- [ ] 跨语言场景（UD 标签）
```

## 练习

1. **简单题。** 使用最常见标签基线在一个小型标注语料库（例如 NLTK 的布朗语料子集）上，测量在保留句子上的准确率。验证约 85% 的结果。
2. **中等题。** 训练上述二元 HMM，并报告每个标签的精确率和召回率。HMM 最常混淆哪些标签？
3. **困难题。** 使用 spaCy 的依存句法分析从 1000 句样本中提取主谓宾三元组。在 50 个人工标注的三元组上进行评估。记录提取失败的地方（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 大家说的 | 实际含义 |
|------|---------|---------|
| POS 标签 | 单词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| 宾州树库 | 标准标签集 | 英语专用。细分的动词时态和名词数量。 |
| 通用依存关系 | 多语言标签集 | 比 PTB 粗糙；语言无关；跨语言工作的默认选择。 |
| 依存句法分析 | 句子树 | 每个单词有一个中心词，每条边有一个语法关系。 |
| 维特比算法 | 动态规划 | 在给定发射和转移概率下，找到概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 关于 POS 和句法分析的经典教科书讲解。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器所使用的跨语言标签集与树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 对象上每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器引入主流的论文。
