# 主题建模 — LDA 和 BERTopic

> LDA：文档是主题的混合体，主题是词语上的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，基本元素不同。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 5 · 02（词袋模型 + TF-IDF）、阶段 5 · 03（Word2Vec）
**时长：** 约45分钟

## 问题

你有1万张客服工单、5万篇新闻文章或20万条推文。你需要在不通读全文的情况下了解这些内容是关于什么的。你没有标注好的类别，甚至不知道有多少个类别存在。

主题建模可以在无监督的情况下回答这个问题。给定一个语料库，返回一组连贯的主题，以及每篇文档在这些主题上的分布。

两个算法家族占据主导地位。LDA（2003）将每篇文档视为一组潜在主题的混合，每个主题是词语上的分布。推理采用贝叶斯方法。它仍然在生产环境中部署，当你需要混合成员归属的主题分配和可解释的词级概率分布时。

BERTopic（2020）使用BERT编码文档，用UMAP降维，用HDBSCAN聚类，并通过基于类别的TF-IDF提取主题词。它在短文本、社交媒体以及任何语义相似性比词重叠更重要的场景中胜出。一篇文档只得到一个主题，这对于长文本内容来说是一个局限。

本课将建立对两者的直观理解，并指明针对给定语料库应选用哪一个。

## 概念

![LDA混合模型 vs BERTopic聚类](../assets/topic-modeling.svg)

**LDA生成故事。** 每个主题是词语上的分布。每篇文档是主题的混合。要生成文档中的一个词，先从文档的混合中采样一个主题，然后从该主题的分布中采样一个词。推理过程反向进行：根据观察到的词，推断每篇文档的主题分布和每个主题的词分布。折叠吉布斯采样或变分贝叶斯负责数学计算。

LDA的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为1（主题的词分布）。

**BERTopic流程。**

1. 使用句子变换器（例如 `all-MiniLM-L6-v2`）对每篇文档进行编码。得到384维向量。
2. 用UMAP将维度降至约5维。BERT嵌入对于聚类来说维度过高。
3. 用HDBSCAN进行聚类。基于密度的算法，产生大小可变的聚类和一个“离群点”标签。
4. 对每个聚类，在聚类内文档上计算基于类别的TF-IDF，以提取最重要的词语。

输出结果是每篇文档一个主题（外加一个-1离群点标签）。可选地，通过HDBSCAN的概率向量获得软归属。

## 动手实现

### 第1步：通过scikit-learn实现LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# doc: list[str] — your corpus
vectorizer = CountVectorizer(
    max_df=0.85,
    min_df=5,
    stop_words="english",
    max_features=10_000,
)
doc_word = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=10, random_state=42)
doc_topic = lda.fit_transform(doc_word)

# top topic words per topic
vocab = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [vocab[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx:2d}: {', '.join(reversed(top_words))}")
```

注意：已移除停用词，`min_df` 和 `max_df` 过滤掉罕见和普遍出现的词，使用 `CountVectorizer`（而非 `TfidfVectorizer`），因为LDA期望原始计数。

### 第2步：BERTopic（生产级）

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
)
topics, probs = topic_model.fit_transform(docs)

# inspect top topics
topic_info = topic_model.get_topic_info()
topic_info[topic_info.Topic != -1].head(10)
```

对 `Topic != -1` 的过滤会丢弃BERTopic的离群桶（HDBSCAN无法聚类的文档）。`min_topic_size` 控制HDBSCAN的最小聚类大小；BERTopic库的默认值是10。此示例为了配合本课的规模而明确设为15。对于超过1万篇文档的语料库，请增加到50或100。

### 第3步：评估

两种方法都输出主题词。问题在于这些词是否凝聚成连贯的主题。

- **主题连贯性（c_v）。** 结合滑动窗口上下文中顶级词对的NPMI（归一化点互信息），将得分聚合成主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel`，参数 `coherence="c_v"`。
- **主题多样性。** 所有主题顶级词中不重复词的比例。越高越好（主题之间不重叠）。
- **定性检查。** 阅读每个主题的顶级词。它们是否指向一个真实的事物？人类判断仍然是最后一道防线。

## 何时选哪个

| 情况 | 选用 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 无GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 用于主题标注的LLM集成 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大化语义连贯性 | BERTopic |

最大的实际考量是文档长度。BERT嵌入会截断；LDA计数适用于任意长度。对于超过嵌入模型上下文的文档，可以分块后聚合，或者使用LDA。

## 实际运用

2026年的技术栈：

- **BERTopic。** 短文本以及任何语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典的LDA，用于生产环境，成熟且久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 用于实验的简易LDA。
- **NMF。** 非负矩阵分解。LDA的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 与BERTopic设计相似。社区较小，但在某些基准测试上表现良好。
- **FASTopic。** 较新，在非常大的语料库上比BERTopic更快。
- **基于LLM的标注。** 运行任意聚类，然后提示一个模型为每个聚类命名。

## 交付成果

保存为 `outputs/skill-topic-picker.md`：

```markdown
# Topic Model Selection Guide

## Problem characteristics
- Document length: ___ (short / medium / long)
- Expected topic mixture: ___ (single / multiple per doc)
- Compute budget: ___ (GPU / CPU)
- Deployment target: ___ (server / edge)

## Decision
- **Pick LDA if:** long documents, mixed-membership needed, CPU-only.
- **Pick BERTopic if:** short text, semantic similarity matters, GPU available.

## Model config
- LDA: n_components = ___
- BERTopic: min_topic_size = ___, embedding_model = ___

## Quality
- c_v coherence: ___
- Topic diversity: ___
- Human inspection passed? ___
```

## 练习

1. **简单。** 在20个新闻组数据集上用5个主题拟合LDA。打印每个主题的前10个词。手动为每个主题贴上标签。算法是否找到了真实的类别？
2. **中等。** 在相同的20个新闻组子集上拟合BERTopic。比较找到的主题数量、前几个词以及与LDA的定性连贯性。哪一个更清晰地呈现了真实的类别？
3. **困难。** 计算语料库中LDA和BERTopic的c_v连贯性。分别使用5、10、20、50个主题运行。绘制连贯性 vs 主题数量的图表。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Topic | 语料库所涉及的一个事物 | 词语上的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| Mixed membership | 文档由多个主题构成 | LDA为每篇文档分配一个在所有主题上的分布。 |
| UMAP | 降维 | 保持局部结构的流形学习；用于BERTopic。 |
| HDBSCAN | 密度聚类 | 找出大小可变的聚类；为离群点生成“噪声”标签（-1）。 |
| c_v coherence | 主题质量指标 | 滑动窗口内主题顶级词的平均点互信息。 |

## 延伸阅读

- Blei, Ng, Jordan (2003). Latent Dirichlet Allocation — LDA论文。
- Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure — BERTopic论文。
- Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures — 引入c_v等指标的论文。
- BERTopic documentation — 生产级参考。优秀的示例。
