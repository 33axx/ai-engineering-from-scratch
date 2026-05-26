# 自然语言推理——文本蕴含

> "t蕴含h"意味着阅读t的人会推断h为真。NLI的任务是预测蕴含/矛盾/中立。表面平淡无奇，生产中却是承重墙。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段5·05（情感分析），阶段5·13（问答系统）
**时长：** ~60分钟

## 问题

你构建了一个摘要生成器，它生成了一段摘要。你怎么知道摘要里没有幻觉？

你构建了一个聊天机器人，它回答了"是"。你怎么知道这个回答有检索段落支持？

你需要将10,000篇新闻文章按主题分类，但没有训练标签。你能复用现有模型吗？

这三个问题都可以归结为自然语言推理。NLI的任务是：给定前提`t`和假设`h`，判断`h`是被`t`蕴含、矛盾，还是中立（无关）。

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的声明。非蕴含 = 幻觉。
- **有依据的问答：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类：** `t` = 文档，`h` = 标签的文本表述（"这是关于体育的"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个RAG评估框架底层都内置了一个NLI模型。

## 概念

![NLI：前提与假设的三路分类](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。"猫在垫子上"蕴含"有一只猫"。
- **矛盾。** `t` → ¬`h`。"猫在垫子上"与"没有猫"矛盾。
- **中立。** 无法推断出任何关系。"猫在垫子上"对"猫饿了"是中立的。

**不是逻辑蕴含。** NLI是*自然*语言推理——典型人类读者会推断出的结论，而非严格的逻辑。"约翰遛了他的狗"在NLI中蕴含"约翰有一条狗"，但严格的一阶逻辑只有在你对"拥有"进行公理化时才会承认这一点。

**数据集。**

- **SNLI** (2015年)。57万个人工标注的句子对，前提来自图像描述。领域狭窄。
- **MultiNLI** (2017年)。覆盖10种类型的43.3万对。2026年的标准训练语料。
- **ANLI** (2019年)。对抗式NLI。人类专门编写了旨在打破现有模型的例子。更难。
- **DocNLI, ConTRoL** (2020–2021年)。文档长度的前提。测试多跳和长距离推理。

**架构。** 一个Transformer编码器（BERT, RoBERTa, DeBERTa）读取`[CLS] 前提 [SEP] 假设 [SEP]`。`[CLS]`表示输入到一个3路softmax。在MNLI上训练，在保留的基准测试上评估，在分布内数据对上达到90%+的准确率。

**基于NLI的零样本学习。** 给定一个文档和候选标签，将每个标签转化为一个假设（"这段文本是关于体育的"）。计算每个标签的蕴含概率。选择概率最大的。这就是Hugging Face的`zero-shot-classification`流水线背后的机制。

## 动手实现

### 第一步：运行预训练的NLI模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产环境中的NLI，`facebook/bart-large-mnli`和`microsoft/deberta-v3-large-mnli`是开源默认选择。DeBERTa-v3在排行榜上领先。

### 第二步：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是"This example is about {label}。"。可以使用`hypothesis_template`自定义。无需训练数据，无需微调，开箱即用。

### 第三步：RAG的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是RAGAS忠实度评估的核心。将生成的答案分解为原子性声明，将每个声明与检索到的上下文进行比对，报告蕴含声明的比例。

### 第四步：手写NLI分类器（概念性）

参见 `code/main.py` 了解一个仅使用标准库的玩具实现：通过词汇重叠加否定检测来比较前提和假设。无法与Transformer模型竞争——但它展示了任务形态：两段文本输入，3路标签输出，损失函数为`{蕴含， 矛盾， 中立}`上的交叉熵。

## 常见陷阱

- **仅依赖假设的捷径。** 模型仅从假设就能以约60%的准确率预测SNLI标签，因为"不"、"没有人"、"从不"等词与矛盾相关。这是检测标签泄漏的强基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能在SNLI上通过，但在HANS/ANLI上会失败。应使用对抗性基准测试。
- **文档级性能退化。** 单句NLI模型在文档长度的前提上F1下降20+分。对于长上下文，应使用在DocNLI上训练过的模型。
- **零样本模板敏感性。** "This example is about {label}"对比"{label}"对比"The topic is {label}"可能导致准确率波动10个百分点以上。应调整模板。
- **领域不匹配。** MNLI在通用英语上训练。法律、医学和科学文本需要领域特定的NLI模型（例如SciNLI, MedNLI）。

## 应用建议

**2026年技术栈：**

| 用例 | 模型 |
|---------|-------|
| 通用NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘设备 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG中的幻觉检测 | RAGAS / DeepEval内部的NLI层 |

**2026年元模式：** NLI是文本理解领域的"胶带"。只要遇到"A是否支持B？"或"A是否与B矛盾？"的需求——在考虑另一次LLM调用之前，先使用NLI。

## 交付物

保存为`outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习题

1. **简单。** 用`facebook/bart-large-mnli`在20个人工构造的（前提，假设，标签）三元组上运行，覆盖所有三个类别。测量准确率。加入对抗性的"子序列启发式"陷阱（"我没有吃蛋糕" vs "我吃了蛋糕"），观察模型是否失效。
2. **中等。** 在100个AG News标题上，比较零样本模板`"This text is about {label}"`与`"The topic is {label}"`和`"{label}"`。报告准确率的波动。
3. **困难。** 构建一个RAG忠实度检查器：原子声明分解 + 每声明NLI。在50个有金标准上下文的RAG生成答案上进行评估。与人工标注相比，衡量假阳性率和假阴性率。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 对前提-假设关系进行3路分类。 |
| RTE | 识别文本蕴含 | NLI的旧称；任务相同。 |
| 蕴含 | "t隐含h" | 典型读者在给定t时会推断h为真。 |
| 矛盾 | "t排除了h" | 典型读者在给定t时会推断h为假。 |
| 中立 | "无法判断" | 无法从t向h作出任何推断。 |
| 零样本分类 | 将NLI用作分类器 | 将标签表述为假设，选择最大蕴含概率。 |
| 忠实度 | 答案是否有依据？ | 对（检索到的上下文，生成的答案）进行NLI。 |

## 扩展阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI基准测试。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026年NLI的主力模型。
