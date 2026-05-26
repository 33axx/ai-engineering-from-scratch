# 命名实体识别

> 把名字提取出来。听起来简单，直到你遇到模糊的边界、嵌套实体和领域术语。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 03（词嵌入）
**时长：** ~75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（组织）、Google（组织）、iPhone（产品）、search deal（可能是个实体）、US（地缘政治实体）。好的 NER 系统会正确提取它们全部并给出正确类型。糟糕的系统会漏掉 iPhone，把水果 Apple 和公司 Apple 搞混，并且把 "US" 标注为人物。

NER 是所有结构化提取流水线背后的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的接地、法律合同提取。你很少直接看到它，但你总是依赖它。

本课程沿着经典路径（规则方法、隐马尔可夫模型、条件随机场）走向现代路径（双向长短期记忆网络-条件随机场，然后是基于 Transformer 的方法）。每一步都解决了前一步的特定局限性。这个模式本身就是课程内容。

## 概念

**BIO 标注**（或 BILOU）把实体提取变成了序列标注问题。给每个词符标注 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（不在任何实体中）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词符实体链：`New B-GPE`、`York I-GPE`、`City I-GPE`。一个理解 BIO 的模型可以提取任意片段。

架构演进：

- **规则方法。** 正则 + 词典查找。对已知实体精确率高，对未知实体覆盖率为零。
- **隐马尔可夫模型。** 隐马尔可夫模型。词符给定标签的发射概率，标签到标签的转移概率。维特比解码。在标注数据上训练。
- **条件随机场。** 条件随机场。类似隐马尔可夫模型但是判别式，所以你可以混合任意特征（单词形状、大写形式、邻近词）。在 2026 年，它仍然是低资源部署的经典生产主力。
- **双向长短期记忆网络-条件随机场。** 神经网络特征代替手工特征。双向 LSTM 读取句子，顶层的 CRF 层强制标签序列一致性。
- **基于 Transformer。** 使用词符分类头微调 BERT。准确率最高。计算量最大。

## 构建

### 第 1 步：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：手工特征

对于经典（非神经）NER，特征就是关键。有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大写模式对专有名词来说信号很强。

### 第 3 步：简单的规则方法 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境中的词典有数百万个从维基百科和 DBpedia 爬取的条目。覆盖率很好。消歧（公司 `Apple` vs 水果 `Apple`）则很糟糕。这就是为什么统计模型胜出了。

### 第 4 步：CRF 步骤（草图，非完整实现）

用 50 行代码从零实现完整 CRF 没有概率论基础是不清晰的。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习到非法序列（例如 `O` 后面的 `I-ORG`）是不太可能的，这就是 CRF 如何在不写约束的情况下强制 BIO 一致性。

### 第 5 步：BiLSTM-CRF 增加了什么

特征变成了学习得到的。输入：词符嵌入（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。拼接后的隐藏状态经过 CRF 输出层。CRF 仍然强制标签序列一致性；LSTM 用学习到的特征取代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相对于手工 CRF 的提升是可测量的，但除非你有数万条标注句子，否则提升幅度小于你的预期。

## 使用

spaCy 开箱即用地提供了生产级的 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标注为 `ORG` 而不是 `PRODUCT` —— spaCy 的小模型对产品实体覆盖较弱。大模型（`en_core_web_lg`）表现更好。Transformer 模型（`en_core_web_trf`）则更好。

Hugging Face 用于基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会将连续的 B-X、I-X 词符合并成一个片段。没有它，你会得到词符级别的标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选择）

零样本和少样本的 LLM NER 现在在许多领域与微调模型竞争，并且在标注数据稀缺时显著更好。

- **零样本提示。** 给 LLM 一个实体类型列表和一个示例 schema。要求输出 JSON。开箱即用；在陌生领域的准确率中等。
- **类似 ZeroTuneBio 的提示。** 将任务分解为候选提取 → 含义解释 → 判断 → 复核。多阶段提示（非单次）在生物医学 NER 上显著提升了准确率。同样的模式适用于法律、金融和科学领域。
- **结合 RAG 的动态提示。** 对每个推理调用，从少量标注的种子集中检索最相似的标注示例；动态构建少样本提示。在 2026 年的基准测试中，这使 GPT-4 的生物医学 NER F1 比静态提示提升了 11-12%。
- **按实体类型分解。** 对于长文档，一次调用提取所有实体类型会随着长度增加而损失召回率。每个实体类型运行一次提取。更高的推理成本，显著更高的准确率。这是临床笔记和法律合同的标准模式。

截至 2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本基线。通常 F1 已经足够好，你永远不需要微调。

### 经典 NER 仍然赢的地方

即使有了 LLM，经典 NER 在以下情况仍然胜出：

- 延迟预算低于 50 毫秒。
- 你有数千个标注样本，需要 98%+ 的 F1。
- 领域具有稳定的本体，预训练的 CRF 或 BiLSTM 可以良好迁移。
- 监管要求使用本地、非生成式模型。

### 它在哪里失败

- **领域迁移。** 在 CoNLL 上训练的 NER 在法律合同上表现比词典还差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是一个组织和一个设施。标准 BIO 无法表示重叠片段。你需要嵌套 NER（多遍或基于片段的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." 词符级模型有时会拆分它。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型对此一无所知。Scispacy 和 BioBERT 是起点。

## 发布

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作）并在 10 个句子上验证 round-trip 一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上面的 sklearn-crfsuite CRF。使用 `seqeval` 报告每个实体的 F1。典型结果：约 84 F1。
3. **困难。** 在领域特定 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型比较。记录数据泄露检查，并写下让你惊讶的内容。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| NER | 提取名字 | 用类型（人物、组织、地缘政治实体、日期等）标注词符片段。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（最后一个）、`U-X`（单个）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 对标签之间的转移进行建模，而不仅仅是发射概率。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个片段与其子片段的实体类型不同。BIO 无法表达这种情况。 |
| 实体级 F1 | 正确的 NER 指标 | 预测的片段必须与真实片段完全匹配。词符级 F1 会高估准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) —— BiLSTM-CRF 论文。经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) —— 介绍了后来成为标准的词符分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) —— 关于 `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) —— 正确的指标库。请一直使用它。
