# 多语言NLP

> 一个模型，100多种语言，其中绝大部分没有任何训练数据。跨语言迁移是2020年代切实可行的奇迹。

**类型：** 学习
**语言：** Python
**前置知识：** 第五阶段·04（GloVe、FastText、子词），第五阶段·11（机器翻译）
**时长：** ~45分钟

## 问题

英语拥有数十亿标注样本。乌尔都语只有几千个。迈蒂利语几乎为零。任何服务于全球受众的实际NLP系统都必须能够在那些没有任务特定训练数据的语言长尾上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型能够将从高资源语言学到的技能迁移到低资源语言。在英语情感分析上微调模型，模型就能在乌尔都语上直接产生出人意料好的情感预测。这就是零样本跨语言迁移，它已经重塑了NLP向世界交付的方式。

本节课将指出其中的权衡、经典模型，以及一个经常让刚接触多语言工作的团队栽跟头的决策：选择用于迁移的源语言。

## 概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用一个基于所有目标语言文本训练得到的SentencePiece或WordPiece分词器。词汇表是共享的：相同的子词单元在相关语言中表示相同的语素。`anti-` 在英语和意大利语中得到相同的token。

**共享表示。** 一个经过多语言掩码语言建模预训练的Transformer会学习到，不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R和NLLB都表现出这一点。"cat"在英语中的嵌入与"chat"（法语）和"gato"（西班牙语）的嵌入聚在一起，完整句子的嵌入也是如此。

**零样本迁移。** 使用一种语言（通常是英语）的标注数据微调模型。推理时，可以在模型支持的任何其他语言上运行。不需要目标语言的标签。对于类型学上相近的语言结果很强，对于差异大的语言则较弱。

**少样本微调。** 添加目标语言的100-500个标注样本。在分类任务上，准确率会跃升至英语基线的95-98%。这是多语言NLP中性价比最高的杠杆。

## 模型

| 模型 | 年份 | 覆盖语言 | 备注 |
|-------|------|----------|------|
| mBERT | 2018 | 104种语言 | 基于Wikipedia训练。第一个实用的多语言LM。低资源语言表现较弱。 |
| XLM-R | 2019 | 100种语言 | 基于CommonCrawl训练（比Wikipedia大得多）。奠定了跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100种语言 | XLM-R的变体，词汇量100万（vs 25万）。低资源语言表现更好。 |
| mT5 | 2020 | 101种语言 | 用于多语言生成的T5架构。 |
| NLLB-200 | 2022 | 200种语言 | Meta的翻译模型；包含55种低资源语言。 |
| BLOOM | 2022 | 46种语言 + 13种编程语言 | 开源176B多语言LLM。 |
| Aya-23 | 2024 | 23种语言 | Cohere的多语言LLM。在阿拉伯语、印地语、斯瓦希里语上表现强劲。 |

根据用例选择。分类任务可以合理默认选择XLM-R-base。生成任务则需要mT5或NLLB，取决于翻译还是开放生成。LLM风格的工作搭配Aya-23或使用显式多语言提示的Claude。

## 源语言决策（2026年研究）

大多数团队默认使用英语作为微调源。最近的研究（2026年）表明这通常是错误的。

语言相似性比原始语料大小更能预测迁移质量。对于斯拉夫语目标，德语或俄语通常优于英语。对于印度语系目标，印地语通常优于英语。**qWALS**相似性度量（2026年，基于世界语言结构图集特征）量化了这一点。**LANGRANK**（Lin等，ACL 2019）是一种独立的、更早的方法，它根据语言相似性、语料大小和遗传关系对候选源语言进行排序。

实用规则：如果你的目标语言有一个在类型学上接近的高资源近亲，先尝试在那个语言上微调，然后与英语微调的结果进行比较。

## 动手实现

### 第一步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同样的API。基于NLI数据训练的XLM-R通过蕴含技巧很好地迁移到分类任务。

### 第二步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译的句子在嵌入空间中彼此靠近。不同的英语句子则更远。这正是跨语言检索、聚类和相似性计算的基础。

### 第三步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于100-500个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。较高的学习率会导致多语言对齐崩溃，模型退化为仅限英语的模型。

## 真正有效的评估

- **每种语言在保留集上的准确率。** 不要聚合。聚合会隐藏长尾表现。
- **与单语言基线对比。** 对于那些数据充足的语言，从头训练的单语言模型有时会击败多语言模型。请测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型对远离拉丁字母的文字系统的分词往往较弱。
- **跨语言一致性。** 两种语言中相同含义应产生相同预测。衡量差距。

## 使用它

2026年的推荐栈：

| 任务 | 推荐 |
|-----|------|
| 分类，100种语言 | 微调后的XLM-R-base（~270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200种语言 | `facebook/nllb-200-distilled-600M`（见第11课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言NLP | XLM-V 或基于相关高资源语言的领域特定微调 |

如果性能重要，始终预留预算在目标语言上进行微调。零样本是起点，不是最终答案。

### 分词代价（低资源语言出问题的地方）

多语言模型在所有语言之间共享一个分词器。该词汇表基于一个由英语、法语、西班牙语、汉语、德语主导的语料库训练而成。对于主流集之外的任何语言，三种代价会悄然叠加：

- **词元膨胀代价。** 低资源语言的文本分词后每个词产生的token数远多于英语。一个印地语句子所需的token数可能是等效英语句子的3-5倍。这3-5倍会吞噬你的上下文窗口、训练效率和延迟。
- **变体恢复代价。** 每个拼写错误、变音符号变体、Unicode归一化不匹配或大小写变化都会在嵌入空间中变成一个冷启动的不相关序列。模型无法学会母语者视为理所当然的正字法对应关系。
- **容量溢出代价。** 代价1和代价2消耗了上下文位置、层深度和嵌入维度。留给实际推理的容量系统地少于高资源语言从同一模型中获得的容量。

实际症状：你的模型在印地语上正常训练，损失曲线看着正常，评估困惑度看似合理，但生产输出有细微错误。形态在句子中途崩溃。罕见屈折形式始终无法恢复。**你无法通过增加数据规模来修复一个损坏的分词器。**

缓解措施：选择一个对目标语言有良好覆盖的分词器（XLM-V的100万token词汇表是一种直接修复）；在训练前对保留的目标文本验证词元膨胀率；对于真正的长尾文字，使用字节级回退（SentencePiece `byte_fallback=True`，GPT-2风格的字节级BPE），确保没有任何词被标为OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 对英语、法语、印地语和阿拉伯语各10个句子运行零样本分类流水线。报告每种语言的准确率。你应该会看到法语很强，印地语尚可，阿拉伯语有波动。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料库上构建一个跨语言检索器。用英语查询，检索任意语言的文档。测量recall@5。
3. **困难。** 比较英语源和印地语源微调在印地语分类任务上的表现。两种方案下使用500个目标语言样本进行少样本微调。报告哪种源在印地语上准确率更高以及高出多少。这是LANGRANK论题的缩影。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词汇和参数。 |
| 跨语言迁移 | 用一种语言训练，用另一种语言运行 | 在源语言上微调，在目标语言上评估，无需目标语言标签。 |
| 零样本 | 没有目标语言标签 | 无需在目标语言上微调即可迁移。 |
| 少样本 | 少量目标语言标签 | 使用100-500个目标语言样本进行微调。 |
| mBERT | 第一个多语言语言模型 | 基于Wikipedia预训练的104语言BERT。 |
| XLM-R | 标准跨语言基线 | 基于CommonCrawl预训练的100语言RoBERTa。 |
| NLLB | Meta的200语言机器翻译 | No Language Left Behind。包含55种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启了跨语言迁移研究路线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere的多语言LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — 关于qWALS/LANGRANK源语言的论文。
