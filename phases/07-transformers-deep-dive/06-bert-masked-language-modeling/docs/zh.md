# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。相差一句话 —— 以及后面五年所有与嵌入形状相关的事物。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 7 · 05（完整 Transformer），阶段 5 · 02（文本表示）  
**时间：** 约 45 分钟

## 问题所在

2018 年，每项 NLP 任务——情感分析、命名实体识别、问答、蕴含——都在各自标注数据上从头训练自己的模型。没有一个预训练“理解英语”的检查点供你微调。ELMo（2018 年）展示了你可以用双向 LSTM 预训练上下文嵌入；它有所帮助，但未能泛化。

BERT（Devlin 等人，2018 年）提出了一个问题：如果我们拿一个 Transformer 编码器，在互联网上的每一句话上训练它，强制它从两侧上下文中预测缺失的词，结果会怎样？然后你只需在下游任务上微调一个头。参数效率是一场启示。

结果：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时所有 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核流水线和语义搜索系统内部都有一个 BERT。

到了 2026 年，仅编码器模型仍然适用于分类、检索和结构化抽取——它们每个 token 的运行速度比解码器快 5–10 倍，其嵌入是所有现代检索栈的骨干。ModernBERT（2024 年 12 月）将架构推至 8K 上下文，并采用了 Flash Attention + RoPE + GeGLU。

## 概念

![掩码语言建模：选取 token，将其掩码，预测原始 token](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型去预测掩码位置上的原始 token。由于编码器是双向的，在位置 1 预测 `[MASK]` 可以利用位置 2 及之后的 `brown fox jumps`。这正是 GPT 做不到的事情。

### BERT 掩码规则

在选定用于预测的 15% 的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不全用 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。如果训练模型在 100% 的掩码位置都期望 `[MASK]`，会造成预训练和微调之间的分布偏移。10% 随机 + 10% 保持不变可以让模型保持诚实。

### 下一句预测（NSP）——以及为何被抛弃

原始的 BERT 还训练了 NSP：给定两个句子 A 和 B，预测 B 是否接在 A 之后。RoBERTa（2019 年）进行了消融实验，证明 NSP 有害无益。现代编码器都跳过了它。

### 2026 年发生了什么变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了模块：

| 组件 | 原始 BERT（2018 年） | ModernBERT（2024 年） |
|-----------|----------------------|-------------------|
| 位置编码 | 可学习的绝对位置 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | 预归一化 RMSNorm |
| 注意力 | 全密集注意力 | 交替局部（128）+ 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

而且与 2018 年的栈不同，它原生支持 Flash Attention。在序列长度 8K 下，推理速度比 DeBERTa-v3 快 2–3 倍，且 GLUE 分数更高。

### 2026 年仍会选择编码器的使用场景

| 任务 | 为什么编码器优于解码器 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个 token 的嵌入质量更高 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 按位置输出，天然双向 |
| 零样本蕴含（NLI） | 编码器之上的分类器头 |
| RAG 的重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

## 动手构建

### 第一步：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词汇表大小和掩码概率。返回输入 ID（经过掩码处理）和标签（仅在掩码位置有效，其余位置为 -100 —— 这是 PyTorch 忽略索引的约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第二步：在微型语料库上运行 MLM 预测

在包含 20 个单词、200 句话的词汇表上训练一个 2 层编码器 + MLM 头。无需梯度——仅做前向合理性检查。完整训练需要 PyTorch。

### 第三步：比较掩码类型

展示三重规则如何让模型在没有 `[MASK]` 的情况下仍然可用。在未掩码的句子和已掩码的句子上分别预测。两者都应产生合理的 token 分布，因为模型在训练中看到了两种模式。

### 第四步：微调头部

在一个玩具情感数据集上，将 MLM 头替换为分类头。只有头部训练；编码器被冻结。这是每个 BERT 应用遵循的模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调过的 BERT。** `sentence-transformers` 中的模型（如 `all-MiniLM-L6-v2`）是经过对比损失训练的 BERT。编码器是相同的，只是损失函数变了。

**交叉编码器重排序器也是微调过的 BERT。** 对 `[CLS] query [SEP] doc [SEP]` 进行配对分类。查询和文档之间的双向注意力正是交叉编码器在质量上优于双编码器的原因。

**2026 年何时不应该选择 BERT。** 任何生成式任务。编码器没有合理的方式来自回归地产生 token。此外：任何参数低于 10 亿的模型，如果一个小型解码器能在更灵活的同时匹配质量（如 Phi-3-Mini、Qwen2-1.5B），那么解码器更优。

## 交付

参见 `outputs/skill-bert-finetuner.md`。这份技能文档为一个新的分类或抽取任务规划了 BERT 微调（骨干网络选择、头部规格、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 上的掩码分布。确认大约 15% 被选中，其中大约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被切分成子词，则一起掩码所有子词，要么都不掩码。在一个 500 句话的语料库上衡量这是否提高了 MLM 准确率。
3. **困难。** 在一个公共数据集上的 10,000 句话上训练一个微型（2 层，d=64）BERT。针对 SST-2 情感分析微调 `[CLS]` token。在相同参数数量下与仅解码器基线进行比较——哪个胜出？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | “掩码语言建模” | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始 token。 |
| 双向 | “两边都看” | 编码器注意力没有因果掩码——每个位置都能看到所有其他位置。 |
| `[CLS]` | “汇聚 token” | 预置于每个序列前的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | “段分隔符” | 分隔成对的序列（例如查询/文档，句子 A/B）。 |
| NSP | “下一句预测” | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被丢弃。 |
| 微调 | “适应任务” | 保持编码器大部分冻结；在顶部为下游任务训练一个小型头部。 |
| 交叉编码器 | “重排序器” | 一个将查询和文档都作为输入并输出相关性分数的 BERT。 |
| ModernBERT | “2024 年刷新” | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) —— 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) —— 如何正确训练 BERT；淘汰了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) —— 替换 token 检测在相同计算量下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) —— ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) —— 经典编码器参考实现。
