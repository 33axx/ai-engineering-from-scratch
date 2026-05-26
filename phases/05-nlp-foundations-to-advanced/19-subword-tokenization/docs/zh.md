# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 单词分词器无法处理未见过的单词。字符分词器会使序列长度爆炸。子词分词器则取中庸之道。每个现代LLM都基于它。

**类型:** 学习  
**语言:** Python  
**前提条件:** 阶段5·01（文本处理），阶段5·04（GloVe / FastText / 子词）  
**时间:** ~60分钟

## 问题

你的词汇表有50,000个单词。用户输入了“untokenizable”。你的分词器返回`[UNK]`。模型现在对这个单词没有任何信号。更糟的是：语料库中第90百分位的文档有40个罕见词，这意味着每个文档丢失了40比特的信息。

子词分词解决了这个问题。常见词保持为单个token。罕见词分解成有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖了一切，因为任何字符串归根结底都是字节序列。

2026年的每个前沿LLM都基于三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。你无法在不选择一个的情况下发布语言模型。

## 概念

![BPE对比Unigram对比WordPiece，逐个字符说明](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词汇表开始。统计每个相邻对。将最频繁的对合并成一个新token。重复直到达到目标词汇表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 相同算法但作用于原始字节（256个基础token）而非Unicode字符。保证零`[UNK]` token——任何字节序列都能编码。GPT-2使用50,257个token（256字节 + 50,000次合并 + 1个特殊token）。

**Unigram。** 从一个庞大的词汇表开始。为每个token分配一个unigram概率。迭代剪枝那些移除后对语料库对数似然损失最小的token。推理时具有概率性：可以对分词结果进行采样（通过子词正则化实现数据增强）。用于T5、mBART、ALBERT、XLNet、Gemma。

**WordPiece。** 合并那些能最大化训练语料库似然的配对，而非原始频率。用于BERT、DistilBERT、ELECTRA。

**SentencePiece vs tiktoken。** SentencePiece是一个直接对原始Unicode文本*训练*词汇表（BPE或Unigram）的库，将空格编码为`▁`。tiktoken是OpenAI的快速*编码器*，使用预构建词汇表；它不进行训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无需预分词）或HF Tokenizers。
- **针对GPT词汇表的快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼顾：** HF Tokenizers——一个库，训练+服务。

## 动手实践

### 步骤1：从零实现BPE

参见 `code/main.py`。循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码了三个事实。`</w>`标记词尾，使得“low”（后缀）和“lower”（前缀）保持区分。频率加权使高频配对优先合并。合并列表是有序的——推理时按训练顺序应用合并。

### 步骤2：使用学习到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现O(n·|merges|)。生产环境实现（tiktoken、HF Tokenizers）使用带有优先级队列的合并排名查找，运行时间复杂度接近线性。

### 步骤3：SentencePiece实践

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为`▁`，`character_coverage`控制罕见字符被保留还是映射到`<unk>`的激进程度。

### 步骤4：用于OpenAI兼容词汇表的tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust后端）。与GPT-4/5分词精确匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026年依然存在的陷阱

- **分词器漂移。** 用词汇表A训练，用词汇表B部署。Token ID不同；模型输出乱码。在CI中检查`tokenizer.json`的哈希值。
- **空格歧义。** BPE处理“hello”与“ hello”会产生不同的token。始终显式指定`add_special_tokens`和`add_prefix_space`。
- **多语言训练不足。** 以英语为主的语料库产生的词汇表会将非拉丁文字符拆成5-10倍的token。相同提示在日语/阿拉伯语中成本高出5-10倍（GPT-3.5）。o200k_base部分修复了这个问题。
- **表情符号拆分。** 单个表情符号可能消耗5个token。在预算上下文时需检查表情符号的处理。

## 使用场景

2026年的技术栈：

| 场景 | 选择 |
|------|------|
| 从头训练单语模型 | HF Tokenizers (BPE) |
| 训练多语言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 提供OpenAI兼容API | tiktoken (`o200k_base` for GPT-4+) |
| 领域特定词汇表（代码、数学、蛋白质） | 在领域语料上训练自定义BPE，与基础词汇表合并 |
| 边缘推理，小模型 | Unigram（较小的词汇表效果更好） |

词汇表大小是一个缩放决策，而非定值。粗略经验：32k用于<1B参数，50-100k用于1-10B，200k+用于多语言/前沿模型。

## 保存输出

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在`code/main.py`的小型语料库上训练一个500次合并的BPE。对三个保留词进行编码。有多少个产生了恰好1个token vs 超过1个token？
2. **中等。** 比较100个英文维基百科句子在`cl100k_base`、`o200k_base`以及你训练的词汇表大小为32k的SentencePiece BPE下的token数量。报告每个的压缩比。
3. **困难。** 用BPE、Unigram和WordPiece训练相同的语料库。在使用每个分词器进行小型情感分类器时，测量下游准确率。分词器的选择是否会使F1分数产生超过1个点的差异？

## 关键术语

| 术语 | 通常含义 | 实际含义 |
|------|---------|---------|
| BPE | 字节对编码 | 贪心地合并最频繁的字符对，直到达到目标词汇表大小。 |
| 字节级BPE | 永远不会出现未知token | 基于原始256字节的BPE；GPT-2 / Llama使用此方法。 |
| Unigram | 概率分词器 | 使用对数似然从大型候选集中剪枝；T5、Gemma使用。 |
| SentencePiece | 处理空格的那个 | 直接对原始文本训练BPE/Unigram的库；空格编码为`▁`。 |
| tiktoken | 快速的那个 | OpenAI基于Rust的BPE编码器，用于预构建词汇表。不支持训练。 |
| 合并列表 | 魔法数字 | `(a, b) → ab`合并的有序列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见算罕见？ | 分词器必须覆盖的训练语料中字符的比例；通常~0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) ——BPE论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) ——Unigram论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) ——该库的论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) ——简洁参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) ——使用手册+编码列表。
