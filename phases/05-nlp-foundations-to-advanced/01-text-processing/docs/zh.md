# 文本处理——分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建  
**语言：** Python  
**前提条件：** 阶段2 · 14（朴素贝叶斯）  
**时间：** 约45分钟

## 问题

模型无法读取“The cats were running.”它只读取整数。

每个NLP系统都以相同的三个问题开始。单词从哪里开始。单词的词根是什么。我们如何将 'run'、'running'、'ran' 在需要时视为同一事物，而在不需要时视为不同事物。

分词搞错了，模型就会从垃圾数据中学习。如果你的分词器将 `don't` 视为一个词元，而将 `do n't` 视为两个，训练分布就会分裂。如果你的词干提取器将 `organization` 和 `organ` 合并为同一个词干，主题建模就毁了。如果你的词形还原器需要词性上下文但你却没有传递，动词就会被当作名词处理。

本课从头构建三个预处理原语，然后展示NLTK和spaCy如何完成相同的工作，以便你了解其中的权衡。

## 概念

三种操作。每种都有其职责和失败模式。

**分词**将字符串拆分为词元。“词元”故意含糊其辞，因为合适的粒度取决于任务。经典NLP使用词级，Transformer使用子词，无空格语言使用字符。

**词干提取**通过规则切除后缀。快速、激进、愚笨。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原**利用语法知识将单词简化为词典形式。较慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道“ran”是“run”的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度重要且你能容忍噪声时（搜索索引、粗略分类），使用词干提取。当意义重要时（问答、语义搜索、任何用户会阅读的内容），使用词形还原。

## 动手构建

### 步骤1：正则表达式单词分词器

最简单实用的分词器在非字母数字字符上分割，同时将标点保留为独立的词元。不完美，不是最终版本，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级顺序。包含可选内部撇号的单词（`don't`，`it's`）。纯数字。任何单个非空白非字母数字字符作为独立词元（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 被分割为 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对于大多数任务来说足够好。URL、电子邮件、话题标签都会出错。对于生产环境，在通用模式之前添加特定模式。

### 步骤2：Porter词干提取器（仅步骤1a）

完整的Porter算法有五个规则阶段。仅步骤1a就覆盖了最常见的英语后缀，并展示了模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下阅读规则。`ies -> i` 规则解释了为什么 `ponies -> poni`，而不是 `pony`。真正的Porter有步骤1b可以修复它。规则相互竞争。前面的规则胜出。顺序比任何单个规则都更重要。

### 步骤3：基于查找的词形还原器

正确的词形还原需要形态学。一个易于教学的版本使用一个小型词元表和后备方案。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键的教学时刻。`watched` 不在我们的表中，而我们的后备方案仅处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、带有音变的复数（`children -> child`）。这就是为什么生产系统使用WordNet、spaCy的形态分析器或完整的形态分析器。

### 步骤4：将它们管线化

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器。阶段5·07（词性标注）将构建一个。现在，将所有内容默认为 `NOUN` 并承认这一限制。

## 使用它们

NLTK和spaCy附带了生产版本。每样只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略词、Unicode、正则表达式遗漏的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将词性标注从NLTK的Penn Treebank方案转换为WordNet的缩写集。上面提到的转换代码是大多数教程跳过的地方。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy将整个流水线隐藏在 `nlp(text)` 之后。分词、词性标注和词形还原全部运行。在大规模运行时比NLTK更快。开箱即用更准确。代价是你不能轻易地替换单个组件。

### 何时选择哪个

| 情况 | 选择 |
|-----------|------|
| 教学、研究、组件替换 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer流水线（反正你会用模型的tokenizer分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告的两个失败模式

大多数教程只教算法就停止了。有两件事会困扰真实的预处理流水线，而且几乎从未被提及。

**可复现性漂移。** NLTK和spaCy在不同版本之间会改变分词和词形还原器的行为。在spaCy 2.x中产生 `['do', "n't"]` 的代码可能在3.x中产生 `["don't"]`。你的模型在一个分布上训练，推理却在另一个分布上运行。准确率悄然下降，无人知晓原因。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，固定20个样本句子的预期分词结果。每次升级都运行它。

**训练/推理不匹配。** 使用激进的预处理（小写化、停用词移除、词干提取）进行训练，部署到原始用户输入上，然后看着性能崩溃。这是生产NLP中最常见的单一失败。如果你在训练期间进行预处理，你必须在推理期间运行相同的函数。将预处理作为模型包内的一个函数交付，而不是作为服务团队重写的笔记本单元格。

## 交付它

一个可复用的提示，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 使URL保持为单个词元。测试：`tokenize("Visit https://example.com today.")` 应生成一个URL词元。
2. **中等。** 实现Porter步骤1b。如果一个单词包含元音且以 `ed` 或 `ing` 结尾，则删除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器，使用WordNet作为查找表，但当WordNet没有条目时回退到你的Porter词干提取器。在标注语料库上测量相对于纯WordNet和纯Porter的准确率。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 词元 | 一个单词 | 模型消费的任何单元。可以是词、子词、字符或字节。 |
| 词干 | 单词的词根 | 基于规则的后缀剥离结果。不总是真实单词。 |
| 词目 | 词典形式 | 你会去查的那种形式。需要语法上下文才能正确计算。 |
| 词性标注 | 词性 | 如NOUN、VERB、ADJ等类别。要准确进行词形还原需要它。 |
| 形态学 | 单词形状规则 | 单词如何根据时态、数、格改变形式。词形还原依赖它。 |

## 进一步阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线的连接方式。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况。
