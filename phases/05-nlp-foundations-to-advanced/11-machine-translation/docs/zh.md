# 机器翻译

> 翻译是自然语言处理研究三十年来的经济支柱，至今仍在持续创造价值。

**类型：** 实践
**语言：** Python
**前置知识：** 第五阶段·第10课（注意力机制），第五阶段·第04课（GloVe、FastText、子词）
**时长：** 约75分钟

## 问题

模型阅读一种语言的句子，生成另一种语言的句子。长度变化，词序变化。某些源语言单词映射到多个目标语言单词，反之亦然。习语拒绝一对一映射。法语的"我想你"是"tu me manques"——字面意思是"你缺少于我"。没有任何词级对齐能保留这种表达。

机器翻译是一项迫使自然语言处理领域发明编码器-解码器、注意力机制、Transformer，并最终催生整个大语言模型范式的任务。每一步进步之所以出现，是因为翻译质量是可量化的，而人机之间的差距是顽固的。

本课跳过历史介绍，直接教授2026年的工作流程：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 和 chrF 评估，以及至今仍会未被捕获就投入生产的少数失败模式。

## 概念

![机器翻译流程：分词 → 编码 → 带注意力的解码 → 去分词](../assets/mt-pipeline.svg)

现代机器翻译是一个在平行文本上训练的 Transformer 编码器-解码器。编码器以源语言的 token 化形式读取源文本。解码器通过交叉注意力（第10课）利用编码器的输出，一次生成一个子词。解码使用束搜索来避免贪心解码陷阱。输出被去分词、去真大小写化，并与参考翻译进行评分。

三个操作选择决定了真实世界机器翻译的质量。

- **分词器。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言的共享词汇正是 NLLB 实现零样本翻译对的关键。
- **模型大小。** NLLB-200-distilled-600M 可以在笔记本电脑上运行。NLLB-200 3.3B 是已发布的生产默认选项。54.5B 是研究天花板。
- **解码。** 通用内容使用束宽4-5。长度惩罚以避免输出过短。当需要术语一致性时使用约束解码。

## 动手构建

### 第1步：调用预训练机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点重要。`src_lang` 告诉分词器使用哪种文字和切分方式。`forced_bos_token_id` 告诉解码器生成哪种语言。这两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自不同的约定，且不能互换。

### 第2步：BLEU 和 chrF

BLEU 衡量输出与参考翻译之间的 n-gram 重叠。使用四种参考 n-gram 大小（1-4），精确率的几何平均值，对过短输出进行长度惩罚。分数范围为 [0, 100]。常用，但解释起来令人沮丧：30 BLEU 为"可用"；40 为"良好"；50 为"优秀"；低于 1 BLEU 的差异是噪声。

chrF 衡量字符级别的 F 值。对形态丰富的语言更敏感，因为这些语言中 BLEU 会低估匹配数。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它会规范 token 化过程，使分数在不同论文之间具有可比性。自己计算 BLEU 是导致误导性基准测试的原因。

### 三级评估层级（2026年）

现代机器翻译评估使用三种互补的度量族。至少使用两种再发布。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释、对改写不敏感。用于遗留对比和回归检测。
- **学习式**（COMET、BLEURT、BERTScore）。基于人类判断训练的神经模型；比较翻译与源语言和参考翻译的语义相似度。自2023年以来，COMET 与机器翻译研究的关联度最高，并且在2026年，在质量重要的场景中是生产默认选项。
- **LLM 作为评判**（无参考）。提示大型模型根据流畅性、充分性、语气、文化适当性对翻译进行评分。如果评分标准设计良好，GPT-4 作为评判与人类一致性达到约80%。用于不存在参考翻译的开放式内容。

2026年的实用组合：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，以及一个提示的 LLM 用于最终面向人类的信号。在信任生产数据之前，先用50-100个人工标注的示例校准每个度量。

无参考度量（COMET-QE、BLEURT-QE、LLM-as-judge）可以在没有参考翻译的情况下评估翻译，这对于长尾语言对（没有参考翻译可用）非常重要。

### 第3步：生产中容易出错的地方

上述工作流程将流畅地翻译80%的内容，而剩余20%会悄然失败。命名的失败模式：

- **幻觉。** 模型编造源文本中没有的内容。常见于不熟悉领域的词汇。症状：输出流畅但声称了源文本未提及的事实。缓解措施：对领域术语进行约束解码，对受监管内容进行人工审查，监控输出长度远大于输入的情况。
- **目标语言错误。** 模型翻译成了错误的语言。NLLB 在罕见语言对上出奇地容易出现这个问题。缓解措施：验证 `forced_bos_token_id`，并始终在解码后使用语言识别模型检查输出。
- **术语漂移。** "Sign up"在文档1中变成"s'inscrire"，在文档2中变成"créer un compte"。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解措施：使用词汇表约束解码或后期编辑词典。
- **敬语不匹配。** 法语的"tu" vs "vous"，日语的礼貌级别。模型会选择训练数据中更常见的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，使用带有敬语标记的提示前缀，或者仅在正式语料上微调一个小模型。
- **短输入导致的长度爆炸。** 非常短的输入句子常常产生过长的翻译，因为长度惩罚在源语言 token 少于约5个时会急剧下降。缓解措施：设置与源语言长度成比例的最大长度硬限制。

### 第4步：领域微调

预训练模型是通才。法律、医学或游戏对话翻译可以通过在领域平行数据上微调获得可测量提升。方法并不新奇：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量的平行示例胜过几十万个噪点网络爬取示例。训练数据的质量是生产中最大的杠杆。

## 使用

2026年机器翻译的生产选型：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言到任意语言，200种语言 | `facebook/nllb-200-distilled-600M`（笔记本电脑）或 `nllb-200-3.3B`（生产环境） |
| 以英语为中心，高质量，50种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行，低成本推理，英-法/德/西 | Helsinki-NLP / Marian 系列模型 |
| 延迟敏感的浏览器端 | ONNX 量化版 Marian（约50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示 |

截至2026年，LLM 在多个语言对上已经超越了专门的机器翻译模型，特别是在习语内容和长上下文方面。代价是每个 token 的成本和延迟。当上下文长度、风格一致性或通过提示进行领域适应比吞吐量更重要时，选择 LLM。

## 交付

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一段5句英文段落翻译成法语，再翻译回英语。测量往返翻译与原始文本的接近程度。你会看到语义保留但词汇选择有漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现翻译输出的语言识别检查。将其集成到机器翻译调用中，以便在返回之前捕获目标语言错误的生成。
3. **困难。** 在你选择的5000对领域语料上微调 `nllb-200-distilled-600M`。在微调前后测量保留集上的 BLEU。报告哪些类型的句子得到了改进，哪些出现了退化。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带长度惩罚的 n-gram 精确率。范围 [0, 100]。 |
| chrF | 字符 F 值 | 字符级别的 F 值。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 Transformer 编码器-解码器。2017年后的默认方案。 |
| NLLB | 不让任何一种语言落后 | Meta 的200种语言机器翻译模型系列。 |
| 约束解码 | 受控输出 | 强制特定 token 或 n-gram 出现在/不出现在输出中。 |
| 幻觉 | 编造内容 | 模型输出在源文本中没有依据。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调指南。
