# T5, BART — 编码器-解码器模型

> 编码器理解，解码器生成。将它们重新组合，你就得到了一个专为输入→输出任务构建的模型：翻译、摘要、改写、转录。

**类型：** 学习  
**语言：** Python  
**先修要求：** 阶段7·05（完整Transformer）、阶段7·06（BERT）、阶段7·07（GPT）  
**预计时间：** ~45分钟

## 问题

仅解码器的GPT和仅编码器的BERT各自为了不同目标，对2017年的架构进行了精简。但许多任务本质上是输入-输出式的：

- 翻译：英语→法语。
- 摘要：5000个token的文章→200个token的摘要。
- 语音识别：音频token→文本token。
- 结构化抽取：散文→JSON。

对于这些任务，编码器-解码器架构最为契合。编码器生成源输入的一个稠密表示，解码器则在每步生成输出时，通过交叉注意力关注该表示。训练时在输出端进行移位一位的操作。损失函数与GPT相同，只是额外以编码器输出为条件。

有两篇论文定义了现代范式：

1. **T5**（Raffel等人，2019年）。"Text-to-Text Transfer Transformer"。将所有NLP任务重新定义为文本输入、文本输出。单一架构、单一词表、单一损失函数。预训练采用掩码跨度预测（在输入中破坏跨度，在输出中解码它们）。
2. **BART**（Lewis等人，2019年）。"Bidirectional and Auto-Regressive Transformer"。去噪自编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原始输入。

到2026年，编码器-解码器形式仍在输入结构重要的场景中得以保留：

- Whisper（语音→文本）。
- Google的翻译栈。
- 某些具有独特上下文与编辑结构的代码补全/修复模型。
- Flan-T5及其变体，用于结构化推理任务。

虽然仅解码器模型占据了聚光灯，但编码器-解码器从未消失。

## 概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键点是，编码器对每个输入只运行一次。解码器以自回归方式运行，但在每一步都会交叉注意力到**相同的**编码器输出。缓存编码器输出对于长输入是一个免费的加速手段。

### T5预训练——跨度破坏

随机选取输入中的若干跨度（平均长度3个token，总占比15%）。每个跨度替换为一个唯一的哨兵token：`<extra_id_0>`、`<extra_id_1>`等。解码器仅输出被破坏的跨度及其哨兵前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

与预测整个序列相比，信号更廉价。在T5论文的消融实验中，它与MLM（BERT）和前缀LM（UniLM）相比具有竞争力。

### BART预训练——多噪声去噪

BART尝试了五种噪声函数：

1. Token掩码。
2. Token删除。
3. 文本填充（掩码一个跨度，解码器插入正确的长度）。
4. 句子置换。
5. 文档旋转。

将文本填充与句子置换结合，在下游任务中取得了最佳结果。解码器始终重建原始文本。BART输出的是完整序列，而不仅仅是破坏的跨度——因此预训练计算量比T5更大。

### 推理

与GPT相同的自回归生成方式。可使用贪心/束搜索/top-p采样。束搜索（宽度4–5）是翻译和摘要的标准做法，因为输出分布比对话场景更窄。

### 2026年何时选择各变体

| 任务 | 编码器-解码器？ | 原因 |
|------|------------------|-----|
| 翻译 | 是，通常 | 清晰的源序列；固定的输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器塑造音频特征 |
| 对话/推理 | 否，仅解码器 | 没有持久化的“输入”——对话本身就是一个序列 |
| 代码补全 | 通常否 | 长上下文的仅解码器模型胜出；像Qwen 2.5 Coder这样的代码模型是仅解码器架构 |
| 摘要 | 两者均可 | BART、PEGASUS超越了早期的仅解码器基线；现代的仅解码器大语言模型也能匹敌 |
| 结构化抽取 | 两者均可 | T5很干净，因为“文本→文本”可以吸收任何输出格式 |

自2022年左右的趋势：仅解码器模型正在接管原来由编码器-解码器承担的任务，原因是(a)经过指令微调的仅解码器大语言模型可以通过提示泛化到任何任务，(b)一种架构比两种更容易扩展，(c)RLHF假设使用解码器架构。编码器-解码器在输入模态不同（语音、图像）或束搜索质量很重要的情况下仍然保持优势。

## 动手构建

参见 `code/main.py`。我们为一个玩具语料库实现T5风格的跨度破坏——这是本课最有用的一个片段，因为它出现在此后每一个编码器-解码器预训练方案中。

### 第一步：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循T5惯例：`<sent0> span0 <sent1> span1 ...`。破坏后的输入将未改变的token与跨度位置上的哨兵token交替排列。

### 第二步：验证往返

给定破坏后的输入和目标，重建原始句子。如果你的破坏是可逆的，那么前向过程就是良定义的。这是一个合理性检查——真实训练中永远不会这样做，但测试成本很低，并且能捕获跨度记账中的差一错误。

### 第三步：BART噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。将其中两个组合，展示结果。

## 使用

HuggingFace参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5的技巧：任务名称被放入输入文本中。同一个模型可以处理数十个任务，因为每个任务都是文本输入、文本输出。到2026年，这一模式已被指令微调的仅解码器模型泛化，但T5首先将其规范化。

## 交付

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为一项新任务选择编码器-解码器或仅解码器。

## 练习

1. **简单。** 运行 `code/main.py`，对一条30个token的句子应用跨度破坏，验证将没有哨兵的源token与解码后的目标跨度拼接起来能还原原始句子。
2. **中等。** 实现BART的 `text_infill` 噪声：将随机跨度替换为单个 `<mask>` token，解码器必须推断出正确的跨度长度和内容。展示一个示例。
3. **困难。** 在小型英语→猪拉丁语语料库（200对）上微调 `flan-t5-small`。在一个50对的保留集上测量BLEU值。与在相同数据和相同计算量下微调 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 | "序列到序列Transformer" | 两个栈：用于输入的双向编码器，以及带有交叉注意力的因果解码器用于输出。 |
| 交叉注意力 | "源与目标对话的位置" | 解码器的Q × 编码器的K/V。编码器信息进入解码器的唯一位置。 |
| 跨度破坏 | "T5的预训练技巧" | 将随机跨度替换为哨兵token；解码器输出这些跨度。 |
| 去噪目标 | "BART的游戏" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵token | "`<extra_id_N>` 占位符" | 特殊token，在源中标记被破坏的跨度，在目标中重新标记它们。 |
| Flan | "指令微调的T5" | 在超过1800个任务上微调的T5；使编码器-解码器在指令跟随方面具有竞争力。 |
| 束搜索 | "解码策略" | 在每一步保留top-k个部分序列；翻译/摘要的标准做法。 |
| 教师强制 | "训练时的输入" | 在训练期间，将真实的先前输出token（而非采样得到的）馈送给解码器。 |

## 进一步阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026年编码器-解码器的典范。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
