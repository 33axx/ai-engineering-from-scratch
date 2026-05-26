# 序列到序列模型

> 两个 RNN 假装翻译。它们遇到的瓶颈正是注意力存在的理由。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第5阶段·08（CNN + RNN 处理文本），第3阶段·11（PyTorch 入门）  
**时间：** 约75分钟

## 问题

分类任务将可变长度的序列映射到单个标签。翻译任务则将可变长度的序列映射到另一个可变长度的序列。输入和输出分属不同的词汇表，可能来自不同语言，且长度没有必然的对应关系。

Seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一种刻意简化的方法解决了这个问题。两个 RNN。一个读取源句子并生成固定大小的上下文向量。另一个读取该向量并逐个 token 生成目标句子。与你为课程08编写的代码相同，只是拼接方式不同。

研究这个架构有两个理由。首先，上下文向量的瓶颈是 NLP 中最具教学价值的失败案例，它推动了注意力和 Transformer 的所有优点。其次，其训练方式（教师强制、计划采样、推理时的束搜索）至今仍适用于所有现代生成系统，包括大语言模型。

## 概念

**编码器。** 一个读取源句子的 RNN。它的最终隐藏状态就是 **上下文向量** —— 整个输入的固定大小摘要。按理说，除了源信息外什么都没丢。

**解码器。** 另一个 RNN，用上下文向量初始化。每一步，它接收之前生成的 token 作为输入，并输出一个在目标词汇表上的概率分布。通过采样或 argmax 选择下一个 token，再将这个 token 反馈回去。重复直到生成 `<EOS>` token 或达到最大长度。

**训练：** 每个解码步的交叉熵损失，沿序列求和。通过时间的标准反向传播对两个网络进行更新。

**教师强制。** 训练时，解码器在时间步 `t` 的输入是位置 `t-1` 的 *真实* token，而不是解码器自己之前的预测。这稳定了训练；没有它，早期错误会级联放大，模型永远无法学习。推理时，你只能使用模型自己的预测，因此训练和推理之间始终存在分布差异。这种差异称为 **暴露偏差**。

**瓶颈。** 编码器从源句子学到的所有信息都必须被压缩到那一个上下文向量中。长句子会丢失细节；稀有词汇会被模糊化；语序调整（如 chat noir 对比 black cat）必须被记忆，而非计算出来。

注意力（课程10）通过让解码器查看 *所有* 编码器隐藏状态（而不仅是最后一个）来解决这个问题。这就是全部要点。

## 构建它

### 第一步：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]` —— 每个输入位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]` —— 最后一个时间步。课程08说“对输出做池化用于分类”。这里我们将最后的隐藏状态作为上下文向量，并忽略每个时间步的输出。

### 第二步：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次被调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 第三步：使用教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得命名的旋钮。`ignore_index=0` 跳过填充 token 的损失。`teacher_forcing_ratio` 是每一步使用真实 token 还是模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中逐步退火到约 0.5，以缩小暴露偏差的差距。

### 第四步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每一步选择概率最高的 token。它可能会偏离：一旦你提交了一个 token，就无法撤回。**束搜索** 则保持顶部 `k` 个部分序列存活，结束时选择得分最高的完整序列。束宽度 3-5 是标准配置。

### 第五步：瓶颈演示

在一个玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记忆一个 40 token 的输入。信息在每一个编码器步骤中都存在，但解码器只看到最后一个状态。注意力直接解决了这个问题。

## 使用它

PyTorch 提供了基于 `nn.Transformer` 和 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库则提供了在数十亿 token 上训练完成的完整编码器-解码器模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器已经用 Transformer 取代了 RNN。其高层形状（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个模块内部的机制不同。

### 何时仍应使用 RNN 的 seq2seq

对于新项目几乎不用。特殊情况包括：

- 流式翻译：用一个 token 一个 token 的方式消耗输入，内存有限。
- 设备端文本生成：Transformer 的内存成本过高。
- 教学目的：理解编码器-解码器瓶颈是通往理解 Transformer 为何获胜的最快路径。

### 暴露偏差及其缓解方法

- **计划采样。** 在训练过程中退火教师强制比率，让模型学会从自身错误中恢复。
- **最小风险训练。** 使用句子级别的 BLEU 分数而不是 token 级别的交叉熵进行训练，更贴近实际需求。
- **强化学习微调。** 用某个指标奖励序列生成器。现代 LLM 的 RLHF 中用到。

这三种方法仍适用于基于 Transformer 的生成。

## 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在输入输出对（目标等于源）上训练一个 GRU seq2seq。测量长度 5、10、20 时的准确率，复现瓶颈。
2. **中等。** 添加束宽度为 3 的束搜索解码。在一个小型平行语料库上对比贪心解码的 BLEU 值。记录束搜索胜出的地方（通常是在最后几个 token）以及无差别的地方。
3. **困难。** 在 10k 对释义数据集上微调 `facebook/bart-base`。比较微调后模型（束宽度 4）与基础模型在留出输入上的输出。报告 BLEU，并挑出 10 个定性示例。

## 关键术语

| 术语 | 人们说的是什么 | 实际含义 |
|------|----------------|---------|
| 编码器 | 输入 RNN | 读取源句子。产生每个时间步的隐藏状态和一个最终的上下文向量。 |
| 解码器 | 输出 RNN | 用上下文向量初始化。逐个 token 生成目标序列。 |
| 上下文向量 | 摘要 | 编码器最终隐藏状态。固定大小。注意力所解决的瓶颈。 |
| 教师强制 | 使用真实 token | 训练时喂入真实的上一 token。稳定学习过程。 |
| 暴露偏差 | 训练/推理差距 | 模型在真实 token 上训练，从未练习过从自身错误中恢复。 |
| 束搜索 | 更好的解码 | 每一步保持顶部 k 个部分序列存活，而不是贪心地提交一个。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。紧接着本课程阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可构建的 seq2seq + 注意力代码。
