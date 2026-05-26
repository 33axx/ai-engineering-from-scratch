# 用于文本的 CNN 和 RNN

> 卷积学习 n-gram。循环网络记住信息。两者都被注意力机制取代。但在受限硬件上，它们仍然重要。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 3 · 11（PyTorch 入门），阶段 5 · 03（词嵌入），阶段 4 · 02（从头实现卷积）
**时间：** ~75 分钟

## 问题

TF-IDF 和 Word2Vec 产生忽略词序的扁平向量。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时携带信号。

在 Transformer 出现之前，有两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为 3 的滤波器是一个可学习的 trigram 检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2, 3, 4, 5）以检测多尺度模式。最大池化到固定大小的表示。扁平、并行、快速。

**循环网络（RNN, LSTM, GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。顺序、有记忆、输入长度灵活。从 2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本课将构建两者，然后指出引发注意力机制的那个缺陷。

## 概念

**TextCNN**（Kim, 2014）。Token 被嵌入。一个宽度为 `k` 的一维卷积在连续的 `k`-gram 嵌入上滑动一个滤波器，生成特征图。对该图进行全局最大池化，选出最强激活。将多个滤波器宽度的最大池化输出拼接起来。馈送到分类头部。

为什么有效。滤波器是一个可学习的 n-gram。最大池化是位置不变的，因此 "not good" 在评论的开头或中间都会触发相同的特征。三个滤波器宽度各 100 个滤波器，就得到 300 个可学习的 n-gram 检测器。训练是并行的，没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。共享 `W`, `U`, `b` 跨时间步。时间 `T` 的隐藏状态是整个前缀的总结。对于分类，对 `h_1 ... h_T` 进行池化（最大、平均或最后一个）。

普通 RNN 面临梯度消失问题。**LSTM** 添加了门控，决定遗忘什么、存储什么、输出什么，在长序列中稳定梯度。**GRU** 将 LSTM 简化为两个门；参数更少，性能类似。

**双向 RNN** 运行一个正向 RNN 和一个反向 RNN，拼接隐藏状态。每个 token 的表示都能看到左右上下文。对于标注任务至关重要。

## 构建

### 步骤 1：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将形状从 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。无论输入长度如何，池化后的输出都是固定大小。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行最大池化，而不是最后一个状态池化。对于分类，最大池化通常优于取最后一个隐藏状态，因为在长序列末尾的信息往往主导最后一个状态。

### 步骤 3：梯度消失演示（直觉）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中的任何位置。如果 `A` 在位置 1，序列长度为 100，那么损失的梯度必须通过 99 次循环权重乘法反向传播。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 通过一个**细胞状态**解决了这个问题，该状态仅通过加性交互在网络中运行（遗忘门以乘法方式缩放它，但梯度仍沿“高速公路”流动）。GRU 以更少的参数做了类似的事情。两者都能让你在超过 100 步的序列上稳定训练。

### 步骤 4：为什么这仍然不够

即使使用 LSTM，仍然存在三个问题。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 次串行的前向/反向传播步骤。无法跨时间步并行化。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器只看到编码器的最终隐藏状态，该状态压缩了整个输入。长输入会丢失细节。第 09 课将直接讨论这一点。
3. **远距离依赖的精度上限。** LSTM 优于普通 RNN，但仍难以在超过 200 步的距离上传播特定信息。

注意力机制解决了所有三个问题。Transformer 完全放弃了循环。第 10 课是转折点。

## 使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已可用于生产。训练代码是标准的。

Hugging Face 提供了预训练嵌入，你可以将其作为输入层插入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

在约束条件下使用的检查清单。

- **边缘/设备端推理。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是合适的方案。
- **流式/在线分类。** RNN 逐个处理 token；Transformer 需要完整序列。对于实时输入的文本，LSTM 仍然胜出。
- **用作基线的微型模型。** 在新任务上快速迭代。在 CPU 上 5 分钟训练一个 TextCNN。
- **数据有限的序列标注。** BiLSTM-CRF（第 06 课）仍然是生产级别的 NER 架构，适用于 1k-10k 条标注句子。

其他一切情况都交给 Transformer。

## 交付

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在三类玩具数据集（自己创造数据）上训练一个 TextCNN。验证使用多个滤波器宽度（2, 3, 4）比使用单一宽度（3）在平均 F1 上表现更好。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后一个状态池化。在小型数据集上比较；记录哪种池化胜出，并推测原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课中仅用 CRF 的基线以及 BERT 微调进行比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 人们所说 | 实际含义 |
|------|----------|----------|
| TextCNN | 用于文本的 CNN | 对词嵌入进行一维卷积堆叠并接全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 在每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 添加输入/遗忘/输出门 + 细胞状态。可在长序列中稳定训练。 |
| GRU | 更简单的 LSTM | 两个门代替三个门。类似精度，更少参数。 |
| Bidirectional | 双向 | 正向 + 反向 RNN 拼接。每个 token 都能看到其上下文的两侧。 |
| Vanishing gradient | 训练信号消失 | 普通 RNN 中反复乘以小于 1 的权重，导致早期步骤的梯度实际上为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。易读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 变得人人可懂的图示。
