# GPT — 因果语言建模

> BERT 能看两边，GPT 只看过去。三角形掩码是现代人工智能中最具影响力的一行代码。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 7 · 02（自注意力）、阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）  
**时长：** 约 75 分钟  

## 问题

语言模型回答一个问题：给定前 `t-1` 个词元，第 `t` 个词元的概率分布是什么？基于这个信号训练——下一个词元预测——你就得到了一个能逐个词元生成任意文本的模型。

为了以端到端方式并行处理整个序列进行训练，每个位置的预测必须仅依赖于前面的位置。否则模型会通过偷看答案而被训练取巧。

因果掩码就是做这件事的。它是一个上三角矩阵，值为 `-inf`，在 softmax 之前加到注意力得分上。经过 softmax 之后，那些位置变成 0。每个位置只能关注自己和前面的位置。由于你只对整个序列应用一次掩码，你就在一次前向传播中得到了 N 个并行的下一个词元预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是仅解码器（decoder-only）的因果 Transformer，核心循环相同。只是规模更大、数据更好、RLHF 更好。

## 概念

![因果掩码生成三角形注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定一个长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力得分上。`exp(-inf) = 0`，因此被掩码的位置贡献的权重为零。注意力矩阵的每一行都是仅针对前面位置的概率分布。

实现代价：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 并行训练，串行推理

训练：一次前向传播整个 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列维度并行。这就是 GPT 训练能够扩展的原因——你可以在一次 GPU 前向传播中处理一个 batch 中的 100 万个词元。

推理：你逐个词元生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）可以保存 `t1…tn` 的隐藏状态，这样你就不必每一步都重新计算它们。但推理时的串行深度等于输出长度。这就是自回归的代价，也是每个 LLM 解码成为延迟瓶颈的原因。

### 损失——移位一个位置

给定词元 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每一个 Transformer 语言模型都使用这个损失进行训练。预训练、微调、SFT——同一损失，不同数据。

### 解码策略

训练完成后，采样选择的重要性超出人们的想象。

| 方法 | 作用 | 何时使用 |
|------|------|----------|
| 贪心解码 | 每一步取 argmax | 确定性任务、代码补全 |
| 温度采样 | 将 logits 除以 T 后采样 | 创意任务，T 越高 → 多样性越大 |
| Top-k | 仅从 top-k 个词元中采样 | 消除低概率尾巴 |
| Top-p（核采样） | 从累积概率 ≥p 的最小集合中采样 | 2020 年后的默认方法；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024 年后的新方法；在拒绝长尾方面优于 top-p |
| 推测解码 | 草稿模型提出 N 个 token，大模型验证 | 同样质量下延迟降低 2–3 倍 |

2026 年，min-p + 温度 0.7 是开放权重模型的合理默认配置。推测解码是任何生产推理栈的必备条件。

### 是什么让“GPT 配方”奏效

1. **仅解码器。** 无编码器开销。每层一次注意力 + FFN 前向传播。
2. **扩展。** 124M → 1.5B → 175B → 万亿参数。Chinchilla 缩放定律（第 13 课）告诉你如何分配算力。
3. **上下文学习。** 在 6B–13B 左右涌现。模型可以遵循少样本示例而无需微调。
4. **RLHF。** 在人类偏好上进行的后训练将原始预训练文本转化为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模训练稳定。

从 GPT-2 开始，核心架构变化不大。所有有趣的事情都发生在数据、规模和后训练上。

## 构建它

### 第 1 步：因果掩码

参见 `code/main.py`。只需一行：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力得分上。这就是整个机制。

### 第 2 步：一个 2 层的类 GPT 模型

堆叠两个解码器块（带掩码的自注意力 + FFN，无交叉注意力）。添加词元嵌入、位置编码和解嵌入（与词元嵌入矩阵共享——自 GPT-2 以来的标准技巧）。

### 第 3 步：下一个词元预测，端到端

在一个 20 个词元的玩具词汇表上，在每个位置产生 logits。计算与移位一位的目标之间的交叉熵损失。不需要梯度——这是一个前向传播的完整性检查。

### 第 4 步：采样

实现贪心、温度、top-k、top-p、min-p。在固定 prompt 上运行每种方法并比较输出。一个采样函数只需 10 行代码。

## 使用它

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，取出最后位置的 logits，采样下一个词元，追加它，然后重复。每个生产级 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现相同的循环，并进行了重度优化——批量 prefill、连续批处理、KV 缓存分页、推测解码。

**GPT vs BERT，一行概括：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型能否生成。

## 交付它

参见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时发出提示。

## 练习

1. **简单。** 运行 `code/main.py` 并验证 softmax 后的因果注意力矩阵是下三角的。抽查：第 3 行应仅在列 0–3 有权重。
2. **中等。** 实现宽度为 4 的集束搜索。在 10 个短 prompt 上比较 beam-4 与贪心解码的困惑度。集束搜索总是胜出吗？（提示：通常对翻译是，但对开放式聊天不是。）
3. **困难。** 实现推测解码：使用一个极小的 2 层模型作为草稿，一个 6 层模型作为验证器。在 100 个长度为 64 的补全上测量墙钟加速。确认输出与验证器的贪心解码一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|----------|
| Causal mask | “那个三角形” | 上三角 `-inf` 矩阵，加到注意力得分上，使位置 `i` 只能看到位置 `≤ i`。 |
| Next-token prediction | “损失” | 模型分布与真实下一个词元在每个位置上的交叉熵。 |
| Autoregressive | “一次生成一个” | 将输出作为输入反馈回去；仅在训练时并行，生成时不能。 |
| Logits | “softmax 之前的分数” | 语言模型头产生 softmax 之前的原始输出；采样是在这些值上进行的。 |
| Temperature | “创意旋钮” | 将 logits 除以 T；T→0 为贪心，T→∞ 为均匀。 |
| Top-p | “核采样” | 将分布截断到求和≥p 的最小集合；从剩余部分中采样。 |
| Min-p | “比 top-p 更好” | 保留 `p ≥ min_p × max_p` 的 token；根据分布锐度自适应截断。 |
| Speculative decoding | “草稿加验证” | 廉价模型提出 N 个 token；大模型并行验证。 |
| Teacher forcing | “训练技巧” | 训练时，喂入真实的前一 token，而不是模型自己的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码论文
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果 LM 参考代码
