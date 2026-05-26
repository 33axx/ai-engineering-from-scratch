# 推测解码 —— 起草、验证、重复

> 自回归解码是串行的。每个词元都要等待前一个词元。推测解码打破了这一链条：一个廉价模型起草 N 个词元，昂贵模型在一个前向传播中验证全部 N 个词元。当草稿正确时，你只需为 N 次生成支付一次大模型前向传播的代价。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 7 · 07（GPT 因果语言模型），阶段 7 · 12（KV 缓存与 Flash Attention）
**时间：** 约 60 分钟

## 问题所在

一个 70B 的 LLM 在 H100 上采样一个词元大约需要 30 毫秒。一个 3B 的草稿模型大约需要 3 毫秒。如果我们让 3B 模型提前起草 5 个词元，然后运行 70B 模型 *一次* 来验证所有 5 个词元，那么总共需要 `5×3 + 30 = 45` 毫秒，最多可接受 5 个词元 —— 而直接生成则需要 `5×30 = 150` 毫秒。这就是完整的推测解码宣传：用少量的额外 GPU 内存（草稿模型）换取 2–4 倍更低的解码延迟。

这个技巧必须保持分布不变。由 Leviathan 等人（2023）以及 Chen 等人同时提出的推测采样，保证了输出序列与大模型自己生成的序列 **同分布**。没有质量折衷。只是更快。

截至 2026 年，主导推理的四种草稿-验证器组合：

1. **原始推测（Leviathan 2023）。** 独立的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。无需独立的草稿模型。
3. **EAGLE 家族（Li 2024, 2025）。** 轻量级草稿，重用验证器的隐藏状态；接受率高于原始推测；典型加速 3–4 倍。
4. **超前解码（Fu 2024）。** Jacobi 迭代；根本不需要草稿模型。自我推测。小众但无依赖。

2026 年，每个生产推理栈都默认提供推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持原始推测 + EAGLE-2。

## 概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿模型 `M_p`：

1. 令 `x_1..x_k` 为已解码的前缀。
2. **起草**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并给出草稿概率 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行一次 `M_q`，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿词元**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 第一次拒绝时：从归一化后的“残差”分布 `(q_j - p_j)_+` 中采样 `t_j`。丢弃位置 `j` 之后的所有草稿。
6. 如果所有 `N` 个词元都被接受：从 `q_{N+1}` 中额外采样一个词元 `t_{N+1}`（免费的奖励词元）。

残差分布技巧是数学上的关键洞察，它确保输出的分布与 `M_q` 从头开始采样时的分布完全一致。

### 决定加速比的因素

令 `α` = 每个草稿词元的预期接受率。令 `c` = 草稿与验证器的成本比。每步：

- 朴素生成的每一步需要 1 次大模型调用。
- 推测生成中，当 `α` 较高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个词元需要 1 次大模型调用。

典型经验法则：当 `α = 0.75` 且 `N = 5` 时，大模型调用次数减少约 3 倍。草稿成本为 5 倍廉价。总挂钟时间下降约 2.5 倍。

**`α` 取决于：**

- 草稿模型与验证器的近似程度。同一家族 / 相同训练数据会显著提高 α。
- 解码策略。贪婪草稿匹配贪婪验证器：α 高。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测）；自由形式创意写作接受率较低。

### Medusa —— 无需草稿模型的起草

Medusa 用验证器上的额外输出头代替草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。在推理时，你对每个头进行采样以获得一个候选序列，然后使用树注意力机制（一次性考虑所有候选延续）进行一次前向传播验证。

优点：没有第二个模型。缺点：增加了可训练参数；需要一个监督微调阶段（约 10 亿词元）；与具有良好草稿的原始推测相比，接受率略低。

### EAGLE —— 通过重用隐藏状态实现更好的草稿

EAGLE-1/2/3（Li 等人，2024–2025）使草稿模型成为一个微型 Transformer（通常为 1 层），它接收验证器最后一层的隐藏状态。由于草稿看到了验证器的特征表示，其预测与验证器的输出分布高度相关。接受率从约 0.6（原始推测）上升到 0.85 以上。

EAGLE-3（2025）增加了对候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测路径。

### KV 缓存的舞蹈

验证时，将 `N` 个草稿词元一次性输入验证器。这会将验证器的 KV 缓存扩展 `N` 个条目。如果某些草稿被拒绝，你必须将缓存回滚到已接受前缀的长度。

生产实现（vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）使用暂存 KV 缓冲区处理此问题。先写入，接受后提交。这在概念上并不困难，但很繁琐。

## 构建它

参见 `code/main.py`。我们实现了核心的推测采样算法（拒绝步骤 + 残差分布），包括：

- 一个“大模型”，它在手工编码的分布上执行确定性 softmax（以便我们可以解析地验证接受数学）。
- 一个“草稿模型”，它是大模型的扰动版本。
- 一个接受/拒绝循环，产生与直接采样相同的边际分布。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个均匀随机数。`q_prob` 是验证器对草稿词元的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明，这个伯努利决策，随后在拒绝时从残差中采样，精确保持了验证器的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

将 `q` 逐元素减去 `p`，将负值截断为零，重新归一化。在任何拒绝时从此分布中采样。

### 步骤 3：一个推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个奖励词元 → 在一次验证器前向传播中产生六个词元。

### 步骤 4：测量接受率

在不同草稿质量水平下运行 10,000 个推测步骤。绘制接受率与草稿和验证器分布之间的 KL 散度关系图。你应该会看到一个清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：推测循环产生的词元直方图应与直接从验证器采样产生的直方图匹配。这是 Leviathan 定理的实践验证。卡方检验确认在采样误差范围内一致。

## 使用它

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 通过一个小型草稿模型为 Whisper-large 包装了推测解码。

**选择草稿策略：**

| 策略 | 何时选择 | 加速比 |
|----------|--------------|---------|
| 原始草稿（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 头 | 你可以对验证器进行微调 | 2–3× |
| EAGLE-2 / 3 | 生产环境，最大速度 | 3–4× |
| 超前解码 | 无需草稿，无需训练，无需额外参数 | 1.3–1.6× |

**什么时候不要推测解码：**

- 长度为 1–5 个词元的单序列生成。开销占主导。
- 极度创意 / 高温度采样（α 下降）。
- 内存受限的部署（草稿模型增加 VRAM）。

## 部署它

参见 `outputs/skill-spec-decode-picker.md`。该技能为一个新的推理工作负载选择一种推测解码策略（原始推测 / Medusa / EAGLE / 超前解码）和调优参数（N，草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个词元上确认推测词元分布与验证器的直接采样分布一致，卡方 p > 0.05。
2. **中等。** 对于 `α = 0.5, 0.7, 0.85`，绘制加速比（每个大模型前向的词元数）关于 `N` 的函数。找出每个 α 的最优 `N`。（提示：每次验证调用的预期词元数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个小型 Medusa：取第 14 课的 capstone GPT，添加 3 个额外的 LM 头，用于预测位置 t+2, t+3, t+4。使用联合多头损失在 tinyshakespeare 上训练。与通过截断同一模型制作的原始草稿比较接受率。
4. **困难。** 实现回滚：从一个包含 10 个词元前缀的 KV 缓存开始，输入 5 个草稿词元，模拟在位置 3 被拒绝。验证你的缓存读取在下一个迭代中正确匹配“前缀 + 前 2 个已接受草稿”。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型 | “廉价的那个” | 一个更小的模型，用于提出候选词元；通常比验证器便宜 10–50 倍。 |
| 验证器 | “大的那个” | 目标模型，我们要保持其分布不变；每个推测步骤运行一次。 |
| 接受率 (α) | “草稿正确的频率” | 每个词元被验证器接受的概率。典型值为 0.7–0.9。 |
| 残差分布 | “拒绝后的后备方案” | `(q - p)_+` 归一化；拒绝后从此分布采样以保持验证器的分布。 |
| 奖励词元 | “免费的那个” | 当所有 N 个草稿被接受时，从验证器的下一步分布中再采样一个词元。 |
| Medusa | “无需草稿的推测” | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | “隐藏状态草稿” | 微型 Transformer 草稿，以验证器最后一层的隐藏状态为条件。 |
| 超前解码 | “Jacobi 迭代” | 使用不动点迭代的自我推测；不需要草稿模型。 |
| 树注意力 | “一次性验证多个候选” | 分支验证，同时考虑多个草稿延续。 |
| KV 回滚 | “撤销被拒绝的草稿” | 暂存 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法及等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同期介绍；简洁的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— 超前解码，无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 规范的生产参考，包含所有四种策略的实现。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。
