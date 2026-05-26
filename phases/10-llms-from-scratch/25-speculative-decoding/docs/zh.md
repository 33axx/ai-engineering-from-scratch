# 推测解码与 EAGLE

> 一个前沿的 LLM 生成一个 token 需要对数十亿参数进行一次完整的前向传播。这个前向传播的资源是严重过剩的：大多数情况下，一个更小的模型就能正确预测接下来的 3-5 个 token，而大模型只需要**验证**这个预测。当预测正确时，你就用一次计算的代价获得了 5 个 token。推测解码（Leviathan 等人，2023）使这一过程精确无误，而 EAGLE-3（2025）将接受率推高至每次验证约 4.5 个 token —— 在保持输出分布一致的情况下实现 4-5 倍的加速。

**类型：** 构建  
**语言：** Python（使用 numpy）  
**前置知识：** 阶段 10 第 12 课（推理优化），阶段 10 第 4 课（预训练 Mini-GPT）  
**预计时间：** 约 75 分钟

## 问题

在 H100 上，70B 规模模型的解码吞吐量通常为 40-80 个 token/秒。每个 token 都需要一次完整的前向传播，从 HBM 读取所有模型权重。你无法在不改变模型输出的前提下让模型更小。你也无法在内存限制内增大批处理大小。你陷入了困境——除非你能让模型每次前向传播输出多个 token。

自回归生成看起来是本质串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个并发的机会。如果你有一个廉价的预测器，它说“接下来的 4 个 token 很可能是 [a, b, c, d]”，那么你可以在大模型的**单次前向传播**中验证全部 5 个位置，并接受最长匹配前缀。

Leviathan、Kalai、Matias（2023 年，“Fast Inference from Transformers via Speculative Decoding”）通过一个巧妙的接受/拒绝规则实现了这一点的精确性，该规则保留了目标模型的采样分布。相同的输出分布，速度提升 2-4 倍。

## 概念

### 双模型设置

- **目标模型** `M_p`：你实际想要采样的大、慢、高质量模型。分布：`p(x)`。
- **草稿模型** `M_q`：一个小、快、低质量模型。分布：`q(x)`。大小为目标的 5-30 分之一。

每一步：

1. 草稿模型自回归地生成 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 `K+1` 个位置并行执行**一次**前向传播，为每个提议的 token 生成 `p(x_k)`。
3. 使用下面的修正拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，则从修正后的分布中采样一个替代 token 并停止。否则从 `p(· | x_1...x_K)` 中采样一个奖励 token。

如果草稿与目标完美匹配，你每次目标前向传播得到 K+1 个 token。如果草稿在位置 1 就错了，你只得到 1 个 token。

### 精确性规则

推测解码在分布上**可证明等价于从 p 中采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致时（`p ≈ q`），接受率接近 1。当它们不一致时，残差分布被构造为使得整体样本仍然恰好是 `p`。

**贪婪情况。** 对于温度=0 的采样，只需检查 `argmax(p) == x_t`。如果是，则接受；如果否，则输出 `argmax(p)` 并停止。

### 预期加速

如果草稿模型的 token 级接受率为 `α`，则每次目标前向传播产生的预期 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每次前向传播。一次目标前向传播的成本大致为 `cost_q * K + cost_p`（K 步草稿加上一次目标验证）。如果 `cost_p >> cost_q * K`，则吞吐量的加速比为 `3.36× / 1 = 3.36×`。

唯一的实际参数是 `α`，它完全取决于草稿与目标的对齐程度。一个好的草稿就是一切。

### 训练草稿：蒸馏

一个随机的较小模型不是一个好的草稿。标准方法是从目标模型进行蒸馏：

1. 选择一个小的架构（对于 70B 目标 ~1B，对于 7B 目标 ~500M）。
2. 在大规模文本语料库上运行目标模型；存储其下一个 token 的分布。
3. 使用 KL 散度训练草稿，使其拟合目标模型的分布（而不是真实 token 的分布）。

结果：`α` 在编程类任务中通常为 0.6-0.8，在自然语言聊天中为 0.7-0.85。生产环境中加速 2-3 倍。

### EAGLE：树状草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024 年，“EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”）发现了标准推测解码中的两个低效问题：

1. 草稿执行 K 步串行步骤，每一步都是完整堆栈。但草稿可以复用最近一次验证中目标模型的特征（隐藏状态）——目标已经计算出了丰富的表示，而草稿正从头重新推导。
2. 草稿输出一条线性链。如果草稿能输出一个候选**树**（每个节点有多个猜测），那么目标模型的单次前向传播就可以通过树注意力掩码并行验证多个候选路径，并选择最长的已接受分支。

EAGLE-1 的更改：
- 草稿的输入 = 目标模型在位置 t 的最终隐藏状态，而不是原始 token。
- 草稿架构 = 1 个 transformer 解码器层（而不是一个独立的小模型）。
- 输出 = 每层 K = 4-8 个候选，深度为 4-6 的树。

EAGLE-2（2024）增加了动态树拓扑：在草稿不确定的地方树变宽，在确定的地方变窄。在不增加验证成本的情况下提高了 `α_effective`。

EAGLE-3（Li 等人，2025 年，“EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”）去除了固定的顶层特征依赖，并使用一种新的“测试时模拟”损失来训练草稿——草稿在匹配目标测试时分布的输出上进行训练，而不是在教师强制训练分布上训练。接受率从 0.75（EAGLE-2）提高到 0.82（EAGLE-3），平均 token/验证从 3.0 提高到 4.5。

### 树注意力验证

当草稿输出一棵树时，目标模型使用**树注意力掩码**在单次前向传播中验证它——这是一种因果掩码，它编码了树拓扑而不是一条直线。每个 token 只关注其在树中的祖先。验证仍然是一次前向传播，一次矩阵乘法；拓扑掩码只增加少量的额外 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一个 token 候选，而 `c, d, e, f` 是第二个 token 的候选，那么所有六个位置都在一次前向传播中得到验证。输出是沿着任何已接受路径的最长前缀。

### 何时有效，何时无效

**有效：**
- 具有可预测文本的聊天/补全（代码、常见英语、结构化输出）。`α` 较高。
- 解码期间 GPU 计算资源未被充分利用的场景（内存受限阶段）。树状草稿利用了可用的 FLOPs。

**无效/无增益：**
- 高度随机的输出（高温度下的创意写作）。`α` 降至 `1/|vocab|`。
- 并发度非常高的批量服务——批量处理已经占满了 FLOPs，几乎没有空间进行树验证。
- 非常小的目标模型，草稿模型无法小很多。

生产环境通常报告聊天场景下加速 2-3 倍，代码生成加速 3-5 倍，而创意写作场景几乎无加速。

## 构建

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，它实现了精确的拒绝规则，并验证其保留了目标模型的分布（与纯目标采样相比，经验 KL < 0.01）。
- 一个 EAGLE 风格的树状草稿器，构建深度为 K、使用 top-p 分支的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试框架，在两个小型 LM 上运行（从一个 GPT-2-medium 目标中蒸馏出一个 GPT-2-small 草稿）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用

- **vLLM** 和 **SGLang** 内置了第一方推测解码支持。标志：`--speculative_model`，`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（用于 Qwen3-32B 的草稿），`meta-llama/Llama-3.2-1B-Instruct-spec`（用于 70B 的草稿）。
- **Medusa 头**（Cai 等人，2024 年，“Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads”）：不是使用草稿模型，而是在目标模型自身添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付

本课程产出 `outputs/skill-speculative-tuning.md` —— 一项技能，用于分析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到普通解码。

## 练习

1. 实现精确的拒绝规则并经验验证。通过 `speculative_decode` 和纯目标采样运行 10K 个样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的预期 token 数。找到 α ∈ {0.5, 0.7, 0.9} 时的最优 K。

3. 训练一个微小的草稿。将 124M GPT-2 作为目标，在 100M token 上使用 KL 损失蒸馏出一个 30M GPT-2 草稿。在保留文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树状草稿。不是一条链，而是让草稿在每层输出 top-3 分支。构建树注意力掩码。验证目标接受了最长的正确分支。

5. 测量失败模式。在温度=1.5（高随机性）下运行推测解码。显示 α 崩溃，并且由于草稿开销，算法比普通解码更慢。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | "大模型" | 你希望采样的大、慢、高质量模型（p 分布） |
| 草稿模型 | "推测器" | 小、快、低质量的预测器（q 分布）；大小为目标的 5-30 分之一 |
| K / 草稿长度 | "前瞻" | 每次验证时推测的 token 数 |
| α / 接受率 | "命中率" | 草稿提议被接受的 token 级概率 |
| 精确拒绝规则 | "接受测试" | 保留目标分布的 r < p/q 比较 |
| 残差分布 | "修正的 p-q" | (p - q)+ / \|\|(p - q)+\|\|_1，拒绝时从中采样的分布 |
| 树状草稿 | "分支推测" | 草稿输出一个候选树，通过树结构注意力掩码在单次前向传播中验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 在目标模型自身添加 K 个额外预测头；无需单独的草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿的输入是目标的最后一个隐藏状态，而非原始 token，从而缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在匹配目标测试时分布的输出上训练草稿，而非教师强制训练 |

## 扩展阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 同时期提出的推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 草稿模型的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用与树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无需推测器的替代方法
