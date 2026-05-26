# 混合专家模型（Mixture of Experts, MoE）

> 一个稠密的 70B Transformer 对每个 token 都会激活每一个参数。而一个 671B 的 MoE 每个 token 只激活 37B 参数，并且在所有基准测试上都超过前者。稀疏化是这十年最重要的扩展思路。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 7 · 05（完整 Transformer）、阶段 7 · 07（GPT）  
**时长：** 约 45 分钟

## 问题所在

稠密 Transformer 在推理时的 FLOPs 等于其参数量（前向传播乘以 2）。扩展一个稠密模型时，每个 token 都会支付全部开销。到 2024 年，前沿模型已经遇到计算墙：要变得更有意义地智能，每个 token 需要的 FLOPs 呈指数级增长。

混合专家模型打破了这一联系。将每个 FFN 替换为 `E` 个独立专家 + 一个路由器，每个 token 选择 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的活跃参数量 = `k × FFN_size`。典型的 2026 年配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3（671B 总参数 / 37B 活跃）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，排名前 10 的开源模型全是 MoE。

## 核心思想

![MoE 层：路由器为每个 token 从 E 个专家中选择 k 个](../assets/moe.svg)

### FFN 替换

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的 FFN（通常是 SwiGLU）。路由器是一个线性层。每个 token 选择自己的 `k` 个专家，并得到它们输出的门控混合。

### 负载均衡问题

如果路由器将 90% 的 token 送入专家 3，其他专家就会挨饿。人们尝试过三种修复方法：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。增加一个与专家使用方差成正比的惩罚项。有效，但引入了一个超参数和第二个梯度信号。
2. **专家容量 + 丢 token**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；超出部分跳过该层。损害质量。
3. **免辅助损失均衡**（DeepSeek-V3）。为每个专家添加一个可学习的偏置，该偏置会改变路由器的 top-k 选择。偏置在训练损失之外更新。不对主目标造成惩罚。2024 年的重大突破。

DeepSeek-V3 的方法：每个训练步骤后，检查每个专家的使用量是否高于或低于目标值。将偏置向 `±γ` 调整。选择使用 `scores + bias`。用于门控的专家概率仍是原始的 `scores`，不做改动。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为**共享专家**和**路由专家**。每个 token 通过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家负责专业化。V3 使用 1 个共享专家加上从 256 个路由专家中选择 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家宽度与完整 FFN 相同。`E` 较小（8–64），`k` 较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（FFN 大小的 1/8）。`E` 较大（256 以上），`k` 较大（8 以上）。总参数量相同，但组合数量扩展更快。每个 token 可能的“专家”组合数为 `C(256, 8) = 400 万亿`。质量提升，延迟保持平坦。

### 成本概况

每个 token，每层：

| 配置 | 每个 token 的活跃参数 | 总参数 |
|------|-----------------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试上击败 Llama 3 70B（稠密），同时每个 token 的**活跃 FLOPs 更少**。更多参数 = 更多知识。更多活跃 FLOPs = 每个 token 更多计算。MoE 将它们解耦。

### 代价：内存

所有专家都住在 GPU 上，无论是否被触发。一个 671B 的模型在 fp16 权重下需要约 1.3 TB 的显存。前沿 MoE 部署需要专家并行——将专家分片到多个 GPU，将 token 路由到网络各处。延迟主要来自 all-to-all 通信，而非矩阵乘法。

## 动手构建

见 `code/main.py`。一个紧凑的 MoE 层，仅使用 Python 标准库实现，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个仅一个线性层，用于演示）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过每个专家偏置实现的免辅助损失均衡

### 第1步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，但不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡，而不影响模型的预测。

### 第2步：让 100 个 token 通过路由器

追踪每个专家被触发的次数。如果没有偏置，使用分布会倾斜。通过偏置更新循环（对于过度使用的专家使用 `-γ`，对于使用不足的专家使用 `+γ`），使用分布会在几次迭代后收敛到均匀分布。

### 第3步：参数数量对比

打印 MoE 配置的“稠密等价”参数数量。类似 DeepSeek-V3 的配置：256 个路由专家 + 1 个共享专家，8 个活跃专家，d_model=7168。总参数量令人惊叹。活跃参数量仅为稠密 Llama 3 70B 的七分之一。

## 使用它

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产级推理：vLLM 原生支持 MoE 路由。SGLang 拥有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你希望以更低的每个 token 推理成本获得前沿质量。
- 你拥有足够的显存 / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而不是上下文密集型（长文档）。

**何时不要选择 MoE：**
- 边缘设备部署——任何活跃 FLOP 都需要支付全部存储开销。
- 延迟敏感的单用户服务——专家路由增加额外开销。
- 小模型（<7B）——MoE 的质量优势仅在超过一定计算阈值（约 6B 活跃参数）时才显现。

## 部署它

见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 数量和部署目标，为新的 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单：** 运行 `code/main.py`。观察免辅助损失偏置更新如何在 50 次迭代内平衡专家使用率。
2. **中等：** 将学习得到的路由器替换为基于哈希的路由器（确定性，无需学习）。比较质量与平衡性。为什么学习得到的路由器更好？
3. **困难：** 实现 GRPO 风格的“rollout-matched routing”（DeepSeek-V3.2 技巧）：记录推理期间哪些专家被触发，在梯度计算时强制使用相同的路由。在一个简单的策略梯度设置上测量其效果。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------|
| 专家 | “众多 FFN 中的一个” | 一个独立的前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| 路由器 | “门控” | 一个微小的线性层，为每个 token 对每个专家打分；进行 top-k 选择。 |
| Top-k 路由 | “每个 token 有 k 个活跃专家” | 每个 token 的 FFN 计算恰好经过 k 个专家，并通过门控加权。 |
| 辅助损失 | “负载均衡惩罚” | 额外损失项，惩罚偏斜的专家使用。 |
| 免辅助损失 | “DeepSeek-V3 的技巧” | 通过路由器选择中的每个专家偏置实现均衡；无额外梯度。 |
| 共享专家 | “始终开启” | 额外专家，每个 token 都会经过；捕获通用知识。 |
| 专家并行 | “按专家分片” | 将不同专家分配到不同 GPU；通过网络路由 token。 |
| 稀疏性 | “活跃参数 < 总参数” | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 该想法。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 免辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课程路由器使用的细粒度 + 共享专家拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 最初的共享专家论文。
