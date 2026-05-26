# 近端策略优化（PPO）

> A2C 在一次更新后丢弃整个 rollout。PPO 将策略梯度包裹在裁剪后的重要性比率中，从而可以在同一批数据上执行 10 轮以上的更新，而不会导致策略爆炸。Schulman 等人（2017）。截至 2026 年，它仍然是默认的策略梯度算法。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第 9 阶段·06（REINFORCE），第 9 阶段·07（Actor-Critic）  
**时间：** 约 75 分钟

## 问题

A2C（第 07 课）是在策略算法：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据来自*当前*的 `π_θ`。进行一次更新后，`π_θ` 就会改变；你使用的数据现在变成了离策略。重复使用这些数据会导致梯度有偏。

Rollout 成本高昂。在 Atari 上，一次 rollout 跨越 8 个环境 × 128 步 = 1024 个转换，以及十几秒的环境时间。在一次梯度步骤后就丢弃它们是很浪费的。

信任区域策略优化（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很清晰，但每次更新都需要求解共轭梯度。2026 年没有人跑 TRPO。

PPO（Schulman 等人，2017）用一个简单的裁剪目标替代了严格的信任区域约束。多一行代码。每次 rollout 进行十轮更新。没有共轭梯度。足够好的理论保证。九年后的今天，它仍然是默认的策略梯度算法，用于从 MuJoCo 到 RLHF 的一切场景。

## 概念

![PPO 裁剪替代目标：比率裁剪在 1 ± ε 范围内](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略之间的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

包含两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪会压平梯度——不要将好的动作推高到高于旧概率 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（意味着我们会使一个坏动作更可能发生，相比于被裁剪减少的情况），裁剪会限制梯度——不要将坏动作推低到低于 `-ε`。

`min` 处理另一个方向：如果比率已经向*有利*方向移动，你仍然获得梯度（在可能损害你的那一侧不进行裁剪）。

典型 `ε = 0.2`。绘制目标关于 `r_t` 的函数：这是一个分段线性函数，在“好的一侧”有平顶，在“坏的一侧”有平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 跨 `N` 个并行环境收集 `N × T` 个转换，每个环境跑 `T` 步。
2. 计算优势（GAE），将其冻结为常数。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 对于 `K` 轮，对于每个小批量 `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步骤。
5. 丢弃该 rollout。返回步骤 1。

`K = 10`，小批量大小为 64 是标准超参数集。PPO 很鲁棒：确切数值在 ±50% 范围内很少影响结果。

**KL 惩罚变体。** 原始论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，并根据观测到的 KL 调整 `β`。裁剪版本成为主流；KL 变体在 RLHF 中存活下来（其中与参考策略的 KL 是你始终需要的一个单独约束）。

## 构建它

### 第 1 步：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照仅在 rollout 时获取一次。在更新轮次中不会改变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。对整个批次进行归一化。

### 第 3 步：裁剪替代目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

这种“裁剪 → 梯度为零”的模式是 PPO 的核心。如果新策略已经向有利方向漂移太远，更新停止。

### 第 4 步：价值和熵

添加标准的 MSE 到评论家目标，以及 actor 上的熵奖励，与 A2C 相同。

### 第 5 步：诊断

每次更新需观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 范围内。如果超过 `0.1`，减小 `K_EPOCHS` 或 `LR`。
- **裁剪比例** —— 比率落在 `[1-ε, 1+ε]` 之外的样本比例。应在 `~0.1-0.3` 左右。如果接近 `0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果接近 `0.5+`，你在过拟合该 rollout → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评论家质量指标。随着评论家学习，应逐步向 1 增加。

## 陷阱

- **裁剪系数调节不当。** `ε = 0.2` 是事实上的标准。降到 `0.1` 会使更新过于胆怯；`0.3+` 会引发不稳定。
- **轮次过多。** `K > 20` 通常会破坏稳定性，因为策略会远离 `π_old`。限制轮次，尤其是对于大型网络。
- **没有奖励归一化。** 大的奖励尺度会侵蚀裁剪范围。在计算优势之前对奖励进行归一化（运行标准差）。
- **忘记优势归一化。** 每批次的零均值/单位标准差归一化是标准做法。跳过它对大多数基准测试都会破坏 PPO。
- **学习率不衰减。** PPO 受益于线性学习率衰减至零。恒定学习率通常更差。
- **重要性比率数学错误。** 为数值稳定性，始终使用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度方向错误。** 最大化替代目标 = *最小化* `-L^{CLIP}`。符号反转是 PPO 最常见的 bug。

## 使用它

PPO 是 2026 年许多领域的默认 RL 算法：

| 用例 | PPO 变体 |
|------|----------|
| MuJoCo / 机器人控制 | 带有高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带有分类策略的 PPO，滚动 128 步 rollout |
| LLM 的 RLHF | 带有 KL 惩罚（相对于参考模型）的 PPO，来自响应末尾的奖励模型 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar, OpenAI Five） |
| 推理型 LLM | GRPO（第 12 课）—— 无评论家的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭形式折叠，无需在线采样 |

PPO 的*损失形状* —— 裁剪替代目标 + 价值 + 熵 —— 是 DPO、GRPO 及几乎所有 RLHF 流水线的基础架构。

## 交付它

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。在匹配的环境步数下，与 A2C（每次 rollout 一轮）对比样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数，并跟踪每次更新的平均 KL。在这个任务上，KL 在哪个 `K` 值下爆炸？
3. **困难。** 用自适应 KL 惩罚（如果 `KL > 2·target` 则 `β` 加倍，如果 `KL < target/2` 则 `β` 减半）替换裁剪替代目标。比较最终回报、稳定性和无裁剪性。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据的策略的偏差。 |
| 裁剪替代目标 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利一侧超过裁剪后梯度变平。 |
| 信任区域 | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任区域" | PPO 的替代方案：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 | "裁剪触发的频率" | 诊断指标——应在 0.1-0.3 之间；超出表示调节不当。 |
| 多轮次训练 | "数据复用" | 每个 rollout 执行 K 轮；方差成本换取样本效率。 |
| 近似在策略 | "主要是在策略" | PPO 名义上是在策略，但 K>1 轮安全地使用略微离策略的数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；在 RLHF 中使用，因为到参考策略的 KL 本身就是一个约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对所有 PPO 超参数的消融研究。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；PPO 在 RLHF 中的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 清晰的现代讲解，附 PyTorch 代码。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 许多论文引用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— PPO 在语言模型上的生产配方；配合第 09 课（RLHF）阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— “37 个代码级优化”论文；哪些 PPO 技巧是关键的，哪些是民间传说。
