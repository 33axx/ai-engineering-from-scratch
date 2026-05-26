# 策略梯度 —— 从头实现 REINFORCE

> 不再估计价值。直接参数化策略，计算期望回报的梯度，沿上坡方向更新。Williams（1992）用一条定理完成了这一工作。这正是 PPO、GRPO 以及所有 LLM 强化学习循环存在的根基。

**类型：** 动手构建
**语言：** Python
**前置知识：** 第 3 阶段 · 03（反向传播），第 9 阶段 · 03（蒙特卡洛），第 9 阶段 · 04（时序差分学习）
**预计时间：** ~75 分钟

## 问题

Q 学习和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 选择动作。这在离散动作和离散状态下没问题。但当动作是连续的（你能对 10 维力矩做 `argmax` 吗？）或者当你想要随机策略时（`argmax` 天生是确定性的），它就不再适用了。

策略梯度直接参数化*策略*。`π_θ(a | s)` 是一个神经网络，输出动作上的分布。从中采样来行动。计算期望回报关于 `θ` 的梯度。沿上坡方向更新。没有 `argmax`。没有贝尔曼递推。只有对 `J(θ) = E_{π_θ}[G]` 的梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合，计算回报，乘以每一步的 `∇ log π_θ(a | s)`，取平均，梯度上升，完成。

2026 年的每一个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。亲手理解它是本阶段其余内容以及第 10 阶段 · 07（RLHF 实现）和第 10 阶段 · 08（DPO）的前提。

## 概念

![策略梯度：softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从第 `t` 步开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 而言的。

**证明很简短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log-导数技巧）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 环境项（不依赖于 θ）`。环境项消失。两步代数推导就得到了定理。

**方差缩减技巧。** 原始 REINFORCE 的方差大得惊人——回报有噪声，`∇ log π` 有噪声，它们的乘积噪声更大。两个标准修正：

1. **基线减法。** 用 `G_t - b(s_t)` 替换 `G_t`，其中 `b(s_t)` 是不依赖于 `a_t` 的任意基线。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)`，由评论家学习得到 → 演化为行动器-评论家（第 07 课）。
2. **奖赏回溯。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。给定动作，只有未来的回报才有意义——过去的回报贡献零均值噪声。

结合起来，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意神经网络，输出每个动作的分数。梯度具有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所采取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。这就是第 9 阶段 · 07 的 SAC 所需要的全部。

## 动手构建

### 第 1 步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格型环境使用线性策略（每个动作一个权重向量）。对于 Atari，换成 CNN 并保留 softmax 输出头。

### 第 2 步：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：捕获对数概率的轨迹采集

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的独热编码减去概率向量）是 softmax 策略梯度的核心。把它刻进肌肉记忆里。

### 第 5 步：基线

使用近期几个回合中 `G` 的滑动均值作为基线，足以让 4×4 网格世界开始运行；大约 500 个回合后收敛。将基线升级为学习得到的 `V̂(s)`，你就得到了行动器-评论家。

## 常见陷阱

- **梯度爆炸。** 回报可能非常大。在乘以 `∇ log π` 之前，一定要将整批数据的 `G` 归一化到 `~N(0, 1)`。
- **熵崩塌。** 策略过早收敛到近乎确定性的动作，停止探索，陷入僵局。修正：在目标中加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个回合。评论家基线（第 07 课）或 TRPO/PPO 的信任区域（第 08 课）是标准的修正方案。
- **样本低效。** 在策略意味着每次更新后你都得丢弃所有转移样本。通过重要性采样进行离策略修正可以重用数据，但代价是方差增加（PPO 的截断比就是被截断的重要性采样权重）。
- **非平稳梯度。** 100 个回合前的同一个梯度使用的是旧的 `π`。在线策略方法为此会每隔几次轨迹采集就更新一次。
- **信用分配。** 如果不使用奖赏回溯，过去的奖励会贡献噪声。始终使用奖赏回溯。

## 应用

2026 年，REINFORCE 很少被直接运行，但它的梯度公式无处不在：

| 使用场景 | 衍生方法 |
|----------|----------|
| 连续控制 | PPO / SAC（高斯策略） |
| LLM RLHF | 带 KL 惩罚的 PPO，运行在词元级策略上 |
| LLM 推理（DeepSeek） | GRPO —— 带组相对基线的 REINFORCE，无评论家 |
| 多智能体 | 集中式评论家 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO —— 重写为偏好似然损失的 REINFORCE，无需采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，那正是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是基于这一行代码的方差缩减技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 网格世界上用线性 softmax 策略实现 REINFORCE。训练 1000 个回合，不使用基线。画出学习曲线；测量方差（回报的标准差）。
2. **中等。** 加入滑动均值基线。再次训练。将样本效率和方差与原始版本对比。基线将收敛所需的步数减少了多少？
3. **困难。** 加入熵奖励 `β · H(π)`。对 `β ∈ {0, 0.01, 0.1, 1.0}` 进行扫描。画出最终回报和策略熵。在这个任务上，最佳的 `β` 是多少？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 策略梯度 | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；由 log-导数技巧推导而来。 |
| REINFORCE | “原始 PG 算法” | Williams (1992)；蒙特卡洛回报乘以对数策略梯度。 |
| Log-导数技巧 | “得分函数估计器” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使得期望的梯度变得可处理。 |
| 基线 | “方差缩减” | 从 `G` 中减去的任意 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| 奖赏回溯 | “只算未来回报” | 使用 `G_t^{from t}` 而非完整的 `G_0`；正确且方差更低。 |
| 熵奖励 | “鼓励探索” | `+β · H(π(·|s))` 项防止策略崩塌。 |
| 在策略 | “用刚看到的数据训练” | 梯度期望是关于当前策略的——不能直接重用旧数据。 |
| 优势 | “比平均好多少” | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) —— REINFORCE 原始论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 使用函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 —— Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式的讲解。
- [OpenAI Spinning Up —— VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 清晰的启蒙式讲解，附 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差缩减以及将 REINFORCE 与信任区域家族（TRPO、PPO）联系起来的自然梯度视角。
