# Actor-Critic — A2C 与 A3C

> REINFORCE 噪声大。加入一个学习 `V̂(s)` 的评论家，从回报中减去它，你就得到了一个期望相同但方差低得多的优势值。这就是 actor-critic。A2C 同步运行；A3C 跨线程运行。两者都是所有现代深度强化学习方法的思维模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 04（时序差分学习），第 9 阶段 · 06（REINFORCE）
**时长：** 约 75 分钟

## 问题

原始的 REINFORCE 可以工作，但其方差非常糟糕。蒙特卡洛回报 `G_t` 在不同回合之间可能波动 10 倍。将这种噪声乘以 `∇ log π` 并求平均，产生的梯度估计需要数千个回合才能将策略朝某个方向移动，而使用 DQN 更新只需少得多的次数就能移动相同的距离。

方差源于使用原始回报。如果你减去一个基线 `b(s_t)` —— 任何关于状态的函数，包括一个学习到的价值函数 —— 期望不变，方差降低。最易处理的基线是 `V̂(s_t)`。此时与 `∇ log π` 相乘的量就是*优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报，它就是好的；低于平均则是坏的。带有学习到的评论家的 REINFORCE 就是 *actor-critic*。评论家为演员提供了一个低方差的教师。这是 2015 年后所有深度策略方法（A2C, A3C, PPO, SAC, IMPALA）的核心思想。

## 概念

![Actor-critic: 策略网络加价值网络，以时序差分残差作为优势值](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员** `π_θ(a | s)`：策略。采样以执行动作。通过策略梯度训练。
- **评论家** `V_φ(s)`：估计从状态开始的期望回报。通过最小化 `(V_φ(s) - 目标)²` 来训练。

**优势值。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现中，Atari 使用 `n = 5`，MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人 (2016) 提出对所有的 n 步优势进行指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年默认值 —— 通过调优直到偏差/方差达到你希望的水平。

**A2C：同步优势 actor-critic。** 跨 `N` 个并行环境收集 `T` 步。为每一步计算优势值。在合并后的批次上更新演员和评论家。重复。A3C 更简单、更易扩展的兄弟版本。

**A3C：异步优势 actor-critic。** Mnih 等人 (2016)。生成 `N` 个工作线程，每个线程运行一个环境。每个工作者在自己的 rollout 上本地计算梯度，然后异步地将它们应用到共享的参数服务器上。不需要经验回放缓冲区 —— 工作者通过运行不同的轨迹来去相关。A3C 证明了你可以在 CPU 上进行大规模训练。在 2026 年，基于 GPU 的 A2C（批量并行环境）占据主导地位，因为 GPU 需要大批量。

**组合损失函数。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失，价值回归，熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是经典的起始值。

## 构建

### 第 1 步：评论家

线性评论家 `V_φ(s) = w · features(s)`，通过 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格型环境中，评论家在几百个回合内收敛。在 Atari 上，将线性评论家替换为共享 CNN 主干 + 价值头部。

### 第 2 步：n 步优势

给定长度为 `T` 的 rollout 以及最终的 bootstrap 值 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论家的目标。`advantages` 是与 `∇ log π` 相乘的量。

### 第 3 步：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

同策略，每次更新一个 rollout，演员和评论家使用不同的学习率。

### 第 4 步：并行化（A3C vs A2C）

- **A3C：** 生成 `N` 个线程。每个线程运行自己的环境和自己的前向传播。定期将梯度更新推送到共享的主模型上。主模型不加锁 —— 竞争是可以接受的，它们只是增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` 批次，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认选择。

我们的示例代码为清晰起见是单线程的；改为批量 A2C 只需要三行 numpy 代码。

## 常见陷阱

- **评论家偏差先于演员梯度。** 如果评论家是随机的，它的基线没有信息量，你是在纯噪声上训练。在开启策略梯度之前，先预热评论家几百步，或者使用较慢的演员学习率。
- **优势值归一化。** 将每个批次的优势值归一化为零均值/单位标准差。极大地稳定训练，且几乎不增加成本。
- **共享主干。** 对于图像输入，使用共享的特征提取器给演员和评论家。分离的头部。共享的特征可以同时从两个损失中获益。
- **同策略约定。** A2C 的数据只用于一次更新。更多次会导致梯度有偏（重要性采样矫正是 PPO 添加的功能）。
- **熵崩溃。** 如果没有 `c_e > 0`，策略会在几百次更新后变得几乎确定，停止探索。
- **奖励尺度。** 优势值的大小取决于奖励的尺度。对奖励进行归一化（例如，除以运行标准差）可以在不同任务上得到一致的梯度大小。

## 应用

A2C/A3C 在 2026 年很少是最终选择，但它们是所有后续方法改进的架构基础：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 剪裁的重要性比率，用于多轮更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC (第 9 阶段 · 07) | 带有软价值评论家的离策略 A2C（下一课） |
| GRPO (第 9 阶段 · 12) | 没有评论家的 A2C —— 群体相对优势 |
| DPO | A2C 坍缩为偏好排序损失，无需采样 |
| AlphaStar / OpenAI Five | 带有联赛训练和模仿预训练的 A2C |

如果你在 2026 年的论文中看到“优势”，请想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上使用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。与第 6 课中的带有运行均值基线的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报与样本效率的关系图。对于这个任务，偏差/方差的甜蜜点在哪里？

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| 演员 (Actor) | "策略网络" | `π_θ(a|s)`，通过策略梯度更新。 |
| 评论家 (Critic) | "价值网络" | `V_φ(s)`，通过 MSE 回归到回报 / TD 目标来更新。 |
| 优势 (Advantage) | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计量。与 `∇ log π` 相乘。 |
| TD 残差 (TD residual) | "δ" | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权和，由 `λ` 参数化。 |
| A2C | "同步 actor-critic" | 跨环境批量化；每个 rollout 进行一次梯度步骤。 |
| A3C | "异步 actor-critic" | 工作线程向共享参数服务器推送梯度。原始论文；2026 年不太常见。 |
| Bootstrap | "在 horizon 处使用 V" | 截断 rollout，加上 `γ^n V(s_{t+n})` 来完成求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始的异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当评论家是神经网络时，与第 9 章关于函数逼近的内容配合阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带有 V-trace 离策略修正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 双时间尺度 actor-critic 分解的基础收敛结果。
