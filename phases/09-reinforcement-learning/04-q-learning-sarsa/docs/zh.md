# 时序差分 — Q-Learning 与 SARSA

> 蒙特卡洛要等到情节结束才更新。时序差分（TD）每步之后都通过自举下一个值估计来进行更新。Q-learning 是离策略且乐观的；SARSA 是在策略且谨慎的。两者都只是一行代码。它们也是本阶段所有深度强化学习方法的基础。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）、阶段 9 · 03（蒙特卡洛）  
**时间：** 约 75 分钟

## 问题

蒙特卡洛方法有效，但它有两个昂贵的需求。它需要能终止的情节，并且只在最终回报到来时更新。如果你的情节有 1000 步，MC 必须等待 1000 步才能更新任何东西。它是高方差、低偏差的，在实际应用中很慢。

动态规划有相反的特性——零方差的自举回溯——但需要已知的模型。

时序差分（TD）学习则折中了两者。从一个单步转移 `(s, a, r, s')` 出发，构造一个一步目标 `r + γ V(s')` 并将 `V(s)` 推向它。不需要模型，不需要完整的情节。由于右侧使用了近似的 `V` 而产生偏差，但方差远低于 MC，并且从第一步开始就在线更新。

这是现代所有强化学习——DQN、A2C、PPO、SAC——的关键转折点。阶段 9 的剩余部分将建立在你在本课中编写的一步 TD 更新之上，加上层层函数逼近和技巧。

## 概念

![Q-learning 与 SARSA 对比：离策略的 max 与在策略的 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`）并且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪心*策略，无论智能体实际采取什么动作。这种解耦使得 Q-learning 能够学习 `Q*`，同时智能体通过 ε-贪心进行探索。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而不是贪心的 `argmax`。收敛到当前运行的 ε-贪心 `π` 的 `Q^π`，在极限 `ε → 0` 下变为 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务（掉下悬崖奖励 -100）中，Q-learning 学会了沿悬崖边缘的最优路径，但在探索期间偶尔会受到惩罚。SARSA 学会了离悬崖一步之遥的更安全路径，因为它将探索噪声纳入了 Q 值。随着训练进行，当 `ε → 0` 时两者都达到最优。在实际部署中，当探索确实发生时，SARSA 的行为更加保守。

**期望 SARSA。** 用 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

方差低于 SARSA（无需对 `a'` 采样），相同的在策略目标。通常是现代教科书中的默认选择。

**n 步 TD 与 TD(λ)。** 通过在自举之前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 使用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行平均。大多数深度强化学习使用介于 3 到 20 之间的 `n`。

## 构建它

### 步骤 1：ε-贪心策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。与 Q-learning 的*唯一*区别在于目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是在策略与离策略之间的区别。

### 步骤 3：学习曲线

跟踪每 100 个情节的平均回报。在简单的确定性 GridWorld 上，Q-learning 收敛更快；在悬崖行走任务中，SARSA 更加保守。在 `code/main.py` 中的 4×4 GridWorld 上，两者在 `α=0.1, ε=0.1` 下约 2000 个情节后都接近最优。

### 步骤 4：与 DP 真实值比较

运行值迭代（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型 TD 智能体在 4×4 GridWorld 上经过 10000 个情节后，误差会落在 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（对于负奖励任务设 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪心策略。
- **α 调度。** 对于非平稳问题，常数 `α` 是可接受的。衰减 `α_n = 1/n` 在理论上保证收敛，但在实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高值开始（`ε=1.0`），衰减到 `ε=0.05`。“GLIE”（无限探索下的极限贪心）是收敛条件。
- **Q-learning 中的最大化偏差。** 当 `Q` 有噪声时，`max` 算子存在向上偏差。导致过估——Hasselt 的双 Q-learning（第 05 课的 DDQN 使用）通过两个 Q 表解决了这个问题。
- **不终止的情节。** TD 可以在没有终止状态的情况下学习，但你需要要么限制步数，要么在截断时正确处理自举。标准做法：将截断视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，请使用可哈希的键（元组，而不是列表；对浮点数四舍五入后的元组，不要用原始值）。

## 使用它

2026 年的 TD 应用场景：

| 任务 | 方法 | 理由 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键场景 | SARSA / 期望 SARSA | 探索期间行为保守。 |
| 高维状态 | DQN（阶段 9 · 05） | 带经验回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（阶段 9 · 07） | Q 网络上的 TD 更新；策略网络输出动作。 |
| LLM 强化学习（基于奖励模型） | PPO / GRPO（阶段 9 · 08, 12） | 演员-评论家架构，通过 GAE 计算 TD 风格的优势。 |
| 离线强化学习 | CQL / IQL（阶段 9 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的 90% 的“强化学习”都是 Q-learning 或 SARSA 的某种变体。在深入学习之前，请将表格型更新理解透彻。

## 交付它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2000 个情节的学习曲线（每 100 个情节的平均回报）。谁收敛更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图显示它们各自采取的路径。哪个更靠近悬崖？
3. **困难。** 实现双 Q-learning。在一个带噪声奖励的 GridWorld（每步奖励加入高斯噪声 σ=5）上，表明 Q-learning 对 `V*(0,0)` 的过估幅度显著，而双 Q-learning 则不会。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| TD 误差 | “更新信号” | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | “一步 TD” | 每一步转移后使用仅下一个状态的估计进行更新。 |
| Q-learning | “离策略强化学习 101” | 使用 `max` 对下一状态动作进行 TD 更新；无论行为策略如何，学习 `Q*`。 |
| SARSA | “在策略 Q-learning” | 使用实际下一个动作的 TD 更新；学习当前 ε-贪心 `π` 的 `Q^π`。 |
| 期望 SARSA | “低方差 SARSA” | 用其在 `π` 下的期望替换采样的 `a'`。 |
| GLIE | “正确的探索调度” | 无限探索下的极限贪心；Q-learning 收敛所需。 |
| 自举 | “在目标中使用当前估计” | 区分 TD 与 MC 的特征。偏差的来源，但大幅降低方差。 |
| 最大化偏差 | “Q-learning 过估” | 在噪声估计上取 `max` 会出现向上偏差；由双 Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文与收敛证明。
- [Sutton & Barto (2018). 第 6 章 — 时序差分学习](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望 SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 修复最大化偏差。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出 SARSA（当时称为“modified connectionist Q-learning”）的论文。
- [Sutton & Barto (2018). 第 7 章 — n 步自举](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹再到 PPO 中 GAE 的路径。
