# MDP（马尔可夫决策过程）：状态、动作与奖励

> 马尔可夫决策过程由五部分组成：状态、动作、转移、奖励、折扣因子。强化学习中的一切——Q学习、PPO、DPO、GRPO——都在优化这个结构。掌握它，就能免费读懂强化学习剩下的部分。

**类型：** 学习
**语言：** Python
**前置知识：** 第一阶段 · 06（概率与分布），第二阶段 · 01（机器学习分类）
**时间：** 约45分钟

## 问题

你在编写一个国际象棋机器人、库存规划器、交易智能体，或者是训练推理模型的PPO循环。四个不同的领域，一个令人惊讶的事实：它们都归结为同一个数学对象。

监督学习给你 `(x, y)` 对，并让你拟合一个函数。强化学习不给你标签——只有一连串的状态、你采取的动作以及一个标量奖励。这步棋赢了比赛吗？补货决策省了钱吗？交易盈利了吗？LLM刚刚生成的token是否让裁判给出了更高的奖励？

在形式化之前，你无法从这个序列中学习。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“那有多好”——每一个都必须变成可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段的所有强化学习算法，包括最后的RLHF和GRPO循环，都在优化这个结构。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣因子](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体决策所需的一切。在网格世界中，是格子。在国际象棋中，是棋盘。在LLM中，是上下文窗口加上任何记忆。
- **动作** `A`。选择项。上/下/左/右移动。走一步棋。生成一个token。
- **转移** `P(s' | s, a)`。给定状态`s`和动作`a`，下一个状态的概率分布。在国际象棋中是确定性的，在库存中是随机的，在LLM解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。赢=+1，输=-1。收入减去成本。GRPO中的对数似然比项。
- **折扣因子** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的重要性。`γ = 0.99` 相当于约100步的视界；`γ = 0.9` 相当于约10步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖于当前状态。如果不满足，说明状态表示不完整——这不是方法的失败，而是状态的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是折扣后的未来奖励总和。价值 `V^π(s) = E[G_t | s_t = s]` 是从状态`s`开始、遵循策略`π`的期望回报。Q值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个强化学习算法都会估计这两个之一，然后据此改进`π`。

**贝尔曼方程。** 本阶段所有算法都使用的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为“这一步的奖励”加上“到达状态的折扣价值”。递归。第九阶段的所有算法要么迭代这个方程直至收敛（动态规划），要么从中采样（蒙特卡洛），要么单步自举（时序差分）。

## 动手实践

### 第一步：一个小型确定性MDP

一个4×4的网格世界。智能体从左上角开始，终点在右下角，每步奖励-1，动作集合`{上, 下, 左, 右}`。详见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

就五行。这就是整个环境。确定性转移，恒定的步数惩罚，吸收性的终止状态。

### 第二步：执行一个策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略1000次。这个4×4棋盘的平均回报大约在-60到-80之间。最优回报是-6（直线向下向右的路径）。缩小这个差距就是第九阶段的全部内容。

### 第三步：通过贝尔曼方程精确计算`V^π`

对于小型MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到数值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是Sutton & Barto书中的第一个算法，也是后续所有强化学习方法的理论基础。

### 第四步：`γ`是一个有物理意义的超参数

有效视界大致为 `1 / (1 - γ)`。`γ = 0.9` → 10步。`γ = 0.99` → 100步。`γ = 0.999` → 1000步。

设置过低，智能体目光短浅。设置过高，信用分配变得嘈杂，因为许多早期步骤共同对遥远的未来奖励负责。LLM的RLHF通常使用`γ = 1`，因为回合很短且有界。控制任务使用`0.95–0.99`。长视界策略游戏使用`0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次观测才能做决策，那么“状态”就不只是当前观测。解决方法：堆叠帧（Atari上的DQN堆叠4帧）或使用循环状态（对观测使用LSTM/GRU）。
- **稀疏奖励。** 仅赢才有的奖励使得在大型状态空间中学习几乎不可能。需要设计奖励形状（中间信号）或通过模仿进行自举（第九阶段 · 09）。
- **奖励破解。** 优化代理奖励常常导致病态行为。OpenAI的赛艇智能体原地打转收集动力增强，而不是完成比赛。始终根据目标结果定义奖励，而不是代理奖励。
- **折扣因子设置不当。** 在无限视界任务中`γ = 1`会导致所有价值无限大。始终通过有限视界或`γ < 1`来限制。
- **奖励尺度。** 奖励集{+100, -100}和{+1, -1}给出相同的最优策略，但梯度幅度差异巨大。在输入PPO/DQN之前，将其归一化到`[-1, 1]`附近。

## 实际应用

2026年的技术栈在接触代码之前，会将每个强化学习流程简化为MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度 + 速度 | 连续力矩 | 任务特定的形状奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法棋步 | 赢=+1 / 输=-1 | 1.0（有限） |
| 库存 / 定价 | 库存 + 需求 | 订购数量 | 收入 - 成本 | 0.95 |
| LLM的RLHF | 上下文token | 下一个token | 结束时的奖励模型分数 | 1.0（回合约200个token） |
| 推理的GRPO | 提示 + 部分回答 | 下一个token | 结束时的验证器0/1 | 1.0 |

在编写任何训练循环之前，先写出这五个元组。大多数“强化学习不工作”的bug报告都追溯到纸上就写错的MDP公式。

## 交付

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现4×4网格世界和随机策略的rollout。运行10,000个回合。报告回报的均值和标准差。与最优回报(-6)进行比较。
2. **中等。** 对均匀随机策略，使用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。对每个γ，以4×4网格打印 `V`。解释为什么靠近终止状态的状态值随着γ增大而增长更快。
3. **困难。** 使网格世界变得随机：每个动作有概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 变好了还是变坏了？为什么？

## 关键术语

| 术语 | 常见的说法 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | “强化学习设置” | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| 状态 | “智能体看到的东西” | 在所选策略类下，未来动态的充分统计量。 |
| 策略 | “智能体的行为” | 条件分布 `π(a | s)` 或确定性映射 `s → a`。 |
| 回报 | “总奖励” | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| 价值 | “一个状态的好坏” | 在`π`下从`s`开始的期望回报。 |
| Q值 | “一个动作的好坏” | 在`π`下从`s`开始、第一个动作为`a`的期望回报。 |
| 贝尔曼方程 | “动态规划递归” | 将价值/Q分解为单步奖励加上折扣后继价值的固定点分解。 |
| 折扣因子 `γ` | “未来 vs 当前” | 对未来奖励的几何加权；有效视界 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第3章涵盖MDP和贝尔曼方程；第1章阐述了奖励假说，这是后续所有课程的基础。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度强化学习角度简明的MDP入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运筹学中关于MDP和精确解法的参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将MDP作为动态规划特化的最清晰推导。
