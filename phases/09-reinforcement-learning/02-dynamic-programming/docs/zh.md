# 动态规划——策略迭代与价值迭代

> 动态规划就是开了上帝视角的强化学习。你已经知道转移函数和奖励函数；只需反复迭代贝尔曼方程，直到 `V` 或 `π` 不再变化。它是所有基于采样的方法试图逼近的标杆。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段9 · 01（MDP）
**时间：** 约75分钟

## 问题描述

你拥有一个已知模型 MDP：对于任意状态-动作对，你可以查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理员知道需求分布。棋盘游戏具有确定性转移。网格世界就是四行 Python 代码。你拥有一个**模型**。

无模型强化学习（Q-learning，PPO，REINFORCE）是为没有模型的情况发明的——你只能从环境中采样。但当你确实拥有模型时，存在更快、更好的方法：动态规划。贝尔曼在 1957 年设计了它们。它们至今仍定义着正确性：当人们说“这个 MDP 的最优策略”时，指的就是动态规划会返回的那个策略。

你在 2026 年需要它们，原因有三。第一，强化学习研究中的每一个表格型环境（GridWorld、FrozenLake、CliffWalking）都是用动态规划求解以产生黄金标准策略。第二，精确的值可以让你**调试**采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与动态规划答案相差 30%，那么你的 Q-learning 有 bug。第三，现代离线强化学习和规划方法（MCTS，AlphaZero 的搜索，阶段9·10中的基于模型强化学习）都在一个学习到的或给定的模型上迭代贝尔曼备份。

## 概念

![策略迭代与价值迭代并排展示](../assets/dp.svg)

**两种算法，都是对贝尔曼方程的不动点迭代。**

**策略迭代。** 交替执行两个步骤，直到策略不再变化。

1. **评估：** 给定策略 `π`，重复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛，计算 `V^π`。
2. **改进：** 给定 `V^π`，使 `π` 相对于 `V^π` 变得贪心：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性得到保证，因为 (a) 每一步改进要么保持 `π` 不变，要么严格增加某些状态的 `V^π`，(b) 确定性策略的空间是有限的。即使在大的状态空间中，通常也只需约 5~20 次外部迭代。

**价值迭代。** 将评估和改进合并为一次扫描。应用贝尔曼*最优*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪心动作提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一的框架。值函数和策略锁定在一个双向改进循环中；任何驱动两者走向相互一致的方法（异步价值迭代、修正策略迭代、Q-learning、Actor-Critic、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子是上确界范数下的 `γ`-压缩：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩意味着唯一不动点和几何收敛。失去 `γ < 1` 你就失去了这个保证——你需要一个有限时域或一个吸收终止状态。

## 构建

### 步骤1：构建 GridWorld MDP 模型

使用与课程01相同的 4×4 GridWorld。我们添加一个随机变体：以概率 `0.1`，智能体滑到随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是完整的模型。

### 步骤2：策略评估

给定一个策略 `π(s) = {action: prob}`，迭代贝尔曼方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤3：策略改进

用相对于 `V` 的贪心策略替换 `π`。如果 `π` 没有变化，则返回——我们已经达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤4：将它们缝合起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上典型收敛：4–6 次外部迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### 步骤5：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行数。

## 常见陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼方程，它仍然会选取一个不改变任何东西的“最佳动作”。使用 `if s == terminal: V[s] = 0` 进行保护。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证基于上确界范数。
- **原地更新 vs 同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比使用单独的 `V_new` 字典（Jacobi）收敛更快。生产代码使用原地更新。
- **策略相等性。** 如果两个动作具有相等的 Q 值，`argmax` 可能每次迭代以不同方式打破平局，导致“策略稳定”检查震荡。使用稳定的平局打破方式（固定顺序中的第一个动作）。
- **状态空间爆炸。** 动态规划是每轮 `O(|S| · |A|)` 复杂度。最多适用于 ~10⁷ 个状态。超出此范围，你需要函数近似（阶段9·05 及以后）。

## 使用场景

在 2026 年，动态规划是正确性基线和规划器的内循环：

| 使用场景 | 方法 |
|----------|------|
| 精确求解一个小型表格型 MDP | 价值迭代（更简单）或策略迭代（更少外部步骤） |
| 验证 Q-learning / PPO 实现 | 在玩具环境中与 DP 最优的 V* 比较 |
| 基于模型的强化学习（阶段9·10） | 在学习到的转移模型上进行贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线强化学习（CQL，IQL） | 保守 Q 迭代——对 OOD 动作施加惩罚的 DP |

每当有人提到“最优值函数”时，他们指的就是“DP 不动点”。当你论文中看到 `V*` 或 `Q*` 时，脑中应浮现这个循环。

## 交付

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上使用 `γ ∈ {0.9, 0.99}` 运行价值迭代。需要多少轮直到 `max |ΔV| < 1e-6`？将 `V*` 打印为 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑倒概率 `0.1`）上比较策略迭代与价值迭代。统计：轮数、墙钟时间、最终 `V*(0,0)`。哪个在迭代次数上收敛更快？在墙钟时间上呢？
3. **困难。** 构建修正策略迭代：在评估步骤中，只运行 `k` 轮而不是收敛。针对 `k ∈ {1, 2, 5, 10, 50}` 绘制 `V*(0,0)` 误差与 `k` 的关系图。这条曲线告诉了你关于评估/改进权衡的什么信息？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 策略迭代 | “DP 算法” | 交替进行评估（`V^π`）和改进（相对于 `V^π` 的贪心 `π`），直到策略不再变化。 |
| 价值迭代 | “更快的 DP” | 在一次扫描中应用贝尔曼最优备份；以几何速率收敛到 `V*`。 |
| 贝尔曼算子 | “递归” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；上确界范数下的 `γ`-压缩。 |
| 压缩 | “为什么 DP 收敛” | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 都有唯一不动点。 |
| GPI | “一切都是 DP” | 广义策略迭代：任何驱动 `V` 和 `π` 走向相互一致的方法。 |
| 同步更新 | “Jacobi 风格” | 在整轮扫描中使用旧的 `V`；分析简洁但较慢。 |
| 原地更新 | “Gauss-Seidel 风格” | 边更新边使用 `V`；实践中收敛更快。 |

## 扩展阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和价值迭代的经典阐述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 对压缩映射论证的严谨处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修正策略迭代及其收敛性分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 策略迭代的原始论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP / 深度强化学习的桥梁，后续每节课都在使用。
