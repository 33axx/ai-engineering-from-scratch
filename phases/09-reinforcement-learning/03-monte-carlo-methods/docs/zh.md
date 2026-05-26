# 蒙特卡洛方法 — 从完整回合中学习

> 动态规划需要模型。蒙特卡洛只需要回合。运行策略，观察回报，然后取平均。强化学习中最简单的想法——也是解锁后续一切的关键。

**类型：** 构建
**语言：** Python
**前置知识：** 第九阶段 · 01（MDP），第九阶段 · 02（动态规划）
**时长：** ~75 分钟

## 问题

动态规划虽然优雅，但它假设你可以针对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有任何事物能如此运作。机器人无法解析地计算关节力矩后摄像头像素的分布。定价算法无法对所有可能的顾客反应进行积分。LLM 无法枚举一个 token 之后的所有可能延续。

你需要一种只需要能够从环境中*采样*的方法。运行策略，获取一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上意义重大：我们从*已知模型 + 精确备份*转向*采样 rollout + 平均回报*。方差增加了，但适用性爆炸了。本课之后的所有强化学习算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时会在其之上叠加自举。

## 概念

![蒙特卡洛：rollout、计算回报、取平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一句话概括：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问状态 `s` 后观察到的回报。

**首次访问 vs 每次访问 MC。** 给定一条多次访问状态 `s` 的回合，首次访问 MC 只计算第一次访问的回报；每次访问 MC 则计入所有访问。两者在极限情况下都是无偏的。首次访问更容易分析（iid 样本）。每次访问每回合使用更多数据，实际中通常收敛更快。

**增量平均。** 不必存储所有回报，而是更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新组织为：`V_new = V_old + α · (目标 - V_old)`，其中 `α = 1/n`。将 `1/n` 替换为恒定步长 `α ∈ (0, 1)`，就得到了一个非平稳的 MC 估计器，能够跟踪 `π` 的变化。这一步正是从 MC 跳到 TD 再到所有现代 RL 算法的关键。

**探索现在成了问题。** DP 通过枚举触及了每个状态。MC 只看到策略访问过的状态。如果 `π` 是确定性的，状态空间的整个区域从未被采样，其价值估计将永远为零。按历史顺序，有三种修复方法：

1. **探索性初始化。** 从随机的 (s, a) 对开始每个回合。保证覆盖性；但在实践中不现实（你无法将机器人“重置”到任意状态）。
2. **ε-贪心。** 相对于当前 Q 采取贪心动作，但以概率 `ε` 随机选择一个动作。所有状态-动作对在渐近意义上都会被采样。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。方差高，但它是通向 DQN 等回放缓冲区方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，获得一条回合。
2. 根据观察到的回报更新 `Q(s, a)`。
3. 使 `π` 相对于 `Q` 成为 ε-贪心策略。
4. 重复。

在温和条件下（每个状态-动作对无限频繁地被访问，`α` 满足 Robbins-Monro），以概率 1 收敛到 `Q*` 和 `π*`。

## 构建

### 第一步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需要 `env.reset()` 和 `env.step(s, a)`。接口与 gym 环境相同，但经过了精简。

### 第二步：计算回报（反向遍历）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重新求和。

### 第三步：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：标记首次访问的状态，递增计数，更新运行均值。

### 第四步：ε-贪心 MC 控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第五步：与 DP 金标准比较

当回合数 → ∞ 时，你的 MC 对 `V^π` 的估计应与第二课中的 DP 结果一致。实际中：在 4×4 GridWorld 上运行 50,000 个回合，精度可达 DP 答案的 `~0.1` 以内。

## 陷阱

- **无限回合。** MC 要求回合*必须终止*。如果你的策略可能永远循环，请设置 `max_steps` 上限，并将此上限视为隐性失败。在随机策略下，GridWorld 经常会超时——这是正常的，只需确保正确计数。
- **方差。** MC 使用完整回报。在长回合中，方差巨大——末尾一次不幸的奖励会以相同幅度改变 `V(s_0)`。TD 方法（第四课）通过自举来降低这种方差。
- **状态覆盖。** 带有平局（ties）的新 Q 值的贪心 MC 只会尝试一个动作。你*必须*进行探索（ε-贪心、探索性初始化、UCB）。
- **非平稳策略。** 如果 `π` 发生变化（如 MC 控制中），旧的回报来自不同的策略。恒定 α 的 MC 可以处理这种情况；样本平均 MC 不能。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在一条轨迹上不断相乘。方差随步长呈爆炸性增长。可以通过每决策加权 IS 或改为 TD 来控制。

## 使用

2026 年蒙特卡洛方法的使用场景：

| 使用场景 | 为何用 MC |
|----------|-----------|
| 短回合游戏（二十一点、扑克） | 回合自然终止；回报清晰。 |
| 已记录策略的离线评估 | 对存储的轨迹计算平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶节点进行的 MC rollout 指导选择。 |
| LLM RL 评估 | 计算给定策略下采样完成结果的平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用了 MC 的 `G_t`。 |
| 强化学习教学 | 最简单的可实际工作的算法——剥离自举以看清核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n` 步回报或 GAE 在纯 MC（完整回报）和纯 TD（一步自举）之间进行插值。两个端点都是同一估计器的实例。

## 交付

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。绘制 `V(0,0)` 随回合数的变化曲线，并与 DP 答案进行比较。
2. **中等。** 实现 ε-贪心 MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个回合后的平均回报。曲线看起来如何？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的*离策略* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS 与每决策 IS 与加权 IS。哪个方差最低？

## 关键术语

| 术语 | 人们常说的话 | 实际含义 |
|------|-------------|---------|
| 蒙特卡洛 | “随机采样” | 通过对分布的 iid 样本取平均来估计期望。 |
| 回报 `G_t` | “未来奖励” | 从步骤 `t` 到回合结束的折扣奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | “每个状态只计一次” | 回合中仅第一次访问对价值估计有贡献。 |
| 每次访问 MC | “使用所有访问” | 每次访问都有贡献；轻微有偏但样本效率更高。 |
| ε-贪心 | “探索噪声” | 以概率 `1-ε` 选择贪心动作，以概率 `ε` 选择随机动作。 |
| 重要性采样 | “对从错误分布采样的校正” | 通过 `π(a|s)/μ(a|s)` 的乘积对回报加权，以从 `μ` 数据估计 `V^π`。 |
| 在策略 | “从我自己的数据中学习” | 目标策略 = 行为策略。原始 MC、PPO、SARSA。 |
| 离策略 | “从别人的数据中学习” | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 经典处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) —— 首次访问 vs 每次访问分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) —— 离策略 MC 与方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) —— 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) —— 首个大规模实证证明 MC/TD 自对弈可达到超人类水平；是该阶段后半部分所有课程的概念先驱。
