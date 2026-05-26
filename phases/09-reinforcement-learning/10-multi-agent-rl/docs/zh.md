# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。如果将两个学习智能体放入同一世界，这一假设便被打破：每个智能体都是对方环境的一部分，且两者都在不断变化。多智能体强化学习是在马尔可夫假设不再成立时，使学习收敛的一系列技巧。

**类型：** 构建  
**语言：** Python  
**先修要求：** Phase 9 · 04（Q 学习），Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）  
**时间：** ~45 分钟

## 问题

机器人学习在房间内导航是单智能体强化学习问题。一支足球队则不是。AlphaStar 与《星际争霸》对手对战不是。一个由竞价智能体构成的市场不是。两辆车在四向停车标志处协商不是。多对多的现实世界问题都不是。

在所有多智能体环境中，从任意一个智能体的视角看，其他智能体*就是*环境的一部分。当它们学习并改变行为时，环境变得非平稳。马尔可夫性质——“下一状态仅取决于当前状态和我的动作”——被违反，因为下一状态还取决于*其他*智能体的选择，而它们的策略是移动靶。

这破坏了表格型收敛证明（Q 学习的保证依赖于平稳环境）。它也破坏了天真的深度强化学习：智能体相互追逐，陷入循环，永远无法收敛到稳定策略。你需要多智能体专用技术：集中训练/分散执行、反事实基线、联赛对弈、自我对弈。

2026 年的应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体大语言模型系统（Phase 16），以及任何包含多个智能玩家的游戏。

## 概念

![四种多智能体强化学习范式：独立、集中评论家、自我对弈、联赛对弈](../assets/marl.svg)

**形式：马尔可夫博弈。** MDP 的泛化：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身策略 `π_i` 下最大化自己的回报。若奖励相同，则为**完全合作**。若为零和，则为**对抗**。若混合，则为**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于 `π_{-i}`，而后者正在变化。
- **信用分配。** 对于共享奖励，究竟是哪个智能体导致了它？
- **探索协调。** 智能体必须探索互补的策略，而不是冗余地在同一状态上探索。
- **可扩展性。** 联合动作空间随 `n` 呈指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态不可见。

**四种主导范式：**

**1. 独立 Q 学习 / 独立 PPO（IQL，IPPO）。** 每个智能体独立学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是当经验回放充当对对手建模的平滑技巧时）。理论收敛性：无。实践中：对于松散耦合的任务表现良好，对于紧密耦合的任务表现糟糕。

**2. 集中训练，分散执行（CTDE）。** 最常用的现代范式。每个智能体拥有自己的*策略* `π_i`，该策略基于局部观测 `o_i` 进行条件化——这在部署时是标准的分散执行。在*训练*期间，一个集中式评论家 `Q(s, a_1, …, a_n)` 基于完整的全局状态和联合动作进行条件化。示例：
- **MADDPG**（Lowe 等人，2017）：带每个智能体集中式评论家的 DDPG。
- **COMA**（Foerster 等人，2017）：反事实基线——问“如果我选择了动作 `a'`，我的奖励会是多少？”——从而隔离我的贡献。
- **MAPPO** / **带共享评论家的 IPPO**（Yu 等人，2022）：带集中式价值函数的 PPO。2026 年在合作式多智能体强化学习中占主导。
- **QMIX**（Rashid 等人，2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，采用单调混合。

**3. 自我对弈。** 同一个智能体的两个副本相互对弈。对手的策略*就是*我自己过去的快照策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和博弈；训练信号是对称的。

**4. 联赛对弈。** 自我对弈对一般和/对抗环境的扩展：保留一个包含过去和当前策略的种群，从联赛中采样一个对手，针对它进行训练。加入“利用者”（专门针对当前最优策略）和“主要利用者”（专门针对利用者）。AlphaStar（《星际争霸 II》）。当博弈存在“石头剪刀布”类的策略循环时，需要这种范式。

**通信。** 允许智能体之间发送可学习的消息 `m_i`。在合作场景中有效。Foerster 等人（2016）证明，可微分的智能体间通信可以端到端训练。当今基于大语言模型的多智能体系统（Phase 16）本质上是用自然语言进行通信。

## 动手构建

本课使用一个 6×6 的网格世界，其中包含两个合作智能体。它们从对角角落出发，必须到达同一个共享目标。共享奖励：若有任一智能体仍在移动，每步奖励 `-1`；当两者都到达时，奖励 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间为 `|A|² = 16`。全局状态为两个位置。

### 步骤 2：独立 Q 学习

每个智能体维护自己的 Q 表，以联合状态为键。每一步：两者都选取 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

该方法在此任务上有效，因为奖励密集且对齐。但在紧密耦合的任务（例如需要一个智能体*等待*另一个）上会失败。

### 步骤 3：集中式 Q 与分解值更新

使用一个联合动作上的 Q `Q(s, a_1, a_2)`。用共享奖励进行更新。在执行时通过边缘化实现分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。以指数级联合动作空间换取一个*正确的*全局视角。

### 步骤 4：简单自我对弈（对抗性 2 智能体）

同一智能体，两个角色。智能体 A 与智能体 B 对练；经过 `K` 个回合后，将 A 的权重复制到 B。对称训练，进度一致。这是 AlphaZero 配方的缩微版。

## 常见陷阱

- **非平稳经验回放。** 对于独立智能体，经验回放比单智能体更糟，因为旧转移是由现已过时的对手产生的。修复：重新标记或按新近程度加权。
- **信用分配模糊性。** 经过长回合后共享奖励；无法清楚指出哪个智能体做了贡献。修复：反事实基线（COMA），或每个智能体单独的奖励塑造。
- **策略漂移 / 追逐。** 每个智能体的最佳响应随着其他智能体的更新而改变。修复：集中式评论家、缓慢的学习率，或每次冻结一个智能体。
- **通过协调进行奖励黑客攻击。** 智能体找到设计者未预料到的协调性漏洞。拍卖智能体收敛到出价为零。修复：精心设计奖励、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或基于角色的条件化。
- **联赛循环。** 纯自我对弈可能陷入主导循环。修复：包含多样化对手的联赛对弈。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。通过函数近似来缓解；使用因子化动作空间（每个策略输出头对应一个智能体）。

## 应用场景

2026 年多智能体强化学习应用图谱：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散执行者。 |
| 双人博弈（国际象棋、围棋、扑克） | 自我对弈 + MCTS（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | 联赛对弈 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分可观测；变规模团队。 |
| 拍卖市场 | 博弈论均衡 + 强化学习 | 平均场强化学习（当 `n` → ∞）。 |
| 基于大语言模型的多智能体系统（Phase 16） | 自然语言通信 + 角色条件化 | 在智能体规划层上的强化学习循环。 |

2026 年，多智能体强化学习最大的增长领域是基于大语言模型的：语言模型智能体群进行协商、辩论、构建软件。强化学习以对*轨迹级*输出（而非 token 级）进行偏好优化的形式出现（Phase 16 · 03）。

## 输出完成

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体合作网格世界上训练独立 Q 学习。平均回报达到 > 0 需要多少个回合？绘制联合学习曲线。
2. **中等。** 添加一个“协调”任务：只有当两个智能体在同一回合同时踏上目标时，目标才达成。独立 Q 学习还能收敛吗？哪里出了问题？
3. **困难。** 实现一个集中式评论家，用于 MAPPO 风格的训练，并与独立 PPO 在协调任务上进行收敛速度比较。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | “集中训练，分散执行” | 训练时使用联合评论家；每个智能体的策略只使用局部观测。 |
| IPPO | “独立 PPO” | 每个智能体独立运行 PPO。简单基线；常常被低估。 |
| MAPPO | “多智能体 PPO” | 带集中式价值函数的 PPO，该函数以全局状态为条件。 |
| QMIX | “单调值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散式 argmax。 |
| COMA | “反事实多智能体” | 优势 = 我的 Q 减去对我的动作边缘化后的期望 Q。 |
| 自我对弈 | “智能体与过去的自己” | 单个智能体，两个角色；零和博弈的标准做法。 |
| 联赛对弈 | “种群训练” | 缓存过去的策略，从池中采样对手；处理策略循环。 |

## 拓展阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) —— 采用集中式评论家的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) —— 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) —— 带单调性的值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) —— PPO 在 MARL 中出奇地强大。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) —— 大规模联赛对弈。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) —— 零和博弈中的纯自我对弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) —— 包含教科书对多智能体场景以及 CTDE 旨在解决的非平稳性问题的简短讨论。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) —— 涵盖合作、竞争和混合式 MARL 并附带收敛结果的综述。
