# 深度 Q 网络（DQN）

> 2013 年：Mnih 等人训练了一个 Q 学习网络处理原始像素，在七款 Atari 游戏上击败了所有经典强化学习智能体。2015 年：扩展到 49 款游戏，发表在《自然》杂志上，开启了深度强化学习时代。DQN 是 Q 学习加上三项技巧，使函数逼近变得稳定。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 3 · 03（反向传播），阶段 9 · 04（Q 学习，SARSA）  
**时间：** 约 75 分钟

## 问题

表格形式 Q 学习需要为每一对（状态，动作）存储一个独立的 Q 值。一副国际象棋棋盘约有 10⁴³ 个状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格形式 RL 在几千个状态时就会失效，更不用说数十亿了。

事后看来，解决方案显而易见：用神经网络 `Q(s, a; θ)` 替代 Q 表格。但这个“显而易见”花了几十年。朴素的 Q 学习函数逼近会在“致命三要素”——函数逼近 + 自举 + 离策略学习——下发散。Mnih 等人（2013, 2015）确定了三项稳定学习的工程技巧：

1. **经验回放（Experience replay）** 去相关转移样本。
2. **目标网络（Target network）** 冻结自举目标。
3. **奖励裁剪（Reward clipping）** 归一化梯度量级。

Atari 上的 DQN 是第一个用单一架构和单一超参数集从原始像素中解决数十个控制问题的方法。此后所有“深度 RL”方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都建立在这三个技巧的基础上。

## 概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、Bellman TD 损失](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每一步通过梯度下降更新。`θ^-` = 目标网络，每隔约 10,000 步从 `θ` 复制一次。`D` = 存储过去转移样本的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量约 `10⁶` 的环形缓冲区。每个训练步骤均匀随机采样一个迷你批次。这打破了时间相关性（连续几帧几乎相同），让网络可以多次从稀有的高奖励转移中学习，并去连续梯度更新之间的相关性。没有它，使用神经网络的在策略 TD 会在 Atari 上发散。

**目标网络。** 在 Bellman 方程两端使用同一个网络 `Q(·; θ)` 会使目标每一步都变化——好比“追逐自己的尾巴”。解决办法：保留一个拥有冻结权重的第二网络 `Q(·; θ^-)`。每 `C` 步执行一次 `θ → θ^-`。这使得回归目标在数千个梯度步骤内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一种更平滑的变体。

**奖励裁剪。** Atari 奖励量级从 1 到 1000 以上。将奖励裁剪到 `{-1, 0, +1}` 可以避免单个游戏主导梯度。当奖励量级重要时该方法不合适；但对于只关心符号的 Atari 来说效果很好。

**Double DQN。** Hasselt（2016）解决了最大化偏差：使用在线网络来*选择*动作，使用目标网络来*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用式替换，效果始终更好。默认使用它。

**其他改进（Rainbow, 2017）：** 优先级回放（优先采样高 TD 误差的转移）、对偶架构（分离 `V(s)` 和优势头）、噪声网络（学习式探索）、n 步回报、分布 Q（C51/QR-DQN）、多步自举。每个方法带来几个百分点的提升；收益大致可叠加。

## 动手构建

这里的代码仅用标准库且不依赖 numpy——我们使用手工编写的单隐层 MLP 在一个小型连续 GridWorld 上运行，因此每个训练步骤只需微秒级别。算法与大规模 Atari DQN 相同。

### 第一步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 容量约 50,000；我们的玩具环境 5,000 足矣。

### 第二步：小型 Q 网络（手工 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 第三步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

其形式与课程 04 中的 Q 学习相同，但有两点不同：（a）我们通过可微的 `Q(·; θ)` 进行反向传播，而不是索引一个表格；（b）目标使用 `Q(·; θ^-)`。

### 第四步：外层循环

每个 episode 中，基于 `Q(·; θ)` 执行 ε-贪婪策略，将转移样本压入缓冲区，采样一个迷你批次，执行一个梯度步骤，并定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的小型 GridWorld（16 维 one-hot 状态）上，智能体大约在 500 个 episode 内学会近似最优策略。在 Atari 上，将规模扩展到 2 亿帧，并添加一个 CNN 特征提取器。

## 陷阱

- **致命三要素。** 函数逼近 + 离策略 + 自举可能导致发散。DQN 通过目标网络 + 回放来缓解；不要移除其中任何一个。
- **探索。** ε 必须衰减，通常从前 10% 训练时间内的 1.0 降至 0.01。如果没有足够的早期探索，Q 网络会收敛到局部洼地。
- **过估计。** 对噪声 Q 取 `max` 会产生向上偏差。生产环境中始终使用 Double DQN。
- **奖励量级。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲区冷启动。** 在缓冲区中有几千个转移之前不要训练。对大约 20 个样本的早期梯度会导致过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 目标过时。Atari DQN 使用 10,000 个环境步。经验法则：每训练周期的约 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧以使状态满足马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 如何使用

到 2026 年，DQN 已经很少是 SOTA，但它仍是离策略算法的参考基准：

| 任务 | 首选方法 | 为什么不选 DQN？ |
|------|----------|------------------|
| 离散动作类 Atari | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（阶段 9 · 07） | DQN 没有策略网络。 |
| 在策略 / 高通量 | PPO（阶段 9 · 08） | 无回放缓冲区；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可以，取决于细节。 |
| LLM RL | PPO / GRPO | 序列级而非步骤级；损失函数不同。 |

这些经验依然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区和所有离线 RL 方法中。奖励裁剪以优势归一化的形式活在 PPO 中。该架构就是蓝图。

## 交付

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个 episode 的回报曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用目标网络（在 Bellman 目标的两侧使用在线网络）。衡量训练不稳定程度——回报是否会振荡或发散？
3. **困难。** 添加 Double DQN：使用在线网络选择 `argmax a'`，目标网络评估。在带有噪声奖励的 GridWorld 上，经过 1,000 个 episode 后，比较使用 Double DQN 和不使用时 `Q(s_0, best_a)` 相对于真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| DQN | “深度 Q 学习” | 使用神经 Q 函数、回放缓冲区和目标网络的 Q 学习。 |
| Experience replay | “打乱的转移样本” | 每个梯度步骤均匀采样环形缓冲区；去相关数据。 |
| Target network | “冻结的自举” | 用于 Bellman 目标的 Q 函数定期拷贝；稳定训练。 |
| Deadly triad | “RL 为何发散” | 函数逼近 + 自举 + 离策略 = 无收敛保证。 |
| Double DQN | “修复最大化偏差” | 在线网络选择动作，目标网络评估它。 |
| Dueling DQN | “V 和 A 头” | 将 Q 分解为 Q = V + A - mean(A)；相同输出，更好的梯度流。 |
| Rainbow | “所有技巧” | DDQN + PER + dueling + n-step + noisy + distributional 于一体。 |
| PER | “优先级回放” | 按 TD 误差量级比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 2013 年 NeurIPS 研讨会论文，开启了深度强化学习。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) —— 《自然》论文，49 款游戏的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 技巧叠加论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 清晰的现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 关于“致命三要素”（函数逼近 + 自举 + 离策略）的教科书处理，DQN 的目标网络和回放缓冲区正是为解决它而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 参考单文件 DQN，用于消融研究；适合与本课的手写版本一起阅读。
