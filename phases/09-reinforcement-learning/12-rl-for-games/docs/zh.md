# 游戏强化学习 —— AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 凭借纯 TD 算法在西洋双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始称霸国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明相同的配方（用 GRPO 取代 PPO）在推理任务上同样有效。游戏是推动这一阶段每一项突破的基准。

**类型：** 构建  
**语言：** Python  
**先修：** 阶段 9 · 05（DQN）、阶段 9 · 08（PPO）、阶段 9 · 09（RLHF）、阶段 9 · 10（MARL）  
**时长：** ~120 分钟

## 问题

游戏拥有强化学习所需的一切。清晰的奖励（赢/输）。无限的回合（自我对弈重置）。完美的仿真（游戏本身就是仿真器）。离散或小型连续动作空间。对抗性鲁棒性所必需的多智能体结构。

而游戏正是每一个重大强化学习突破的试验场。TD-Gammon（西洋双陆棋，1992 年）。Atari-DQN（2013 年）。AlphaGo（2016 年）。AlphaZero（2017 年）。OpenAI Five（Dota 2，2019 年）。AlphaStar（星际争霸 II，2019 年）。MuZero（学习模型，2019 年）。AlphaTensor（矩阵乘法，2022 年）。AlphaDev（排序算法，2023 年）。DeepSeek-R1（数学推理，2025 年）—— 最新证明游戏强化学习技术对文本也行之有效的案例。

本结业项目通过一个统一的视角审视三个里程碑式的架构 —— AlphaZero、MuZero 和 GRPO：**自我对弈 + 搜索 + 策略改进**。每一个都是前一个的泛化；GRPO 尤其将 AlphaZero 的配方应用于 LLM 推理，以 token 作为动作，以数学验证作为获胜信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017 年）。** Silver 等人。给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个单塔网络 `f_θ(s) → (p, v)`。`p` 是合法动作的先验分布。`v` 是预期的游戏结果。
- 蒙特卡洛树搜索（MCTS）：每一步，展开一个可能后续局面的树。使用 `(p, v)` 作为先验和自助法。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：让智能体与自己进行对局。在步骤 `t`，MCTS 的访问分布 `π_t` 成为策略训练的目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。单一的配方，经过数千万次自我对弈后便称霸国际象棋、将棋和围棋。

**MuZero（2019 年）。** Schrittwieser 等人。移除了规则已知的要求。

- 不依赖固定环境，而是学习一个**潜在动力学模型** `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在**学习到的潜在空间**中运行。同样的搜索，同样的训练循环。
- 适用于围棋、国际象棋、将棋 **以及** Atari —— 一个算法，无需规则知识。

**Stochastic MuZero（2022 年）。** 增加了随机动力学和机会节点；扩展到西洋双陆棋这类游戏。

**Muesli、Gumbel MuZero（2022-2024 年）。** 在样本效率和确定性搜索方面进行了改进。

**GRPO（2024-2025 年）。** DeepSeek-R1 配方。相同的 AlphaZero 循环，应用于语言模型推理：

- “游戏”：回答一个数学/编程/推理问题。“获胜” = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：Token。状态：提示 + 已生成的回复。
- 无评论家（PPO 中的 V_φ）。取而代之，对每个提示，从策略中采样 `G` 个完成序列。计算每个序列的奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略施加 KL 惩罚以防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型、无评论家、无 MCTS。组相对基线取代了三者。在推理基准上以远低于 PPO-RLHF 的计算量达到或超过其质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文中介绍了两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型开始。无须 SFT。直接使用 GRPO，包含两个奖励组件：*准确性奖励*（基于规则的 —— 最终答案是否解析为正确的数字 / 代码是否通过单元测试）和*格式奖励*（完成序列是否将其思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均回复长度从约 100 增长到约 10,000 token，数学基准分数攀升至接近 o1-preview 的水平。模型从零开始学会推理。缺点是思维链通常难以阅读、语言混杂、缺乏风格上的润色。
- **R1。** 通过四阶段流程修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集数千条格式清晰的思维链演示。对基座模型进行监督式微调。这提供了一个可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，使用准确性+格式奖励，并增加一个*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点中采样约 60 万条推理轨迹，只保留最终答案正确且思维链可读的轨迹，并与约 20 万条非推理 SFT 示例（写作、问答、自我认知）合并。再次微调基座模型。
  4. **全频谱 GRPO。** 再进行一轮 RL，涵盖推理（基于规则的奖励）和通用对齐（基于偏好利弊的奖励）。

结果在 AIME 和 MATH-500 上匹配 o1，且开源权重，小到可以蒸馏。同一篇论文还发布了六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B），通过对 R1 的推理轨迹进行 SFT 得到 —— 学生无需 RL。对强 RL 教师进行蒸馏，在学生规模上始终优于从零开始 RL。

**为什么推理任务用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出了三个原因：(1) 无需训练价值网络，内存减半；(2) 组基线自然处理推理任务产生的稀疏的轨迹末端奖励；(3) 每个提示的归一化使得优势在不同难度的问题之间具有可比性，而 PPO 的单一评论家无法做到。

**无搜索 vs. 基于搜索。** 游戏已经分化：

- *完全信息且长视野的游戏*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：目前生产中尚无 MCTS；GRPO 使用完整 rollout，推理时使用 Best-of-N。过程奖励模型（PRMs）暗示将逐步搜索重新加入。

## 构建

`code/main.py` 中的代码实现了 **微型 GRPO** —— 一个包含多组样本的老虎机。算法与在 LLM 上使用的相同；只有策略和环境更简单。它教你理解*损失*和*组相对优势*，这是 2025 年的创新。

### 步骤 1：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示的 K 个答案 token 上的 Softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示条件下最后一层的输出。

### 步骤 3：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的诀窍。无需评论家。“基线”是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基线比较（无价值函数）

相同的设置，相同的计算，普通的 REINFORCE。GRPO 收敛更快且更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断：相对于参考策略的平均 KL、策略熵、奖励随时间的变化。一旦这些稳定，训练即完成。

## 陷阱

- **通过玩弄验证器进行奖励破解。** GRPO 继承了 RLHF 的风险：如果验证器有误或可被利用，LLM 就会找到利用点。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组大小太小。** 组基线的方差大约为 `1/√G`。当 `G < 4` 时，优势信号噪声较大；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 完成序列具有不同的对数概率。按 token 数归一化，或使用序列级别的对数概率，或截断至最大长度。
- **纯自我对弈循环。** 在一般和博弈中，AlphaZero 式训练可能会陷入支配循环。通过多样化的对手池（联赛式训练，第 10 课）来缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练就会停滞。
- **计算门槛。** MuZero / AlphaZero 需要大量计算。单次消融实验往往需要数百 GPU 小时。存在用于学习的微型演示（例如 Connect Four 上的 AlphaZero）。
- **验证器覆盖率。** 通过有 bug 的解决方案的单元测试会强化该 bug。设计能捕捉边缘情况的验证器。

## 使用

2026 年游戏强化学习格局，按领域划分：

| 领域 | 主导方法 |
|--------|-----------------|
| 两人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息卡牌游戏（扑克） | CFR + 深度学习（DeepStack, Libratus, Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人在线策略游戏（Dota, 星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five, AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1, Qwen-RL, 开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证） |
| 机器人 | PPO + DR（非游戏强化学习，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor, AlphaDev） |

这个**配方** —— 自我对弈、搜索增强改进、策略蒸馏 —— 涵盖文本、像素和物理控制。GRPO 是最年轻的实例；更多即将到来。

## 交付

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 老虎机。在 2 个提示 × 每个提示 4 个答案 token 上训练。使用 `G=8` 在 <1,000 次更新内收敛。
2. **中等。** 接入 PPO（带裁剪）和普通 REINFORCE。比较样本效率和奖励方差与 GRPO 在同一老虎机上的表现。
3. **困难。** 扩展为长度为 2 的“推理链”：智能体生成两个 token，验证器对这对 token 奖励。测量 GRPO 如何处理跨两步序列的信用分配。（提示：按*完整序列*计算组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | “带学习网络的树搜索” | 蒙特卡洛树搜索；使用学习到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | “自我对弈 + MCTS” | 策略-价值网络训练使其匹配 MCTS 访问分布及游戏结果。 |
| MuZero | “学习模型的 AlphaZero” | 相同循环，但在潜在空间中通过学习的动力学进行。 |
| GRPO | “无评论家的 PPO” | 组相对策略优化；REINFORCE 带组均值基线 + KL。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)` — 用先验平衡价值估计。 |
| 自我对弈 | “智能体与过去的自己对抗” | 零和博弈的标准；对称的训练信号。 |
| 联赛式训练 | “基于种群的自我对弈” | 过去 + 当前 + 掠夺者作为对手被采样。 |
| 验证器奖励 | “可验证的 RL” | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| 过程奖励 | “PRM” | 对每个推理步骤评分，而不仅仅是最终答案。 |

## 扩展阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 提出 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + 深度学习在大规模下的应用。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 应用 GRPO 与自定义奖励函数的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 开源的 R1 配方多规模复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书框架，将自我对弈、搜索与“设计奖励”置于 LLM 规模的情境中。
