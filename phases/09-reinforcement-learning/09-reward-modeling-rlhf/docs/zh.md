# 奖励建模与 RLHF

> 人类无法为“优秀的助手回答”编写奖励函数，但他们可以比较两个回答并选出更好的一个。将奖励函数拟合到这些比较上，然后让语言模型通过强化学习（RL）依据该奖励进行优化。Christiano 2017。InstructGPT 2022。正是这一配方将 GPT-3 变成了 ChatGPT。到 2026 年，它大部分被 DPO 取代——但思维模型保持不变。

**类型：** 构建
**语言：** Python
**先修知识：** 第 5 阶段 · 05（情感分析），第 9 阶段 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你以“预测下一个 token”为目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、漫无边际地闲聊，并且拒绝该拒绝的请求。你无法通过更多预训练来修复这个问题——网络文本本身就是问题，而不是解药。

你需要一个**标量奖励**，能够说“对于指令 X，回答 A 比回答 B 更好”。手动编写这样的奖励函数是不可能的。“有用性”并不是一个关于 token 的封闭形式表达式。但人类可以比较两个输出并标记偏好。这在大规模收集时成本低廉。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转换为奖励模型，然后通过 PPO 基于该奖励优化 LM。分三步：SFT → RM → PPO。正是这个配方交付了 ChatGPT、Claude、Gemini 以及 2023–2025 年间所有其他对齐的大语言模型。

到 2026 年，PPO 步骤大部分被 DPO（第 10 阶段 · 08）取代，因为它更便宜，并且在对齐微调方面效果几乎一样好。但**奖励模型**部分仍然支撑着每一个 Best-of-N 采样器、每一个基于可验证奖励的 RL 流水线，以及每一个使用过程奖励模型的推理模型。理解了 RLHF，你就理解了整个对齐栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**第 1 阶段：有监督微调（SFT）。** 从一个预训练基座模型开始。基于人工编写的目标行为演示（遵循指令的回答、有用的回复等）进行微调。结果：一个 `π_SFT` 模型，**倾向于良好行为**，但动作空间仍是无限的。

**第 2 阶段：奖励模型训练。**

- 收集针对提示 `x` 的回答对 `(y_+, y_-)`，并由人类标记为“y_+ 优于 y_-”。
- 训练一个奖励模型 `R_φ(x, y)`，使其对 `y_+` 赋予更高分数。
- 损失函数：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励的差值隐含了偏好的对数几率。Bradley-Terry 自 1952 年以来一直是标准，也是现代 RLHF 中的主导选择。

- `R_φ` 通常从 SFT 模型初始化，并在其顶部加上一个标量输出头。相同的 transformer 骨干网络；单个线性层输出奖励。

**第 3 阶段：使用 KL 惩罚的 PPO 与奖励模型。**

- 从 `π_SFT` 初始化可训练的策略 `π_θ`。保留一个冻结的**参考模型** `π_ref = π_SFT`。
- 在回答 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 与 `π_SFT` 偏离太远——它是一个**正则化项**，而不是硬性的信任区域。`β` 通常取 `0.01`–`0.05`。
- 使用这个奖励运行 PPO（第 08 课）。优势是在 token 级别的轨迹上计算的，但 RM 只对整个回答进行评分。

**为什么需要 KL？** 如果没有它，PPO 会愉快地找到奖励黑客策略——RM 只在分布内的补全上训练过。一个分布外的回答可能比任何人工编写的回答得分都高。KL 使 `π_θ` 保持在 RM 训练过的流形附近。它是 RLHF 中最重要的一个旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：封闭形式的代数将第 2 和第 3 阶段合并为单个关于偏好数据的有监督损失。不需要 RM，也不需要 PPO。在对齐基准上质量相同，但计算量大大减少。详见第 10 阶段 · 08。
- **GRPO**（DeepSeek 2024–2025）：使用组相对基线（而非评论家）的 PPO，奖励来自**验证器**（代码运行 / 数学答案匹配）而非人工训练的 RM。在推理模型中占主导地位。详见第 9 阶段 · 12。
- **过程奖励模型（PRM）：** 对部分解决方案（每个推理步骤）进行评分，在 RLHF 和 GRPO 变体中用于推理。
- **宪法 AI / RLAIF：** 使用对齐的大语言模型生成偏好，取代人工。扩展了偏好的预算。

## 构建

本课使用微小的合成“提示”和“回答”，以字符串形式表示。RM 是基于 token 袋表示法的线性评分器。不使用真正的大语言模型——重要的是流水线的**形状**，而非规模。参见 `code/main.py`。

### 第 1 步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在实际的 RLHF 中，这由人工标注员完成。形状——`(prompt, preferred_response, rejected_response)`——完全相同。

### 第 2 步：Bradley-Terry 奖励模型

线性得分：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对 log 损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 将正权重分配给好词 token，将负权重分配给坏词。

### 第 3 步：基于 RM 的 PPO 风格策略

我们的玩具策略从词汇表中生成单个 token。我们在 RM 下对该 token 评分，计算 `log π_θ(token | prompt)`，加上对参考模型的 KL 惩罚，并应用裁剪后的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第 4 步：监控 KL

每次更新时跟踪平均 `KL(π_θ || π_ref)`。如果超过 `~5-10`，说明策略已经严重偏离 `π_SFT`——`β` 设置得太低，或者奖励黑客已经开始。这是实际 RLHF 中最重要的诊断指标。

### 第 5 步：使用 TRL 的生产配方

一旦你理解了玩具流水线，这就是真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于第 2 阶段，`PPOTrainer`（内置了对参考模型的 KL）用于第 3 阶段。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做了三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 加倍；如果低于一半，β 减半。参考模型按惯例冻结——你必须小心不要与 `policy` 共享参数。值头与策略共享同一个骨干网络（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化 / 奖励黑客。** RM 不完美；`π_θ` 会找到那些得分高但实际很差的对抗性补全。症状：奖励无限上升，而人类评估分数停滞或下降。解决方法：提前停止，提高 `β`，扩大 RM 训练数据。
- **长度黑客。** 在有用回答上训练的 RM 往往会隐式地奖励长度。策略学会了填充回答。补救措施：长度归一化奖励，或使用具备长度意识的 RM 的 RLAIF。
- **RM 太小。** RM 至少需要和策略一样大。一个小的 RM 无法忠实地对策略的输出进行评分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是使用**自适应** β，目标是每步固定的 KL。
- **偏好数据噪声。** 大约 30% 的人工标记有噪声或模糊。通过在一致过滤后的数据上训练 RM，或对 BT 使用温度来校准。
- **离策略问题。** PPO 数据在第一个 epoch 后略微离策略。像第 08 课那样监控裁剪比例。

## 使用

2026 年的 RLHF 分层如下：

| 层级 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有用性、无害性 | 对齐 | DPO（第 10 阶段 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（第 9 阶段 · 12）。 |
| 长程多步任务 | 智能体 | 使用过程奖励模型按步骤的 PPO / GRPO。 |
| 安全性 / 拒绝行为 | 安全 | 使用独立安全 RM 的 RLHF-PPO，或宪法 AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；不需要策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结的 LM 之上训练一个小型“奖励头”。 |

RLHF 在 2022–2024 年是**主要方法**。到 2026 年，生产级对齐流水线以 DPO 为先，仅在对 RM 要求高或安全关键的步骤中使用 PPO。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中对 500 个合成偏好对训练 Bradley-Terry 奖励模型。在 100 个保留对上进行成对准确率测量。应超过 90%。
2. **中等。** 使用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每个 β，绘制 RM 分数与对参考模型 KL 随更新的关系图。哪个运行出现了奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（封闭形式偏好似然损失），并与 RLHF-PPO 流水线在计算使用量和最终 RM 得分上进行对比。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | “对齐 RL” | 三阶段 SFT + RM + PPO 流水线（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（RM） | “评分网络” | 通过 Bradley-Terry 拟合到成对偏好的学习标量函数。 |
| Bradley-Terry | “成对逻辑损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准的 RM 目标。 |
| KL 惩罚 | “靠近参考模型” | 奖励中的 `β · KL(π_θ || π_ref)`；用于防止奖励黑客的正则化项。 |
| 奖励黑客 | “古德哈特定律” | 策略利用 RM 缺陷；症状：奖励上升，人类评估平坦。 |
| RLAIF | “AI 标记的偏好” | RLHF 中标签来自另一个 LM 而非人类。 |
| PRM | “过程奖励模型” | 对部分推理步骤评分；用于推理流水线。 |
| 宪法 AI | “Anthropic 的方法” | 由 AI 根据明确规则生成的偏好。 |

## 延伸阅读

- [Christiano et al.（2017）. Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) —— 开启 RLHF 的论文。
- [Ouyang et al.（2022）. InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon et al.（2020）. Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) —— 较早的用于摘要的 RLHF。
- [Rafailov et al.（2023）. Direct Preference Optimization](https://arxiv.org/abs/2305.18290) —— DPO；2026 年后 RLHF 的默认选择。
- [Bai et al.（2022）. Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) —— RLAIF 和自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码了解自适应 KL 和值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) —— Lambert、Castricato、von Werra、Havrilla 撰写，附图的三阶段流水线权威讲解。
- [von Werra et al.（2020）. TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) —— 该库；`examples/` 中包含针对 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto（2018）. Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假设观点；思考奖励黑客的重要先决知识。
