# 直接偏好优化家族

> Rafailov 等人（2023）证明了 RLHF 的最优解在偏好数据方面具有闭式形式，因此你可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个都修复了 DPO 的一种失效模式。到 2026 年，直接对齐算法在前沿后训练中比 PPO 更常见。但第 2 课中的过度优化曲线仍然适用：DAA 无法逃脱古德哈特定律，它们只是改变了问题发生的位置。

**类型：** 学习  
**语言：** Python（stdlib，六变体偏好损失比较器）  
**前置知识：** 阶段 18 · 01（InstructGPT），阶段 18 · 02（奖励黑客），阶段 10 · 08（DPO 基础）  
**时间：** ~75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导出 DPO 闭式形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失效模式。
- 区分“隐式奖励差距”和“偏好强度”，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAA 即使没有显式 RM 也会过度优化。

## 问题

RLHF 目标函数（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比值隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 因仅依赖 `x` 而抵消。剩下的只是策略参数上的损失——不需要奖励模型。这就是 DPO。

问题在于：该推导假设最优解可达、偏好数据分布内、参考策略是真正的模态锚点。这些条件均不严格成立。家族中的每个成员都修复了一个不同的被违反假设。

## 概念

### DPO（Rafailov 等人，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。微小的偏好可能产生任意大的差距。
- 损失使选中和拒绝的对数概率朝相反方向移动。它可能拉低选中的绝对对数概率，只要拒绝下降得更快。这就是降级的选择响应现象。
- 分布外偏好（罕见 vs 罕见对）会产生任意的隐式奖励。

### IPO（Azar 等人，2024）

恒等偏好优化将对数 sigmoid 替换为偏好概率上的恒等映射。损失变成有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际由 `1/(2 beta)` 界定。偏好强度与隐式奖励差距成正比。不会爆炸。

### KTO（Ethayarajh 等人，2024）

Kahneman-Tversky 优化完全放弃了成对结构。给定一个带有标签的输出和一个二值的“期望”或“不期望”信号，它映射到一个展望理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

收益和损失权重不同（损失厌恶）。优点：可以使用不成对数据，这类数据丰富得多。

### SimPO（Meng 等人，2024）

简单偏好优化使训练信号与生成对齐。完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有边际 `gamma` 以稳定训练。长度归一化消除了利用 DPO 长度偏差失效模式的动机（更长的 `y_w` 自然产生更大的对数概率差距）。

### ORPO（Hong 等人，2024）

比值比偏好优化在标准 SFT 负对数似然上添加了一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无参考策略——SFT 项就是正则化器。从基模型到对齐模型单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别了降级的选择响应问题：DPO 保持了排序 `y_w > y_l`，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一行修正，惩罚选中响应的向下移动。在 Llama-3.1-8B-Instruct 数学推理上报告比 DPO 准确率提升 +10.1%。

### 通用结果：DAA 仍然过度优化

Rafailov 等人《直接对齐算法中奖励模型过度优化的缩放定律》（NeurIPS 2024）使用 DPO、IPO、SLiC 在多个数据集上训练策略，跨 KL 预算。真实奖励 vs KL 曲线具有与 Gao 等人相同的峰值后崩溃形状。隐式奖励在训练过程中查询了分布外样本；KL 正则化无法稳定这一过程。

DAA 无法逃脱古德哈特定律。它们只是将问题表面从“奖励模型过度优化”变为“参考策略比值过度优化”。通用的修复方法——更好的数据、集成、早停——对两者都适用。

### 如何选择（2026 年）

- 如果你有大量成对偏好数据：使用保守 beta 的 DPO；若长度偏差明显则用 SimPO。
- 如果你有不成对二值反馈：用 KTO。
- 如果你希望从基模型单阶段流水线：用 ORPO。
- 如果你在 DPO 日志中看到降级的选择对数概率：用 BPO。
- 如果偏好强度变化很大且 DPO 饱和：用 IPO。

每个实验室都会对这五种方法进行电池测试，并针对每个任务选出最优。没有理由认为数学推理和安全的最优方法是相同的。

## 使用它

`code/main.py` 在一个玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度随样本对变化。每种损失都在同一个包含 500 对的样本上使用小型 softmax 策略进行优化。输出每种方法的最终胜率、选择对数概率漂移和隐式奖励分布。

## 部署它

本课程产出 `outputs/skill-preference-loss-selector.md`。根据数据集统计（成对 vs 不成对、偏好强度可变 vs 均匀、长度分布）和训练目标（单阶段或 SFT 后偏好），推荐一种偏好损失并报告它防范的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 最终的选中对数概率下降。BPO 应保持更高的选中绝对概率——验证这一点。

2. 修改偏好数据，使所有对的强度相等。六种方法中哪一种最鲁棒？哪一种退化？解释 IPO 在这方面的优势。

3. 使拒绝响应的平均长度是选中响应的 2 倍。在不更改其他设置的情况下，数值上展示 DPO 的长度利用问题以及 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 过度优化。重现一个单点版本：绘制选中减拒绝的 KL 散度，并观察大 beta 下 DPO 的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写出 BPO 对 DPO 添加的一行修正。对照 `code/main.py` 中的实现进行确认。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| DPO | “无奖励模型的 RLHF” | 从 RLHF 闭式最优解推导的损失；仅策略参数 |
| 隐式奖励 | “对数比值” | `beta * log(pi(y|x) / pi_ref(y|x))` — DPO 隐含的奖励 |
| IPO | “有界 DPO” | 将对数 sigmoid 替换为恒等映射；隐式奖励差距由 `1/(2 beta)` 界定 |
| KTO | “无配对 DPO” | 对单个标签的展望理论效用，带有损失厌恶 |
| SimPO | “无参考 DPO” | 长度归一化对数似然 + 边际；无参考策略 |
| ORPO | “单阶段 DPO” | NLL + 比值比偏好项；从基模型一次训练 |
| BPO | “保留选中的 DPO” | DPO 加上惩罚，降低选中响应绝对对数概率的减少 |
| 降级的选择 | “选中下降” | DPO 降低选中的对数概率，只要拒绝下降更快 |
| DAA | “直接对齐算法” | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov 等人 — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar 等人 — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等人 — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
