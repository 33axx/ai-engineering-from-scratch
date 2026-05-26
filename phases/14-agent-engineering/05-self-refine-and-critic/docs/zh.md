# Self-Refine 和 CRITIC：迭代输出改进

> Self-Refine（Madaan 等，2023）在循环中使用同一 LLM 扮演三个角色——生成、反馈、优化——平均在 7 个任务上获得 +20 的绝对提升。CRITIC（Gou 等，2023）通过将验证步骤路由到外部工具来强化反馈环节。到 2026 年，这种模式在所有框架中都以“评估器-优化器”（Anthropic）或护栏循环（OpenAI Agents SDK）的形式出现。

**类型：** 构建  
**语言：** Python (stdlib)  
**前置要求：** 第14阶段 · 01（智能体循环）、第14阶段 · 03（反思）  
**时间：** 约60分钟

## 学习目标

- 说明 Self-Refine 的三个提示（生成、反馈、优化），并解释为什么历史信息对优化提示很重要。
- 解释 CRITIC 的关键洞见：LLM 在没有外部依据的情况下进行自我验证是不可靠的。
- 用 stdlib 实现一个带历史信息的 Self-Refine 循环，并可选地加入外部验证器。
- 将此模式映射到 Anthropic 的“评估器-优化器”工作流和 OpenAI Agents SDK 的输出护栏。

## 问题

智能体产生了一个几乎正确的答案。可能代码行有语法错误，可能摘要太长，可能计划遗漏了一个边界情况。你想要的是：智能体自我批评其输出，然后修复它。

Self-Refine 表明，使用单个模型、无需训练数据、无需强化学习就能实现这一点。但有一个问题：LLM 在硬事实上的自我验证能力很差。CRITIC 指出了修复方法——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认方案：生成、验证（可能时使用外部工具）、优化、在验证器通过时停止。

## 概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整的历史记录——所有先前的输出和批评——因此它不会重复错误。论文通过消融实验证明：去掉历史记录后质量会大幅下降。

标题：在 7 个任务（数学、代码、首字母缩略词、对话）上平均获得 +20 的绝对提升，包括 GPT-4。无需训练、无需外部工具、单个模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4，2024年2月）

Self-Refine 的弱点：反馈步骤是 LLM 自我评分。对于事实性声明，这是不可靠的（模型产生的幻觉通常对自身来说看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：
- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、linter）。

验证器基于工具结果产生结构化的批评。然后优化器以此为条件进行优化。

标题：CRITIC 在事实性任务上优于 Self-Refine，因为批评是有根据的。在没有外部验证器的任务（创意写作、格式排版）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。首选这种情况（单元测试、类型检查器、护栏断言）。
2. **未发出反馈。** 模型说“输出没问题。”更便宜但不可靠；配合最大迭代次数上限使用。

2026 年默认：两者结合。“当验证器通过或模型说没问题且迭代次数 >=2 或迭代次数 >= 最大迭代次数时停止。”

### 评估器-优化器（Anthropic，2024）

Anthropic 在 2024 年 12 月的博文中将其列为五个工作流模式之一。两个角色：
- 评估器：对输出评分并产生批评。
- 优化器：根据批评修改输出。

循环直到评估器通过。这是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 增加的关键工程细节：评估器和优化器的提示应该显著不同，这样模型就不会只是走过场。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将这种模式作为“输出护栏”提供。护栏是一个验证器，在智能体的最终输出上运行。如果护栏触发（抛出 `OutputGuardrailTripwireTriggered`），则输出被拒绝，智能体可以重试。护栏可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年陷阱

- **走过场循环。** 相同模型使用相同提示风格进行生成和批评会收敛到“看起来没问题。”使用结构不同的提示，或使用更小更便宜的模型进行批评。
- **过度优化。** 每次优化都会增加延迟和 token 消耗。预算 1-3 次；之后升级到人工审查。
- **在琐碎任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器付出延迟代价。

## 动手构建

`code/main.py` 实现了一个玩具任务上的 Self-Refine 和 CRITIC：根据主题生成一个短列表。验证器检查格式（3 个条目，每个不超过 60 个字符）。CRITIC 增加了一个外部“事实验证器”，用于惩罚已知的幻觉。

组件：
- `generate` —— 脚本化的生产者。
- `feedback` —— LLM 风格的自我批评。
- `verify_external` —— CRITIC 风格的有根据验证器。
- `refine` —— 根据历史重写输出。
- 停止条件 —— 验证器通过或最多 4 次迭代。

运行它：

```
python3 code/main.py
```

比较 Self-Refine 与 CRITIC 的运行结果。CRITIC 捕捉到了 Self-Refine 遗漏的一个事实错误，因为外部验证器具有自我批评者所没有的依据。

## 使用它

Anthropic 的评估器-优化器是这种模式在 Claude 友好语言下的表述。OpenAI Agents SDK 的输出护栏是 CRITIC 形状的（护栏可以调用工具）。LangGraph 提供了一个反射节点，读起来像 Self-Refine。Google 的 Gemini 2.5 Computer Use 添加了一个每步安全评估器，它是 CRITIC 的变体：每个动作在提交前都会被验证。

## 交付它

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置了一个评估器-优化器循环。会输出生成器、评估器/验证器和优化器的提示，以及停止策略。

## 练习

1. 将 `max_iterations=1` 运行玩具任务。CRITIC 仍然有用吗？
2. 用噪声验证器（随机 30% 的误报）替换外部验证器。循环会怎样？这就是 2026 年大多数护栏栈的现实。
3. 实现一个“生成器-批评者使用不同模型”的变体：大模型生成，小模型批评。它能超过相同模型吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列出三类验证工具，并为每类给出一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 在哪些方面做得不对，哪些方面做得好？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Self-Refine | “自我修正的 LLM” | 在单个模型中循环执行生成 -> 反馈 -> 优化，带历史记录 |
| CRITIC | “工具依据验证” | 用外部验证器（搜索、代码、计算、测试）替代反馈 |
| 评估器-优化器 | “Anthropic 工作流模式” | 两个角色——评估器评分，优化器修改——循环直至收敛 |
| 输出护栏 | “事后检查” | OpenAI Agents SDK 在智能体产生输出后运行的验证器 |
| 验证步骤 | “批评阶段” | 关键决策：有根据还是自我评分 |
| 优化历史 | “模型已经尝试过的内容” | 先前的输出 + 批评附加到优化提示中；去掉会导致质量崩溃 |
| 走过场循环 | “自我一致失败” | 相同提示的批评返回“看起来没问题”；通过结构不同的提示修复 |
| 停止条件 | “收敛测试” | 验证器通过 或 无反馈且达到迭代上限；永远不要单一条件 |

## 延伸阅读

- [Madaan 等，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 经典论文
- [Gou 等，CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) —— 工具依据验证
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents) —— 评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— 作为 CRITIC 形状验证器的输出护栏
