# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 将思考与行动交错在同一个流中。ReWOO 将它们分离：先制定一个总体计划，然后执行。在 HotpotQA 上减少 5 倍的 token 消耗，准确率提升 4%，并且可以将规划器蒸馏为 7B 模型。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 01 (Agent Loop)
**时间：** 约60分钟

## 学习目标

- 解释为何 ReWOO 的 Planner / Worker / Solver 拆分相比 ReAct 的交错循环能节省 token 并提高鲁棒性。
- 仅用标准库实现一个计划 DAG、一个依赖排序执行器，以及一个组合 Worker 输出的求解器。
- 使用 Anthropic 2026 年的“五种工作流模式”框架，决定任务是应该按计划-执行模式运行，还是使用交错的 ReAct。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长程网页或移动端任务。

## 问题所在

ReAct 的思考-行动-观察交错循环简单而灵活，但每次工具调用都必须携带完整的先前上下文——包括之前的每一次思考。Token 消耗随深度呈二次增长。更糟的是：当工具在循环中失败时，模型必须根据错误观测重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023年5月）注意到了这一点并做出了一个赌注：提前规划好所有内容，并行获取证据，最后组合答案。一次 LLM 调用用于计划，N 次工具调用用于证据（可并行），一次 LLM 调用用于求解。代价是灵活性降低（计划是静态的），但换来的是更好的 token 效率和更清晰的失败模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点指定一个工具、它的参数以及它依赖的较早节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有内容拼接在一起。

### 为什么 token 消耗减少 5 倍

ReAct 的提示长度随步骤数线性增长。到第10步时，提示中包含思考1 + 行动1 + 观测1 + 思考2 + 行动2 + 观测2，依此类推。每个中间步骤还会冗余包含原始提示。

ReWOO 支付一个 planner 提示（较大）、N 个小的 worker 提示（每个只含工具调用，无链式）以及一个 solver 提示。在 HotpotQA 上，论文测量出约 5 倍的 token 减少，同时准确率绝对提升 4 个百分点。

### 为什么更鲁棒

如果 worker 3 在 ReAct 中失败，循环必须在中途从错误中推理。而在 ReWOO 中，worker 3 返回一个错误字符串；solver 结合原始计划在上下文中看到它，并能优雅降级。失败定位是按节点而非按步骤。

### 规划器蒸馏

论文的第二个结果是：由于 planner 不观察结果，你可以用 175B 教师模型输出的 planner 结果微调一个 7B 模型。小模型处理规划；推理时不需要大模型。这已成为标准做法——2026 年的许多生产级智能体使用小规划器和大执行器，或反之。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。前置 planner 输出一个步骤列表，executor 运行每一步，一个可选的 replanner 可以在观察结果后进行修订。这比 ReWOO 更接近 ReAct（replanner 将观测带回规划），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长程网页和移动端智能体。关键贡献是合成计划数据：一个带标签的轨迹生成器产生明确计划的训练数据。用于微调规划器模型，使其能在 WebArena 类任务中持续工作超过 30–50 步，而单一 ReAct 轨迹会在这些任务中失去连贯性。

### 何时选择哪种模式

| 模式 | 适用场景 |
|------|----------|
| ReAct | 短任务，环境未知，需要响应式异常处理 |
| ReWOO | 已知工具的结构化任务，对 token 敏感，证据可并行 |
| Plan-and-Execute | 像 ReWOO 但可以在部分执行后重新规划 |
| Plan-and-Act | 长程（>30步），网页/移动端/计算机操控 |
| Tree of Thoughts | 值得为搜索付出成本（第04课） |

Anthropic 2024年12月的指导：从最简单的模式开始。如果任务只是一个工具调用加总结，不要构建 ReWOO。如果任务是 40 步的研究作业，不要单独使用 ReAct。

## 构建它

`code/main.py` 实现了一个玩具 ReWOO：

- `Planner` — 一个脚本化策略，根据提示输出一个计划 DAG。
- `Worker` — 通过注册表分发每个节点的工具调用。
- `Solver` — 脚本化组合，读取证据并生成最终答案。
- 依赖解析 — 引用如 `#E1` 会在分派时被替换为先前的 worker 输出。

该演示回答了“法国首都的人口是多少（四舍五入到百万）？”使用两步计划：(1) 查找首都，(2) 查找人口，然后求解。

运行它：

```
python3 code/main.py
```

跟踪显示完整的计划，然后是 worker 结果，最后是 solver 组合。将 token 数量（我们打印了粗略的字符数）与 ReAct 风格的交错运行进行比较——在这种结构化任务上 ReWOO 胜出。

## 使用它

LangGraph 将 Plan-and-Execute 作为配方提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码该模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要处于研究阶段；其运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 投入生产。

## 交付它

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成一个 ReWOO 计划 DAG。在交给执行器之前，它会验证计划（无环，所有引用已解析，所有工具存在）。

## 练习

1. 对独立的计划节点并行化 worker 执行。对于一个有 2 个并行组的 6 节点 DAG，这能带来什么好处？
2. 添加一个 replanner 节点，当任一 worker 返回错误时触发。对 ReWOO 做最小的改动使其变成 Plan-and-Execute？
3. 将 `Planner` 替换为小模型（7B 类），并让 `Solver` 保留在前沿模型上。对比端到端质量——拆分在何处失败？
4. 阅读 ReWOO 论文第4节关于规划器蒸馏的内容。从概念上复现 175B -> 7B 的结果：你需要什么训练数据，以及如何评估计划质量？
5. 将玩具示例移植到 Plan-and-Act 的轨迹形式：计划是一个序列而不是 DAG。哪些权衡发生了变化？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| ReWOO | “无观测推理” | 先计划，然后并行获取证据，最后求解——计划提示中不含观测 |
| Plan-and-Execute | “LangChain 的计划-执行模式” | ReWOO 加上执行后可选的 replanner 节点 |
| Plan-and-Act | “扩展的计划-执行” | 显式的规划器/执行器拆分，带有用于长程任务的合成计划训练数据 |
| 证据引用 | "#E1, #E2, ..." | 计划节点占位符，分派时替换为先前 worker 的输出 |
| 规划器蒸馏 | “小规划器，大执行器” | 用大教师模型的规划轨迹微调小模型 |
| Token 效率 | “更少的往返” | 在论文的 HotpotQA 上相比 ReAct 减少 5 倍 token |
| DAG 执行器 | “拓扑调度器” | 按依赖顺序运行计划节点；每一层可并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 经典论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 使用合成计划扩展的规划器-执行器
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单且有效的模式
