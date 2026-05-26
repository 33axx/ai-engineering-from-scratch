# 多智能体原语模型

> 2026 年推出的每一个多智能体框架 —— AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework —— 都只是一个四维设计空间中的一个点。仅有四个原语，别无其它：智能体（agent）、移交（handoff）、共享状态（shared state）、编排器（orchestrator）。本课将从零构建它们，在一个玩具系统上运行全部四个原语，然后将每个主流框架映射到相同的坐标轴上，让你能用一段话读懂任何新版本。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** 阶段 14（智能体工程），阶段 16 · 01（为什么需要多智能体）
**时间：** 约 60 分钟

## 问题

每六个月就会有一个新的多智能体框架发布。2023 年的 AutoGen，2024 年的 CrewAI，2024 年的 LangGraph 和 OpenAI Swarm，2025 年 4 月的 Google ADK，2026 年 2 月的 Microsoft Agent Framework RC。每份新闻稿都声称自己是“正确的抽象”。

如果你试图逐一学习它们，你会精疲力尽。API 看起来各不相同。文档对“智能体”的定义也互相矛盾。一个框架称其共享内存为“黑板”（blackboard），另一个称为“消息池”（message pool），第三个称为“StateGraph”。你会开始怀疑这个领域只是在原地打转。

事实并非如此。在营销表象之下，四个原语是稳定的。学会它们一次，就能用一段话读懂每一个新框架。

## 概念

### 四个原语

1. **智能体（Agent）** —— 一个系统提示词加一个工具列表。无状态；每次运行都从系统提示词和当前消息历史开始。
2. **移交（Handoff）** —— 控制权从一个智能体到另一个智能体的结构化转移。从机制上讲，可以是一个返回新智能体的工具调用，也可以是一个跟随条件的图边。
3. **共享状态（Shared state）** —— 任何可以被多个智能体读取（有时写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）** —— 决定谁下一个发言的人。选项：显式图（确定性）、LLM 发言者选择器（软性）、上一个发言者的移交调用（OpenAI Swarm）、或基于队列的调度器（群体架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余的都是表面语法。

### 每个 2026 框架如何映射到它

| 框架 | 智能体 | 移交 | 共享状态 | 编排器 |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一个移交调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | 基于发言者选择器的 GroupChat | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 制品 | 主机决定 |

表面差异看起来很大。本质上：相同的四个旋钮。

### 为什么这很重要

一旦你看到了原语，框架对比就变成了一个简短清单：

- 编排器是信任 LLM 进行路由（Swarm），还是将路由固定在代码中（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影式（StateGraph 归约器）？
- 智能体可以修改彼此的提示词（CrewAI 管理器），还是只能移交（Swarm）？

这三个问题能回答 80% 的框架适配哪个特定问题。你不再选购“最好的多智能体框架”，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除了共享状态，每个原语都是无状态的。智能体是（提示词，工具）的函数。移交是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西是共享状态。** 所有有趣的 bug 都在这里：记忆中毒（第 15 课）、消息排序、版本控制、写入冲突。

隐藏共享状态的框架（Swarm）会把问题推给调用者。集中化共享状态的框架（LangGraph 检查点、AutoGen 池）使其可检查，但将协调成本转移到了共享状态实现上。

### 单一原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。拥有相同系统提示词和工具的两个智能体是可互换的。所有看似智能体状态的东西实际上都在共享状态或移交协议中。

#### 移交

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导地位：

- **函数返回** —— 工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由信息。
- **图边** —— LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **发言者选择** —— AutoGen GroupChat。一个选择器函数（有时本身就是一个 LLM 调用）读取池并选择谁下一个发言。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少情况下，是一个消息列表。通常更多：结构化制品（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑：**全量池**（每个智能体看到每条消息）和**投影池**（智能体看到角色限定的视图）。全量池简单但扩展性差。投影池可扩展但需要预先设计模式。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态** —— 图在构建时固定（LangGraph 确定性、CrewAI 顺序）。
- **LLM 选择** —— LLM 读取池并选择下一个发言者（AutoGen、CrewAI 分层）。
- **移交驱动** —— 当前智能体通过调用移交工具决定（Swarm）。
- **队列驱动** —— 工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间的变化

一旦原语固定下来，剩余的设计决策包括：

- **记忆策略** —— 临时记忆 vs. 持久化检查点（LangGraph 检查点）。
- **安全边界** —— 谁可以批准移交（人机交互）。
- **成本核算** —— 每个智能体的 token 预算。
- **可观测性** —— 跟踪移交、持久化状态以便重放。

所有这些都可以在原语之上实现。它们都不是新的原语。

## 构建

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真正的 LLM —— 每个智能体是一个脚本化策略，这样焦点就能保持在协调结构上。

文件导出了：

- `Agent` —— 一个包含名称、系统提示词、工具、策略函数的数据类。
- `Handoff` —— 一个返回新智能体的函数。
- `SharedState` —— 一个线程安全的消息池。
- `Orchestrator` —— 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排器类型运行相同的三智能体流水线（研究 → 撰写 → 审阅），并在最后打印消息池。你可以看到输出仅在*谁选择下一个*上有所不同；智能体和共享状态在三次运行中完全相同。

运行：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次打印最终的消息池。如果研究者判定任务提前完成，移交驱动的运行会到达更少的智能体 —— 这就是 LLM 路由权衡的缩影。

## 使用

`outputs/skill-primitive-mapper.md` 是一项技能，它可以读取任何多智能体代码库或框架文档，并返回四个原语的映射。在新框架版本发布时运行它，就能在深入阅读文档之前获得一段话的理解。

## 部署

在采用一个新框架之前，为它编写原语映射。如果你做不到，说明文档不完整，或者框架在发明第五个原语（很少见 —— 检查是否有你未曾见过的共享状态变体）。

将映射固定在你的架构文档中。当新团队成员加入时，先发送映射，再发送 API 文档。当框架版本变化时，比较映射的差异，而不是变更日志。

## 练习

1. 使用不同的智能体策略运行 `code/main.py` 三次。观察编排器选择如何改变哪些智能体会运行。
2. 实现第四种编排器类型：队列驱动式，智能体轮询共享状态以获取工作。可能会发生什么死锁？你如何检测它？
3. 获取 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并将其重写为四个原语。LangGraph 的哪些抽象是一对一映射的，哪些是便利包装器？
4. 阅读 OpenAI Swarm 食谱（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 使四个原语中的哪一个最易用，以及它将哪一个推给了调用者。
5. 在此表中找到一个完全隐藏共享状态的框架。解释当智能体需要在多次移交之间协调而不重新读取历史时，会发生什么故障。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|----------------|------------------------|
| 智能体 | “一个带工具的 LLM” | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| 移交 | “控制权转移” | 一个命名下一个智能体和可选负载的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| 共享状态 | “记忆” / “上下文” | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器 | “协调者” | 决定谁下一个运行的人。静态图、LLM 选择器、移交驱动或队列驱动。 |
| 原语 | “抽象” | 每个框架参数化的四个轴之一。不是框架特性。 |
| 消息池 | “共享聊天历史” | 全历史共享状态。易于推理，扩展性差。 |
| 投影状态 | “限定的视图” | 共享状态中角色特定的视图。可扩展，需要模式设计。 |
| 发言者选择 | “谁下一个发言” | 编排器模式，一个函数（通常是 LLM）从组中选出下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 移交驱动编排最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) —— GroupChat + 发言者选择是 LLM 选择编排的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 图边编排和基于归约器的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) —— 角色-目标-背景智能体，顺序/分层流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) —— 微软将 v0.4 移至维护后正在活跃的 AutoGen v0.2 分支
