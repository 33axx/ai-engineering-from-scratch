# 智能体框架权衡——LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都卖同样的演示（研究型智能体撰写报告），并藏着同样的 bug（状态模式与编排层打架）。选择其抽象与问题形状匹配的框架；其余都是你写两次的胶水代码。

**类型：** 学习  
**语言：** Python  
**前置知识：** 阶段 11 · 09（函数调用），阶段 11 · 16（LangGraph）  
**时长：** ~45 分钟

## 问题

你有一个任务，需要的 LLM 调用不止一次。也许是研究工作流（规划、搜索、总结、引用），也许是代码审查流水线（解析差异、批评、补丁、验证），也许是多轮助手——帮人订机票、写邮件、报销费用。你选了一个框架。

三天后，你发现框架的抽象已经泄漏了。CrewAI 给你角色，但当“研究员”需要把结构化计划交给“写作者”时却跟你打架。AutoGen 给了你智能体之间的聊天，但没有一等公民的状态，所以你的检查点是一个聊天记录的 pickle。LangGraph 给了你状态图，但强制你在知道智能体要做什么之前就命名每个转移。Agno 给你一个单智能体原语，当你试图扇出到三个并发工作者时它就会尖叫。

解决办法不是“选最好的框架”，而是将框架的核心抽象与问题的形状相匹配。本课将绘制这张地图。

## 概念

![智能体框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

2026 年的格局中，四个框架占主导地位。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最佳适用场景 | 最差适用场景 |
|------|---------|-------------|-------------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、条件边、检查点器 | 具有显式状态和人在回路中中断的工作流；需要时间旅行调试的生产级智能体 | 松散的、角色驱动的头脑风暴，拓扑未知 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级） | 角色扮演或角色驱动的工作流，短线性/层级计划 | 任何超越队伍轮次历史的有状态场景；复杂的分支 |
| **AutoGen** | `ConversableAgent` 对——两个或多个智能体轮流说话，直到退出条件 | 多智能体*对话*（师生、提议者-批评者、演员-评审者），思考从聊天中涌现 | 具有已知 DAG 的确定性工作流；任何需要跨重启持久状态的任务 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 记忆，可组合成队伍 | 快速构建的单智能体和轻量级队伍；强多模态和内置存储驱动 | 深度、显式分支的图，带有自定义化简器 |

### “抽象”到底意味着什么

框架的核心抽象就是你在白板上画的那个东西，当你推销架构时。

- **LangGraph** → 你画一个图。节点是步骤，边是转移，每个位置的状态对象是类型化的。心智模型是有限状态机。
- **CrewAI** → 你画一个组织图。每个角色有职位描述，一个管理者路由任务。心智模型是一支小型专家团队。
- **AutoGen** → 你画一个 Slack 私信。两个智能体互相发消息；如果需要主持人，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带着工具的盒子。把盒子放在一起就成了一支队伍。心智模型是“内置电池的智能体”。

### 状态问题

状态是大多数框架选择在生产中崩掉的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型）、字段级化简器、一等公民检查点器（SQLite/Postgres/Redis）。恢复、中断和时间旅行是免费的。*（参见阶段 11 · 16。）*
- **CrewAI。** 状态作为字符串通过 `context` 字段在任务间流动，或者通过 `output_pydantic` 结构化。没有现成的持久化按队伍存储；如果队伍必须存活于重启，你自己加上。
- **AutoGen。** 状态是聊天历史和你定义的任何 `context`。会话记录持久化；任意工作流状态除非你写适配器，否则不会持久化。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent` —— 对话会话和用户记忆自动持久化。不是完整的图检查点器；是会话存储。

### 分支问题

每个非平凡的智能体都会分支。谁决定分支很重要。

- **LangGraph** —— 你决定，通过条件边。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点器记录选择了哪个分支。
- **CrewAI** —— 在层级模式下由管理者决定；在顺序模式下你在构建时决定。路由在任务列表中是隐式的；除了管理者的提示之外，没有一等公民的“if”。
- **AutoGen** —— 智能体通过聊天决定。分支是隐式地从谁下一个说话中涌现的。`GroupChatManager` 选择下一个说话者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** —— 智能体通过接下来调用哪个工具来决定。队伍有协调器/路由器/协作模式；超出此范围的分支由开发者负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel 导出器的 OpenTelemetry。每个节点转移都是一个跟踪跨度；检查点同时充当可回放的跟踪。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年末起有一等公民的 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。跟踪粒度是按智能体消息，而不是按节点。
- **Agno** —— 内置 `monitoring=True` 标志加上 OpenTelemetry 导出器；与 Langfuse 的紧密集成用于会话跟踪。

### 成本与延迟

所有四个框架都增加每次调用的开销（框架逻辑、验证、序列化）。粗略的开销递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。区别主要由框架所做的额外 LLM 路由量决定。CrewAI 的层级管理者花费 token 来决定谁下一步；AutoGen 的 `GroupChatManager` 同样如此。LangGraph 只在写 `llm.invoke` 的地方花费 token。Agno 的单智能体路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而不是 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。一等公民 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具均可适配。通过 `allow_delegation=True` 实现队伍间委托。
- **AutoGen** → `FunctionTool` 封装任何 Python 可调用；MCP 适配器可用。与 AG2 生态系统紧密耦合以实现智能体间模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可在智能体和队伍间共享。

## 技能

> 你可以用一句话解释，为什么某个框架适合某个特定的智能体问题。

构建前的检查清单：

1. **画出形状。** 这是图（类型化状态、命名转移）？角色扮演（专家移交工作）？聊天（智能体说话直到完成）？单智能体带工具？
2. **决定谁分支。** 开发者决定分支 → LangGraph。管理者-智能体决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复？时间旅行？运行中的人类中断？如果是，LangGraph 是默认选择；Agno 会话覆盖对话范围的状态。
4. **检查成本预算。** LLM 选择的路由每轮花费额外 token。如果智能体每天运行数千次，优先选择显式路由。
5. **预算框架开销。** 每个框架都是一个额外的依赖。如果任务只是两次 LLM 调用和一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

拒绝在你能画出图、组织图、聊天或智能体盒子之前拿起框架。拒绝选择一个让你为了实际需要而对抗其状态模型的框架。

## 决策矩阵

| 问题形状 | 推荐框架 | 原因 |
|---------|---------|------|
| 带有类型化状态、人工审批、长时间运行的工作流 DAG | LangGraph | 一等公民状态、检查点器、中断、时间旅行 |
| 具有明确角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 按任务分配角色在 CrewAI 中很容易表达；当分支复杂时扩展到 LangGraph |
| 提议者-批评者或师生对话 | AutoGen | 双智能体聊天是其原生形状 |
| 带有工具、会话、记忆的单智能体 | Agno | 最薄的设置，内置存储和记忆 |
| 数千个带有化简器的并行扇出 | LangGraph + `Send` | 唯一具有一等公民并行调度原语的 |
| 快速原型，无框架承诺 | 纯 Python + 提供商 SDK | 没有框架是最快的框架 |

## 练习

1. **简单。** 取相同的任务——“研究 Anthropic 总部，写 200 词简报，引用来源”——并在 LangGraph（四个节点：规划、搜索、写、引用）和 CrewAI（三个角色：研究员、写作者、编辑）中实现它。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 写作者聊天，编辑通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的单智能体，外加会话存储）中构建同一个任务。对四个实现进行排名，依据 (a) 每次运行成本，(b) 崩溃后恢复能力，(c) 在写步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，它接受简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带有一句话理由的建议。在你自己设计的六个案例上验证。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-------------|---------|
| 编排 | “智能体如何协调” | 决定接下来运行哪个节点/角色/智能体的层次 |
| 持久化状态 | “重启后恢复” | 在进程死亡后仍然存活的状态，附着在检查点或会话存储上 |
| LLM 选择的路由 | “让模型决定” | 计划 LLM 每轮选择下一步；灵活但每次决策花费 token |
| 显式路由 | “开发者决定” | Python 函数或静态边选择下一步；便宜且可审计 |
| 队伍 | “一个 CrewAI 团队” | 角色 + 任务 + 流程（顺序或层级）绑成一个可运行单元 |
| GroupChat | “AutoGen 的多智能体聊天” | 一个管理的 N 个智能体之间的对话，带有说话者选择器 |
| Team（Agno） | “Agno 的多智能体” | 一组智能体上的路由/协调/协作模式 |
| StateGraph | “LangGraph 的图” | 类型化状态、节点、条件边、检查点器原语 |

## 扩展阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、检查点器、中断、时间旅行
- [CrewAI 文档](https://docs.crewai.com/) — 队伍、流程、智能体、任务、进程
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、队伍、工具
- [Agno 文档](https://docs.agno.com/) — Agent、Team、Workflow、存储、记忆
- [Anthropic — 构建高效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 模式库（提示链、路由、并行化、编排者-工作者、评估者-优化器），框架无关
- [Yao 等，“ReAct: Synergizing Reasoning and Acting”（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个框架都粉饰的原语
- [Wu 等，“AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation”（2023）](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文
- [Park 等，“Generative Agents: Interactive Simulacra of Human Behavior”（UIST 2023）](https://arxiv.org/abs/2304.03442) — CrewAI 风格角色堆栈所依托的角色扮演基础
- 阶段 11 · 16（LangGraph） — 本课用作基准的框架
- 阶段 11 · 19（Reflexion） — 一个可以干净地映射到 LangGraph 但在 CrewAI 中却很别扭的模式
- 阶段 11 · 22（生产可观测性） — 如何为你选择的框架进行仪器化
