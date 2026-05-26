# LangGraph：有状态图与持久化执行

> LangGraph 是 2026 年低层级有状态编排的参考标准。智能体是一个状态机；节点是函数；边是转换；状态是不可变的，并且在每一步之后都会进行快照。可以从任意故障点精确恢复。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（智能体循环），Phase 14 · 12（工作流模式）
**时间：** ~75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：具有不可变状态、函数节点、条件边和步后检查点的状态机。
- 列举文档强调的四个能力：持久化执行、流式传输、人机协同、全面记忆。
- 解释 LangGraph 支持的三种编排拓扑：监督者、点对点（群蜂）、分层（嵌套子图）。
- 使用标准库实现一个包含不可变状态、条件边和检查点/恢复循环的状态图。

## 问题

智能体和工作流共享一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二等状态模型迫使运维人员在假设每次运行都是从零开始的库上拼凑重试逻辑。

LangGraph 的设计答案：状态是一等公民的带类型对象，变更是显式的，检查点在每个节点后持久化。恢复只需调用 `load_state(session_id)`。

## 概念

### 图

一个图由以下部分定义：

- **状态类型。** 一个带类型的字典（或 Pydantic 模型），每个节点都读取并变更加。
- **节点。** 纯函数 `(state) -> state_update`。返回后更新会合并到状态中。
- **边。** 节点之间的条件转换或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——一个路由工作流作为图。

### 持久化执行

每个节点返回后，运行时序列化状态并将其写入检查点（SQLite、Postgres、Redis、自定义）。如果在第 N 步失败，运行时可以调用 `resume(session_id)` 并从第 N+1 步以精确状态继续。

LangGraph 文档明确强调了对此有需求的用户：Klarna、Uber、J.P. Morgan。其价值不在于图的结构，而在于图结构加上检查点使得恢复成本低廉。

### 流式传输

每个节点可以产生部分输出。图按节点增量事件流式传输给调用者，这样 UI 可以在图运行的同时更新。

### 人机协同

在节点之间检查并修改状态。实现方式：在关键节点前暂停，将状态展示给人类，接受修改，然后恢复。检查点使这变得容易，因为状态已经序列化。

### 记忆

短期记忆（运行内——状态中的对话历史）和长期记忆（跨运行——通过检查点加上独立的长时存储进行持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **监督者。** 中央路由器 LLM 将任务分配给专家子智能体。`create_supervisor()` 在 `langgraph-supervisor` 中（尽管 LangChain 团队在 2026 年建议直接通过工具调用来获得更多上下文控制）。
2. **群蜂 / 点对点。** 智能体通过共享工具表面直接交接。没有中央路由器。
3. **分层。** 监督者管理子监督者，通过嵌套子图实现。

### 这种模式何时会出错

- **检查点太小。** 只对对话轮次进行快照会导致工具状态和记忆写入无法恢复。必须序列化完整状态。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、墙钟时间、外部 API 必须被捕获。
- **过度使用条件边。** 每条边都是条件边的图是一个无法推导的状态机。优先使用线性链，偶尔分支。

## 构建它

`code/main.py` 实现了一个基于标准库的有状态图：

- `State` — 一个带有 `messages`、`step`、`route`、`output`、`human_approval` 的带类型字典。
- `Node` — 可调用对象，接收状态并返回更新字典。
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存模拟）— 在每个节点之后序列化状态；`load(session_id)` 恢复状态。
- 一个演示图：分类 -> 分支（退款 / 缺陷 / 销售）-> 人工门控 -> 发送。

运行它：

```
python3 code/main.py
```

追踪显示第一次运行在人工门控处失败，状态持久化，然后恢复产生最终输出。

## 使用它

- **LangGraph** — 参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建你自己的图。
- **AutoGen v0.4**（第 14 课）— 适用于高并发场景的参与者模型替代方案。
- **Claude Agent SDK**（第 17 课）— 带有内置会话存储的管理框架。
- **自定义** — 当你需要对状态形状或检查点后端有精确控制时。

## 交付它

`outputs/skill-state-graph.md` 生成一个 LangGraph 形状的状态图，适用于任意目标运行时，并内置了检查点和恢复功能。

## 练习

1. 当分类置信度低于阈值时，添加一条从 `classify` 到 `end` 的条件边。在人类手动设置 `route` 后恢复运行。
2. 将类似 SQLite 的模拟替换为真正的 SQLite 检查点。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义归约器合并。不可变状态在这里带来了什么好处？
4. 阅读 `langgraph-supervisor` 参考。将示例移植到 `create_supervisor`。比较追踪形状。
5. 添加流式传输：每个节点在运行时产生部分状态。打印到达的增量。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| State graph | "智能体作为状态机" | 带类型状态 + 节点 + 边 + 归约器 |
| Checkpointer | "持久化后端" | 在每个节点后序列化状态；支持恢复 |
| Reducer | "状态合并器" | 将当前状态与节点的更新合并的函数 |
| Conditional edge | "分支" | 由状态函数选择的边 |
| Subgraph | "嵌套图" | 作为另一个图内部节点使用的图 |
| Durable execution | "从失败中恢复" | 从最后一个成功节点以精确状态重新开始 |
| Supervisor | "路由器 LLM" | 专家子智能体的中央调度器 |
| Swarm | "点对点智能体" | 智能体通过共享工具进行交接；无中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 参与者模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储与子智能体
