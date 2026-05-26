# LangGraph — 智能体的状态机

> 手写的 ReAct 循环是 `while True`。用 LangGraph 写的 ReAct 循环是一个可检查点、可中断、可分支、可时间旅行的图。智能体本身没变，变的是它周围的封装。

**类型:** 构建
**语言:** Python
**前置条件:** 阶段 11 · 09（函数调用），阶段 11 · 14（模型上下文协议）
**时间:** 约 75 分钟

## 问题

你部署了一个函数调用智能体。它正常运转了三轮，然后出了问题：模型尝试了一个返回 500 的工具、用户中途改变了主意、或者智能体在没有人签署的情况下决定退款订单。`while True:` 循环没有任何钩子。你无法暂停它、无法回退它、也无法分支到“如果模型当时选了另一个工具会怎样”。一旦你把这个东西部署出演示阶段，智能体就变成了一个要么成功要么失败的黑盒。

一旦你看透这一点，下一步就显而易见了。智能体本身已经是一个状态机——系统提示词加上消息历史、待处理的工具调用、以及下一步动作。把状态机显式化：节点包括“模型思考”、“工具运行”、“人工审批”，边是它们之间的条件转移。一旦图变得显式，封装就免费获得了四样东西：检查点（在步骤间保存状态）、中断（暂停等待人工）、流式（流式输出令牌和中间事件）、以及时间旅行（回退到之前的状态并尝试不同的分支）。

LangGraph 就是实现这种抽象的库。它不是 LangChain 意义上的智能体框架（“给你一个 AgentExecutor，祝你好运”）。它是一个图运行时，拥有一等公民的状态、一等公民的持久化、以及一等公民的中断。智能体循环是你画出来的，不是你手写的。

## 概念

![LangGraph StateGraph: 节点、边和检查点器](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三个部分。

1. **状态。** 一个类型化的字典（TypedDict 或 Pydantic 模型），在图里流动。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的 *reducer* 来合并这些更新——对于应该累积的列表用 `operator.add`，默认是覆盖。
2. **节点。** Python 函数 `state -> partial_state`。每个节点是一个离散步骤：“调用模型”、“运行工具”、“总结”。
3. **边。** 节点之间的转移。静态边通往固定位置。条件边接受一个路由函数 `state -> next_node_name`，这样图可以根据模型输出进行分支。

你把图编译。编译会绑定拓扑结构、附加一个检查点器（对于生产环境可选但必不可少）、并返回一个可运行对象。你用初始状态和一个 `thread_id` 来调用它。每一步执行都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四大超能力

**检查点。** 每次节点转移都会将新状态写入存储（测试时用内存，生产环境用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图即可恢复。图会从它暂停的地方继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停止。状态会被持久化。你的 API 会向用户响应“等待审批”。稍后对同一个 `thread_id` 发出带有 `Command(resume=...)` 的请求，即可恢复执行。

**流式。** `graph.stream(state, mode="updates")` 会在状态变更发生时产出增量。`mode="messages"` 会流式输出模型节点内的 LLM 令牌。`mode="values"` 会产生完整的快照。你选择要在 UI 中展示哪些内容。

**时间旅行。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任何一个之前的 `checkpoint_id` 传给 `graph.invoke`，你就可以从那个点分叉。非常适合调试（“如果模型当时选了工具 B 会怎样？”）以及用于回放生产轨迹的回归测试。

### Reducer 是关键

每个状态字段都有一个 reducer。大多数默认值就够用了——新值覆盖旧值。但消息列表需要 `operator.add` 以便新消息追加而非替换。并行边会通过 reducer 合并它们的更新。如果两个节点都更新了 `messages` 而你忘了加 `Annotated[list, add_messages]`，第二个会静默胜出，你会丢失半轮对话。Reducer 是这个库中唯一微妙的东西；搞对了它，其余部分就自然组合起来了。

### 四个节点的 ReAct 图

一个生产级的 ReAct 智能体由四个节点和两条边组成：

1. `agent` —— 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` —— 执行最后一条助手消息中的所有 tool_calls，将工具结果作为工具消息追加。
3. 从 `agent` 出发的条件边：如果最后一条消息包含 tool_calls 则路由到 `tools`，否则路由到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

仅此而已。你得到了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），带有检查点、中断和流式，大约 40 行代码。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 让一个节点可以调度并行的子图。例如：智能体决定同时查询三个检索器。每个 `Send` 会产生目标节点的一次并行执行；它们的输出通过状态 reducer 合并。这就是 LangGraph 表达编排器-工作者模式的方式，而不需要线程原语。

### 子图

一个编译好的图可以作为另一个图的一个节点。外层图只看到一个节点；内层图有自己的状态和自己的检查点。这就是团队构建监督者-工作者智能体的方式：监督者图将用户意图路由到每个领域的工作者子图。

## 动手构建

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def run_tools(state: AgentState):
    # 假设最后一条消息有 tool_calls
    tool_results = tool_executor.invoke(state["messages"][-1])
    return {"messages": tool_results}
```

`add_messages` 是让消息列表累积而非覆盖的 reducer。忘记它是 LangGraph 最常见的一个 bug。

### 步骤 2：使用线程运行

```python
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", run_tools)
builder.add_conditional_edges(
    "agent",
    lambda s: "tools" if s["messages"][-1].tool_calls else END
)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "th-1"}}
for chunk in graph.stream({"messages": [("user", "Hello")]}, config, stream_mode="updates"):
    print(chunk)
```

每次更新是一个字典 `{node_name: state_delta}`。你的前端可以流式输出这些更新到 UI，让用户看到“智能体正在思考…正在调用 search_web…得到结果…正在回答。”

### 步骤 3：添加人工介入中断

标记一个节点，在执行前暂停。

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"]  # 在工具运行前暂停
)
# 第一次调用
for chunk in graph.stream({"messages": [("user", "Approve refund")]}, config, stream_mode="updates"):
    print(chunk)  # 在 "agent" 之后停止

# 用人工输入恢复
graph.invoke(
    Command(resume="approve"),
    config
)
```

状态、检查点和线程在中断期间都持续存在。除执行期间外，内存中不保存任何内容。

### 步骤 4：用于调试的时间旅行

```python
# 获取所有检查点
history = list(graph.get_state_history(config))
checkpoint = history[2]  # 倒退到之前的步骤

# 在那个检查点处恢复（不提供新输入）
graph.invoke(None, {"configurable": {"thread_id": "th-1", "checkpoint_id": checkpoint["checkpoint_id"]}})
```

传入 `None` 作为输入会从给定的检查点开始重放；传入一个值则会将其作为该检查点状态的更新追加，然后恢复。这就是在不重跑整个对话的情况下复现一次失败的智能体运行的方式。

### 步骤 5：为生产环境更换检查点器

```python
from langgraph.checkpoint.sqlite import SqliteSaver
# 或使用 Postgres/Redis

with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = builder.compile(checkpointer=saver)
    # ... 运行
```

SQLite、Redis 和 Postgres 都是内置支持的。`MemorySaver` 仅用于测试。任何需要在重启后持久化的场景都需要真正的存储。

## 技能

> 你将智能体构建为图，而不是 `while True` 循环。

在你使用 LangGraph 之前，先花 60 秒做设计：

1. **命名节点。** 每个离散的决策或副作用操作都是一个节点。“智能体思考”、“工具运行”、“审查者批准”、“响应流式输出”。如果你不能列举它们，那这个任务还不具备智能体形态。
2. **声明状态。** 最小的 TypedDict，每个列表字段都要有一个 reducer。不要把什么都塞进 `messages`；把任务特定的字段（一个正在进行的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画出边。** 如果下一步依赖于模型输出，就用条件边。每个条件边都需要一个带有命名分支的路由函数。
4. **提前选好检查点器。** 测试用 `MemorySaver`，其他所有情况用 Postgres/Redis/SQLite。没有检查点器就不要部署——没有检查点器就意味着没有恢复、没有中断、没有时间旅行。
5. **在工具运行之前决定中断，而不是之后。** 审批应该放在进入有副作用节点的边上，这样你就可以在造成损害之前取消；验证应该放在模型输出的边上，这样你可以低成本地拒绝错误的调用。
6. **默认进行流式输出。** UI 用 `mode="updates"`，模型节点内令牌级别的流式输出用 `mode="messages"`，评估期间完整快照用 `mode="values"`。

拒绝部署一个没有检查点器的 LangGraph 智能体。拒绝部署一个在副作用*之后*才中断的智能体。拒绝部署一个 `messages` 字段没有 `add_messages` 作为 reducer 的智能体。

## 练习

1. **简单。** 实现上述四节点 ReAct 图，包含一个计算器工具和一个网络搜索工具。验证 `list(app.get_state_history(config))` 对于一轮两轮的对话至少返回四个检查点。
2. **中等。** 添加一个 `planner` 节点，它在 `agent` 之前运行，并将一个结构化的 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在检查点恢复后丢失（错误的 reducer），则测试失败。
3. **困难。** 构建一个监督者图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的状态和检查点器。在外层图上添加 `interrupt_before=["writer"]`，以便人可以批准研究简报。确认从之前的一个检查点进行时间旅行只会重新运行被分叉的那个分支。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-------------|----------|
| StateGraph | “LangGraph 的图” | 在编译之前添加节点和边的构建器对象。 |
| Reducer | “字段如何合并” | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| Thread | “一个会话 ID” | 一个 `thread_id` 字符串，用于限定一个会话的所有检查点范围。 |
| Checkpoint | “一个暂停的状态” | 节点转移后整个图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | “暂停等待人工” | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | “从之前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)` 从那个检查点开始向前重放。 |
| Send | “并行子图分发” | 节点可以返回的一个构造器，用于生成目标节点的 N 个并行执行。 |
| Subgraph | “作为节点的已编译图” | 一个已编译的 StateGraph 用作另一个图中的节点；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、reducer、检查点器和中断的权威参考。
- [LangGraph 概念：状态、reducer、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课程使用的思维模型，直接来自官方源。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细内容。
- [LangGraph 人工介入](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态模式。
- [Yao 等人，“ReAct: Synergizing Reasoning and Acting in Language Models”（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 智能体都实现的模式；阅读它以理解推理轨迹的原理。
- [Anthropic — 构建高效智能体（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 哪些图形态（链式、路由器、编排器-工作者、评估器-优化器）在何时更优。
- 阶段 11 · 09（函数调用）— 每个 LangGraph 智能体节点复用的工具调用原语。
- 阶段 11 · 14（模型上下文协议）— 通过 MCP 适配器接入 LangGraph `ToolNode` 的外部工具发现。
- 阶段 11 · 17（智能体框架权衡）— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
