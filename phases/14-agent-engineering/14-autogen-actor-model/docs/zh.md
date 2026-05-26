# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（微软研究院，2025 年 1 月）围绕 actor 模型重新设计了 agent 编排。异步消息交换、事件驱动的 agent、故障隔离、原生并发。该框架现处于维护模式，而微软 Agent 框架（2025 年 10 月公开预览）将成为后续版本。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent 循环），Phase 14 · 12（工作流模式）
**时长：** ~75 分钟

## 学习目标

- 描述 actor 模型：agent 作为 actor，消息作为唯一的 IPC，每个 actor 的故障隔离。
- 说出 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各自的作用。
- 解释为什么将消息投递与处理解耦能够实现故障隔离和原生并发。
- 在 Python 中实现一个基于标准库的 actor 运行时，并将一个双 agent 的代码审查流程移植到其上。

## 问题所在

大多数 agent 框架是同步的：一个 agent 产生，一个 agent 消费，形成调用栈。故障会导致栈崩溃。并发是事后添加的。分布式需要重写代码。

AutoGen v0.4 的答案是：actor 模型。每个 agent 是一个拥有私有收件箱的 actor。消息是唯一的交互方式。运行时将投递与处理解耦。故障隔离在单个 actor 内。并发是原生的。分布式只是不同的传输层。

## 概念

### Actors

一个 actor 包含：

- 私有状态（外部无法直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中 effects 可以是“回复”、“发送给其他 actor”、“生成新 actor”、“更新状态”、“停止自身”。

两个 actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 中的 `ConversableAgent`）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并立即返回。运行时稍后投递。由此带来三个后果：

- **故障隔离。** Agent B 崩溃不会导致 Agent A 崩溃——运行时在 B 的处理器中捕获故障，并决定如何处理（记录日志、重试、死信）。
- **原生并发。** 多条消息同时传输；actor 并发处理各自的收件箱。
- **分布式就绪。** 无论 actor 是在进程内还是在另一台主机上，收件箱 + 传输层都是相同的抽象。

### 拓扑结构

- **RoundRobinGroupChat。** Agent 按固定顺序轮流发言。
- **SelectorGroupChat。** 一个选择器 agent 根据对话上下文决定下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多 agent 团队。基于 AgentChat 构建。

### 可观测性

内置了 OpenTelemetry 支持。每条消息发出一个 span；工具调用携带 `gen_ai.*` 属性，符合 2026 年 OTel GenAI 语义约定（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 在研究和原型开发中保持稳定。微软已将活跃开发转移到微软 Agent 框架（2025 年 10 月 1 日公开预览；1.0 GA 目标 2026 年第一季度末）。AutoGen 的模式可以顺利地移植过去——actor 模型是持久的核心思想。

## 构建它

`code/main.py` 实现了一个基于标准库的 actor 运行时：

- `Message`——带类型信息的负载，包含 `sender`、`recipient`、`topic`、`body`。
- `Actor`——抽象类，包含 `receive(message, runtime)`。
- `Runtime`——带有共享队列、投递、故障隔离的事件循环。
- 一个双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行它：

```
python3 code/main.py
```

输出轨迹显示消息投递、在一个 actor 中模拟的故障不会导致另一个 actor 崩溃，以及最终收敛到共享判断。

## 使用它

- **AutoGen v0.4/v0.7**（维护模式）——适用于研究、原型开发、多 agent 模式。
- **微软 Agent 框架**（公开预览）——未来发展路径；相同的 actor 模型思想，刷新后的 API。
- **LangGraph swarm 拓扑**（第 13 课）——通过共享工具切换实现的类似模式。
- **自定义 actor 运行时**——当你需要特定的传输层（NATS、RabbitMQ、gRPC）时。

## 交付它

`outputs/skill-actor-runtime.md` 生成一个最小化的 actor 运行时以及一个团队模板（RoundRobin 或 Selector），用于给定的多 agent 任务。

## 练习

1. 添加一个死信队列：当处理器抛出异常时，将失败的消息暂存以供人工检查。在你的玩具示例中，死信队列被触发的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 根据对话状态决定谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为基于 JSON-over-HTTP 的服务器，以便 actor 可以在不同进程中运行。
4. 为每条消息连接一个 OTel span（或一个无操作替代品）。根据第 23 课，发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具示例移植到真正的 `autogen_core` API 上。你跳过了哪些在生产环境中重要的东西？

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|--------------|----------|
| Actor | “Agent” | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | “事件” | 带类型信息的负载；actor 互动的唯一方式 |
| Inbox | “邮箱” | 每个 actor 待处理消息的队列 |
| Runtime | “Agent 宿主” | 路由消息并隔离故障的事件循环 |
| Topic | “通道” | actor 之间命名的发布-订阅路由 |
| Fault isolation | “让它崩溃” | 一个 actor 失败不会导致其他 actor 崩溃 |
| RoundRobinGroupChat | “固定轮转团队” | Agent 按顺序轮流发言 |
| SelectorGroupChat | “上下文路由团队” | 选择器决定谁下一个发言 |
| Magentic-One | “参考团队” | 用于 Web + 代码 + 文件的多 agent 团队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)——图状替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——AutoGen 默认发出的 spans
