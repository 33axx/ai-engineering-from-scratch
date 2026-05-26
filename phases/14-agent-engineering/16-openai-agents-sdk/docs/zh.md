# OpenAI Agents SDK：交接、护栏、追踪

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多智能体框架。五大原语：智能体（Agent）、交接（Handoff）、护栏（Guardrail）、会话（Session）、追踪（Tracing）。交接被建模为名为 `transfer_to_<agent>` 的工具。护栏会在输入或输出上触发。追踪默认开启。

**类型：** 学习 + 动手构建
**语言：** Python（标准库）
**前置知识：** 阶段14·01（智能体循环），阶段14·06（工具使用）
**时间：** 约75分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释交接：为何将其建模为工具，模型看到的名称形状，以及上下文如何传递。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 使用标准库运行时实现交接 + 护栏 + 跨度（span）风格的追踪。

## 问题所在

无法清晰委派的智能体会将所有内容塞进一个提示词（prompt）中。没有护栏的智能体会泄露 PII、输出违反策略的内容，或者无限循环。OpenAI 的 SDK 将这三个原语规范化，使得多智能体工作变得可行。

## 核心概念

### 五个原语

1. **智能体（Agent）。** LLM + 指令 + 工具 + 交接。
2. **交接（Handoff）。** 委托给另一个智能体。对模型来说，它表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **护栏（Guardrail）。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）进行校验。
4. **会话（Session）。** 跨轮次自动保存对话历史。
5. **追踪（Tracing）。** 内置的跨度（span），用于 LLM 生成、工具调用、交接、护栏。

### 交接作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它意味着运行时需要：

1. 复制对话上下文（或通过 `nest_handoff_history` 测试版功能折叠上下文）。
2. 用目标智能体的指令初始化目标智能体。
3. 用目标智能体继续运行。

这本质上就是（第13课/第28课）中的监督者模式（supervisor pattern）的产品化实现。

### 护栏

三种类型：

- **输入护栏。** 在第一个智能体的输入上运行。在 LLM 调用之前拒绝不安全或超出范围的内容。
- **输出护栏。** 在最后一个智能体的输出上运行。捕获 PII 泄露、违反策略、格式错误的响应。
- **工具护栏。** 在每个函数工具上运行。校验参数、检查权限、审计执行。

模式：

- **并行（Parallel，默认）。** 护栏 LLM 与主 LLM 同时运行。降低尾延迟。如果触发，主 LLM 的工作将被丢弃（浪费 token）。
- **阻塞（Blocking，`run_in_parallel=False`）。** 护栏 LLM 先运行。如果触发，主调用不会浪费任何 token。

触发时会引发 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每次 LLM 生成、工具调用、交接和护栏都会产生一个跨度。设置环境变量 `OPENAI_AGENTS_DISABLE_TRACING=1` 可退出追踪。使用 `add_trace_processor(processor)` 可以将跨度发送到你自己的后端，同时也会发送到 OpenAI 的后端。

### 会话

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 会自动加载并追加历史。

### 这种模式可能出错的地方

- **交接漂移。** 智能体 A 交接给智能体 B，B 又交接回 A。需要添加一个跳数计数器。
- **护栏绕过。** 工具护栏只会在函数工具上触发；内置工具（文件读取器、网络获取器）需要额外的策略。
- **过度追踪。** 跨度中包含敏感内容。配合 OTel GenAI 内容捕获规则（第23课）——将内容存储在外部，通过 ID 引用。

## 动手构建

`code/main.py` 使用标准库实现了 SDK 的形状：

- `Agent`、`FunctionTool`、`Handoff`（作为具有传递语义的函数工具）。
- `Runner`，包含输入/输出/工具护栏、交接分发和跳数计数器。
- 一个简单的跨度发射器，用于展示追踪的形状。
- 一个分诊智能体（triage agent），根据用户查询交接给计费（billing）或支持（support）智能体；一个输入护栏会触发。

运行它：

```
python3 code/main.py
```

追踪结果显示两次成功的交接、一次输入护栏触发，以及一个跨度树，与真实 SDK 输出的结构一致。

## 如何使用

- **OpenAI Agents SDK** 适用于以 OpenAI 为主的产品。
- **Claude Agent SDK**（第17课）适用于以 Claude 为主的产品。
- **LangGraph**（第13课）当需要显式的状态和持久化恢复时。
- **自定义实现** 当需要精确控制（语音、多提供商、联邦部署）时。

## 交付品

`outputs/skill-agents-sdk-scaffold.md` 构建了一个 Agents SDK 应用的脚手架，包含分诊智能体、交接、输入/输出/工具护栏、会话存储以及一个追踪处理器。

## 练习

1. 添加一个交接跳数计数器：超过 N 次转移后拒绝。追踪其行为。
2. 实现 `nest_handoff_history` 作为一个选项——在转移前将之前的消息折叠成一条摘要。
3. 编写一个阻塞式输出护栏。比较在可能触发它的提示词与正常通过的提示词上的延迟。
4. 将 `add_trace_processor` 连接到 JSON 日志记录器。每个跨度会发射出什么形状？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你在哪些地方建模错了？

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| Agent（智能体） | “LLM + 指令” | SDK 中的智能体类型；拥有工具和交接 |
| Handoff（交接） | “转移” | 模型调用的工具，用于委托给另一个智能体 |
| Guardrail（护栏） | “策略检查” | 对输入/输出/工具调用的校验 |
| Tripwire（触发线） | “护栏触发” | 护栏拒绝时引发的异常 |
| Session（会话） | “历史存储” | 运行之间持久化的对话记忆 |
| Tracing（追踪） | “跨度” | 内置的可观测性，覆盖 LLM + 工具 + 交接 + 护栏 |
| Blocking guardrail（阻塞式护栏） | “顺序检查” | 护栏先运行；触发时不浪费 token |
| Parallel guardrail（并行式护栏） | “并发检查” | 护栏与主流程同时运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 原语、交接、护栏、追踪
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应实现
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时才需要使用交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK 跨度映射到的标准
