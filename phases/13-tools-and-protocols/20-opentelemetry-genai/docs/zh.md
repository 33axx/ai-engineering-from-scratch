# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用五个工具、三个 MCP 服务器和两个子智能体。你需要一个贯穿所有调用的单一追踪。OpenTelemetry GenAI 语义约定（v1.37 及以上版本的稳定属性）是 2026 年标准，原生支持 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps。本课程说明所需的属性，讲解 Span 层级（智能体 → LLM → 工具），并提供一个可直接接入任何 OTel 导出器的标准库 Span 发射器。

**类型：** 构建  
**语言：** Python（标准库，OTel Span 发射器）  
**前提条件：** 阶段 13 · 07（MCP 服务器），阶段 13 · 08（MCP 客户端）  
**时间：** 约 75 分钟

## 学习目标

- 说出 LLM Span 和工具执行 Span 所需的 OTel GenAI 属性。
- 构建一个覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的追踪层级。
- 决定捕获哪些内容（选择性加入）与屏蔽哪些内容（默认行为）。
- 将 Span 发射到本地收集器（Jaeger、Langfuse），无需重写工具代码。

## 问题

2026 年 2 月的一次调试：用户报告“我的智能体有时需要 30 秒响应，有时只需要 3 秒。”没有追踪。日志显示了 LLM 调用，但没有工具分发、MCP 服务器往返、子智能体。你只能猜测。最终发现：某个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法发现这个问题。OTel GenAI 解决了它。

这些约定在 2025-2026 年间在 OpenTelemetry 语义约定组下确定。它们定义了稳定的属性名称，因此 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 Span。只需一次插桩，即可发送到任意后端。

## 概念

### Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在一个 trace id 下。Span id 连接父子关系。

### 必需属性

根据 2025-2026 语义约定：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 提供者的响应 id，用于关联。

对于工具 Span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体 Span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供者、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 选择性内容捕获

默认情况下，Span 携带指标和时间信息，而不携带提示或完成内容。大负载和 PII 默认不包含。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定的内容捕获环境变量以包含内容。在生产环境中启用前请仔细审查。

### Span 上的事件

可以在 Span 上添加令牌级别的事件作为 Span 事件：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 Span 内按时间排序，用于详细回放。

### 导出器

OTel Span 导出到：

- **Jaeger / Tempo.** 开源，本地部署。
- **Langfuse.** 专门用于 LLM 可观测性；可视化令牌用量。
- **Arize Phoenix.** 评估 + 追踪结合。
- **Datadog.** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb.** 列式存储；查询友好。

所有导出器都支持 OTLP（线格式）。你的代码无需关心。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头部注入请求。流式 HTTP 支持标准头部。Stdio 本身不携带 HTTP 头部；规范在 2026 年路线图中讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在该功能发布之前：在每个请求的 `_meta` 中手动包含 traceparent。服务器记录 trace id。

### 指标

除了 Span，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

用于不需要每次调用细节的仪表板。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它包装流行的框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel Span。如果你的栈使用了支持的框架则很有用；否则使用手动插桩。

## 使用

`code/main.py` 以 OTel 格式的 Span 输出到 stdout（类似 OTLP-JSON 格式），模拟一个智能体调用 LLM、分发两个工具并执行一次 MCP 往返。没有真实导出器——本课程侧重于 Span 结构和属性集。将输出粘贴到兼容 OTLP 的查看器中，或者直接阅读。

需要关注：

- 所有 Span 共享同一个 trace id。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一种场景通过环境变量开启。

## 交付

本课程生成 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，技能将生成插桩方案：在何处添加 Span、填充哪些属性、以及针对哪些导出器。

## 练习

1. 运行 `code/main.py`。计算 Span 数量，并识别哪些是 CLIENT 类型，哪些是 INTERNAL 类型。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并在每次调用时作为直方图样本发出。

4. 将父智能体 Span 中的 traceparent 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出一个规范中列出但本课程代码未发出的属性，并添加它。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| OTel | “OpenTelemetry” | 追踪、指标、日志的开放标准 |
| GenAI semconv | “GenAI 语义约定” | LLM / 工具 / 智能体 Span 的稳定属性名称 |
| `gen_ai.*` | “属性命名空间” | 所有 GenAI 属性共享此前缀 |
| Span | “计时操作” | 具有开始、结束和属性的工作单元 |
| Trace | “跨 Span 的祖先关系” | 共享一个 trace id 的 Span 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 关于 Span 方向的提示 |
| OTLP | “OpenTelemetry 线格式” | 导出器的线格式 |
| Opt-in content | “提示 / 完成内容捕获” | 默认关闭；通过环境变量启用 |
| traceparent | “W3C 头部” | 在服务之间传播追踪上下文 |
| Exporter | “后端特定发送器” | 将 Span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI Span、指标和事件的规范约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 Span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级别 `invoke_agent` Span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的真实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成指南
