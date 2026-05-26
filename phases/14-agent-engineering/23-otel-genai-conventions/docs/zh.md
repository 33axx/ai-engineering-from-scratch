# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI 特别兴趣小组（2024 年 4 月成立）定义了智能体遥测的标准模式。跨厂商统一了 Span 名称、属性和内容捕获规则，使得在 Datadog、Grafana、Jaeger 和 Honeycomb 中，智能体追踪具有相同的含义。

**类型：** 学习 + 动手实践
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 13（LangGraph），阶段 14 · 24（可观测性平台）
**预计用时：** ~60 分钟

## 学习目标

- 说出 GenAI span 的类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL 两类 span，以及各自适用的场景。
- 列举顶层 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获的约定：选择性加入、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题所在

每个厂商都发明自己的 span 名称。运维团队最终需要为每个框架搭建单独的仪表盘。OpenTelemetry 的 GenAI 特别兴趣小组通过定义一套整个生态系统都遵循的标准解决了这一问题。

## 概念

### Span 类别

1. **模型/客户端 span。** 覆盖原始的大语言模型调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **智能体 span。** `create_agent`（智能体构建时）和 `invoke_agent`（智能体运行时）。
3. **工具 span。** 每次工具调用一个；通过父子关系连接到智能体 span。

### 智能体 span 命名

- Span 名称：如果智能体有名称，则为 `invoke_agent {gen_ai.agent.name}`；否则回退为 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 实际解析的模型（由于路由可能与请求不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 对于 RAG：查询了哪个语料库或存储。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在技术特定的约定。

### 内容捕获

默认规则：插装默认不应捕获输入/输出。通过以下属性选择加入：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非散文内容）。这是第 27 课中嵌入了可观测性的内容污染防御机制。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。通过以下方式加入稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生地将 GenAI 属性映射到其 LLM 可观测性模式中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这种模式容易出现的问题

- **在 span 中捕获完整提示词。** 在运维人员可读的追踪中暴露了个人身份信息、机密、客户数据。应存储在外部。
- **缺少 `gen_ai.provider.name`。** 当属性丢失时，多提供商仪表盘会失效。
- **span 没有父链接。** 工具 span 变成孤儿。始终要传播上下文。
- **未设置稳定性选择加入。** 你的属性可能在后端升级时被重命名。

## 动手实践

`code/main.py` 实现了一个符合 GenAI 约定的标准库 span 发射器：

- 带有 GenAI 属性模式的 `Span`。
- 支持 `start_span` 和嵌套上下文的 `Tracer`。
- 一个脚本化的智能体运行，发射：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span、用于大语言模型调用的 `chat` span。
- 一种内容捕获模式：将提示词存储在外部，并在 span 上记录 ID。

运行它：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的 span 树，以及一个显示选择加入内容引用的“外部存储”。

## 使用它

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）——自动插装生态系统。
- **Jaeger / Honeycomb / Grafana Tempo**——原始 OTel 追踪；基于 GenAI 属性构建仪表盘。
- **自托管**——运行带有 GenAI 处理器的 OTel Collector。

## 交付物

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有的智能体，采用内容捕获默认值和外部引用存储。

## 练习

1. 为第 1 课的 ReAct 循环添加 `invoke_agent`（INTERNAL）和每个工具的 span。发送到 Jaeger 实例。
2. 以“仅引用”模式添加内容捕获：提示词存入 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入第 9 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性不会被 Collector 重命名。
5. 构建一个仪表盘：“哪些工具错误与哪些模型相关”，仅使用 GenAI 属性。

## 关键术语

| 术语 | 日常说法 | 实际含义 |
|------|----------|----------|
| GenAI SIG | “OpenTelemetry GenAI 小组” | 定义模式的 OTel 工作组 |
| invoke_agent | “智能体 span” | 表示智能体运行的 span 名称 |
| CLIENT span | “远程调用” | 对远程智能体服务的调用 span |
| INTERNAL span | “进程内” | 进程内智能体运行的 span |
| gen_ai.provider.name | “提供商” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 来源” | 检索命中的语料库/存储 |
| Content capture | “提示词日志记录” | 选择加入的消息捕获；生产环境中存储在外部 |
| Stability opt-in | “预览模式” | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范文档
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认生成 GenAI span
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内建 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
