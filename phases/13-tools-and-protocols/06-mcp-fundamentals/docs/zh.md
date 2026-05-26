# MCP 基础 —— 原语、生命周期、JSON-RPC 基础

> 在 MCP 之前，每次集成都是一次性的。由 Anthropic 于 2024 年 11 月首次发布、现由 Linux 基金会代理人工智能基金会（Agentic AI Foundation）维护的模型上下文协议（Model Context Protocol），标准化了发现与调用，使得任意客户端都能与任意服务器通信。2025-11-25 规范定义了六种原语（三种服务器端、三种客户端）、一个三阶段生命周期以及 JSON-RPC 2.0 线上格式。掌握了这些，本阶段 MCP 章节的其余内容便只是阅读。

**类型：** 学习
**语言：** Python（stdlib，JSON-RPC 解析器）
**前置条件：** 阶段 13 · 01 到 05（工具接口与函数调用）
**时间：** 约 45 分钟

## 学习目标

- 列举全部六种 MCP 原语（服务器端：工具、资源、提示；客户端：根、采样、征询）并为每个原语给出一个使用场景。
- 梳理三阶段生命周期（初始化、运行、关闭）并说明在每个阶段由谁发送哪条消息。
- 解析并生成 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 处能力协商的作用，以及缺少它会导致什么问题。

## 问题所在

在 MCP 之前，每个使用工具的智能体都有自己的协议。Cursor 有一套 MCP 形状但不兼容的工具系统。Claude Desktop 使用另一套。VS Code 的 Copilot 扩展是第三套。一个团队构建了一个“Postgres 查询”工具，却要为不同宿主 API 重复编写三次相同的代码。复用则需要复制代码。

结果是集成方案呈现寒武纪大爆发，生态发展速度受限。

MCP 通过标准化线上格式解决了这一问题。一个单一的 MCP 服务器可以在所有 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf……截至 2026 年 4 月已有超过 300 个客户端。每月 SDK 下载量达 1.1 亿次。公开服务器超过 1 万个。Linux 基金会于 2025 年 12 月在新的代理人工智能基金会（Agentic AI Foundation）下接管了项目。

本阶段使用的规范版本为 **2025-11-25**。它新增了异步任务（SEP-1686）、URL 模式征询（SEP-1036）、带工具的采样（SEP-1577）、增量范围同意（SEP-835）以及 OAuth 2.1 资源指示器语义。阶段 13 · 09 到 16 涵盖了这些扩展。本课止于基础部分。

## 概念

### 三种服务器端原语

1. **工具 (Tools)。** 可调用的动作。与阶段 13 · 01 中相同的四步循环。
2. **资源 (Resources)。** 公开的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义协议。
3. **提示 (Prompts)。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三种客户端原语

4. **根 (Roots)。** 服务器允许访问的 URI 集合。客户端声明这些 URI，服务器遵守。
5. **采样 (Sampling)。** 服务器请求客户端的模型执行一次补全操作。这使得服务器端可以运行智能体循环而无需服务器端 API 密钥。
6. **征询 (Elicitation)。** 服务器中途请求客户端的用户提供结构化输入。形式或 URL（SEP-1036）。

MCP 中的每一项能力恰好属于这六种之一。阶段 13 · 10 到 14 会对每种能力进行深入介绍。

### 线上格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求 (Requests)：`{jsonrpc: "2.0", id, method, params}`。
- 响应 (Responses)：`{jsonrpc: "2.0", id, result | error}`。
- 通知 (Notifications)：`{jsonrpc: "2.0", method, params}` —— 不含 `id`，不期望响应。

基础规范约有 15 个方法，按原语分组。重要的包括：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：初始化 (Initialize)。**

客户端发送带有其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器以其自身的 `capabilities`、`serverInfo` 以及它所支持的规范版本进行响应。当客户端消化完响应后，发送 `notifications/initialized`。此后，双方都可以根据协商好的能力发送请求。

**阶段 2：运行 (Operation)。**

双向通信。客户端调用 `tools/list` 来发现工具，然后调用 `tools/call` 来调用工具。如果服务器声明了采样能力，它可以发送 `sampling/createMessage`。当工具集发生变化时，服务器可以发送 `notifications/tools/list_changed`。当用户改变根范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：关闭 (Shutdown)。**

任一方关闭传输通道。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，阶段 13 · 09）负责传递连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是契约。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自身能力来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，则服务器不得调用 `sampling/createMessage`。对称地：如果服务器未声明 `resources.subscribe`，则客户端不得尝试订阅。

这就是防止生态分化的关键。一个不支持采样的客户端仍然是一个有效的 MCP 客户端；一个不调用 `sampling` 的服务器仍然是一个有效的 MCP 服务器。只是它们不会一起使用该功能。

### 结构化内容与错误形状

`tools/call` 返回一个 `content` 数组，包含类型化块：`text`、`image`、`resource`。阶段 13 · 14 将 MCP 应用（`ui://` 交互式 UI）添加到该列表中。

错误使用 JSON-RPC 错误码。规范定义的补充：`-32002`“资源未找到”、`-32603`“内部错误”，以及作为 `error.data` 的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见的混淆点：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否会调用特定工具是由其模型运行时决定的，而非能力标志。能力标志是规范层面的契约。模型的选择是正交的。

### 为什么用 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一种轻量级的双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（采样、通知），因此 JSON-RPC 的对称请求/响应形状非常合适。JSON-RPC 还能干净地组合到 stdio 和 WebSocket/Streamable HTTP 之上，而无需重新发明 HTTP 的请求格式。

## 使用它

`code/main.py` 提供了一个最小的 JSON-RPC 2.0 解析器和生成器，然后手动执行了 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，并打印每条消息。没有真正的传输层；只是消息格式。可与“进一步阅读”中链接的规范进行对比，验证每个信封。

值得关注的地方：

- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块数组。

## 提交作业

本课产生 `outputs/skill-mcp-handshake-tracer.md`。给定一份 MCP 客户端-服务器交互的 pcap 风格转录，该技能需标注每条消息所属的原语、生命周期阶段以及它依赖的能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生的行，并描述如果服务器未声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息格式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 过程中发出该通知，并确认客户端处理程序会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 —— 整个文档大约 80 页。找出大多数服务器不需要的一个能力标志。提示：与资源订阅有关。

4. 在纸上勾画一个假设的“cron 任务”功能属于哪种原语。（提示：服务器希望客户端在预定时间调用它。目前六种原语都不适合。）MCP 2026 年路线图中包含一个关于此功能的草案 SEP。

5. 从 GitHub 上某个开源 MCP 服务器中解析一个会话日志。统计请求、响应和通知消息的数量。计算生命周期流量与运行流量各占的比例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MCP | “模型上下文协议” | 用于模型与工具之间发现和调用的开放协议 |
| 服务器端原语 | “服务器暴露什么” | 工具（动作）、资源（数据）、提示（模板） |
| 客户端原语 | “客户端让服务器用什么” | 根（范围）、采样（LLM 回调）、征询（用户输入） |
| JSON-RPC 2.0 | “线上格式” | 对称的请求/响应/通知信封 |
| `initialize` 握手 | “能力协商” | 第一条消息对；服务器和客户端声明它们支持的功能 |
| `tools/list` | “发现” | 客户端询问服务器当前的工具集 |
| `tools/call` | “调用” | 客户端请求服务器使用参数执行一个工具 |
| `notifications/*_changed` | “变更事件” | 服务器告诉客户端其原语列表已发生变化 |
| 内容块 (Content block) | “类型化结果” | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “规范演进提案” | 命名的草案提案（例如，用于异步任务的 SEP-1686） |

## 进一步阅读

- [模型上下文协议 — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 权威规范文档
- [模型上下文协议 — 架构概念](https://modelcontextprotocol.io/docs/concepts/architecture) —— 六原语心智模型
- [Anthropic — 介绍模型上下文协议](https://www.anthropic.com/news/model-context-protocol) —— 2024 年 11 月发布博文
- [MCP 博客 — MCP 一周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 一周年回顾及 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 规范更新](https://workos.com/blog/mcp-2025-11-25-spec-update) —— SEP-1686、1036、1577、835 和 1724 的总结
