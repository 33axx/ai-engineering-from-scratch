# MCP 资源和提示 —— 超越工具的场景上下文暴露

> 工具占据了 MCP 关注的 90%。但另外两个服务器原语解决了不同的问题。资源暴露数据供读取；提示以斜杠命令的形式暴露可复用的模板。许多服务器应当使用资源而非将读操作包装成工具，使用提示而非在客户端提示中硬编码工作流。本课将给出判断规则，并讲解 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库，资源 + 提示处理器）
**前置条件：** 阶段 13 · 07（MCP 服务器）
**时间：** ~45 分钟

## 学习目标

- 判断给定的领域能力应该作为工具、资源还是提示来暴露。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，带参数模板。
- 识别宿主何时将提示作为斜杠命令呈现，以及何时作为自动注入的上下文。

## 问题

一个用于笔记应用的简陋 MCP 服务器将所有功能都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这样每个数据访问都包装在由模型驱动的工具调用中。后果：

- 模型必须针对每一个可能受益于上下文的查询判断是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的“包含文件”选择器）无法展示这些数据。

正确的拆分：将数据暴露为资源，将可变或计算性的操作暴露为工具，将可复用的多步工作流暴露为提示。每个原语都有其 UX 交互方式和访问模式。

## 概念

### 工具 vs 资源 vs 提示 —— 判断规则

| 能力 | 原语 |
|------|------|
| 用户想要搜索、过滤或转换数据 | 工具 (tool) |
| 用户希望宿主将此数据作为上下文包含进来 | 资源 (resource) |
| 用户希望有一个可以重复运行的模板化工作流 | 提示 (prompt) |

指导原则：如果模型在每个相关查询时都能受益于调用它，那么它就是工具。如果用户受益于将其附加到对话中，那么它就是资源。如果整个多步骤工作流是用户希望复用的单元，那么它就是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义协议）
- `memory://session-2026-04-22/recent`（服务器特有）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码的字符串，加上 `mimeType`。

### 资源订阅

在 capabilities 中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源发生变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器，其资源是磁盘上的文件；文件监听器触发更新通知；当文件在宿主外部被编辑时，Claude Desktop 将文件重新拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 是一个补全目标。客户端可以在资源选择器中自动补全 id。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

一个提示是一个模板，它会填充成一组消息，然后宿主将其送入模型。例如，`code_review` 提示接收一个 `file_path` 参数，并返回一个包含三条消息的序列：一条系统消息、一条包含文件内容的用户消息，以及一条带有推理模板的助手启动消息。

### 宿主与提示

Claude Desktop、VS Code 和 Cursor 将提示以斜杠命令的形式暴露在聊天 UI 中。用户键入 `/code_review` 并从表单中选择参数。服务器的提示是“用户快捷方式”与“发送给模型的完整提示”之间的契约。

并非所有客户端都支持提示——请检查能力协商。如果服务器声明了提示能力但客户端不支持，那么斜杠命令将不会出现。

### “列表已更改”通知

资源和提示在集合发生变化时都会发出 `notifications/list_changed`。一个刚刚导入了 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

对于文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
对于二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
对于 MCP 应用（第 14 课）：使用 `text/html;profile=mcp-app` 配合 `ui://` URI。

### 动态资源

资源 URI 不一定对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以动态计算内容。

规则：如果客户端可以通过 URI 进行缓存，那么 URI 必须是稳定的。如果计算是一次性的，URI 应包含时间戳或随机数，以防客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主可以通过重新读取来进行轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种方式。

订阅的成本：服务器上每个会话的状态（谁订阅了什么）。保持订阅集合有界；断开的客户端应超时。

### 提示 vs 系统提示

MCP 中的提示并不是系统提示。宿主的系统提示（其自身的操作指令）和 MCP 提示（由用户调用的服务器提供模板）是并存的。行为良好的客户端绝不会让服务器提示覆盖其自己的系统提示；而是将它们分层叠加。

## 使用它

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 每条笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示，渲染成三条消息的模板。
- 一个文件监听器模拟，当笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最新的五条笔记。

运行演示以查看完整流程。

## 交付成果

本课产出 `outputs/skill-primitive-splitter.md`。对于一个拟议的 MCP 服务器，该技能将每个能力分类为工具/资源/提示，并附上理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件是否触发。

2. 添加一个 `resources/list_changed` 发射器：当新笔记创建时，发送通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个提示带有参数模式。提示正文应能在不进一步编辑的情况下直接运行。

4. 从第 07 课服务器中选取一个现有工具，判断它应该保持为工具还是拆分为资源+工具对。用一句话说明理由。

5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少被填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-------------|----------|
| 资源 (Resource) | “暴露的数据” | 宿主可读的 URI 可寻址内容 |
| 资源 URI (Resource URI) | “数据指针” | 带协议前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | “监听变化” | 客户端选择加入服务器推送更新到特定 URI |
| `notifications/resources/updated` | “资源已更改” | 向客户端发送信号，表明已订阅的资源有新内容 |
| 资源模板 (Resource template) | “参数化 URI” | 带有补全提示的 URI 模式，用于宿主选择器 |
| 提示 (Prompt) | “斜杠命令模板” | 带参数槽的命名多消息模板 |
| 提示参数 (Prompt arguments) | “模板输入” | 宿主在渲染前收集的带类型参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充后的消息列表 |
| 内容块 (Content block) | “带类型的块” | `{type: text | image | resource | ui_resource}` |
| 斜杠命令 UX (Slash-command UX) | “用户快捷键” | 宿主将提示以 `/` 开头的命令形式呈现 |

## 进一步阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) —— 资源 URI、订阅与模板
- [MCP — 概念：提示](https://modelcontextprotocol.io/docs/concepts/prompts) —— 提示模板与斜杠命令集成
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) —— 完整的 `resources/*` 消息参考
- [MCP — 服务器提示规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) —— 完整的 `prompts/*` 消息参考
- [MCP — 协议信息网站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) —— 社区指南，扩展了官方文档
