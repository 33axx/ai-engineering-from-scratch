# 构建 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world 示例。一个真正的服务器需要同时暴露工具（tools）、资源（resources）和提示（prompts），处理能力协商，发出结构化错误，并在不同 SDK 中表现一致。本课程将端到端构建一个笔记服务器：stdlib stdio 传输、JSON-RPC 分发、三种服务器原语，以及一种纯函数风格，可无缝迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型:** 构建
**语言:** Python（stdlib，stdio MCP 服务器）
**前置条件:** 阶段 13 · 06（MCP 基础）
**时长:** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息，并将响应写入 stdout。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 将 stdlib 实现迁移到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题

在你可以使用远程传输（阶段 13 · 09）或认证层（阶段 13 · 16）之前，你需要一个简洁的本地服务器。本地意味着 stdio：服务器由客户端作为子进程启动，消息通过 stdin/stdout 以换行符分隔的方式传输。

2025-11-25 规范规定，stdio 消息被编码为 JSON 对象，并带有显式的 `\n` 分隔符。这里没有 SSE；SSE 是旧的远程模式，将在 2026 年中移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用它；Keboola 于 2026 年 4 月 1 日弃用它）。对于 stdio，每行一个 JSON 对象就是整个线路格式。

笔记服务器是一个很好的示例，因为它涵盖了所有三种服务器原语。工具执行变更（`notes_create`）。资源暴露数据（`notes://{id}`）。提示提供模板（`review_note`）。本课程的结构可推广到任何领域。

## 概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 输出任何非 JSON-RPC 信封的内容。调试日志发送到 stderr。
- 每个请求都必须匹配一个携带相同 `id` 的响应。
- 通知（notification）不得被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的内容。客户端依赖能力集来开启或关闭功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

内容块（content blocks）是带类型的。最常见的类型：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形式。协议级错误（未知方法、参数错误）是 JSON-RPC 错误。工具级错误（调用有效但工具执行失败）作为 `{content: [...], isError: true}` 返回。这样模型就能在其上下文中看到失败信息。

### 实现资源

资源在设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义方案如 `notes://`。

当你将数据作为资源而非工具暴露时：

- 模型不会“调用”它；客户端可以在用户请求时将其注入上下文。
- 订阅允许服务器在资源变化时推送更新（阶段 13 · 10）。
- 阶段 13 · 14 通过 `ui://` 扩展了交互式资源。

### 实现提示

提示是带有命名参数的模板。主机将它们作为斜杠命令展示。一个 `review_note` 提示可能接收 `note_id` 参数，并生成一个多消息提示模板，客户端将其输入给模型。

### stdio 传输的微妙之处

- 换行符分隔的 JSON。无长度前缀的帧。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可以携带 `annotations` 描述安全属性：

- `readOnlyHint: true` — 纯读取，可安全重试。
- `destructiveHint: true` — 不可逆的副作用；客户端应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

客户端使用这些属性来决定用户界面（确认对话框、状态指示器）和路由（阶段 13 · 17）。

### 迁移路径

`code/main.py` 中的 stdlib 服务器大约 180 行。FastMCP（Python）将同样的逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 具有相似的形式。迁移路径是即插即用的，当你准备好时；概念（能力、分发、内容块）保持不变。

## 使用它

`code/main.py` 是一个完整的基于 stdio 的笔记 MCP 服务器，仅使用 stdlib。它处理 `initialize`、三个工具（`notes_list`、`notes_search`、`notes_create`）的 `tools/list` 和 `tools/call`、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示。你可以通过管道传递 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的地方：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个内容块列表，而不是纯字符串。
- 当执行器抛出异常时，设置 `isError: true`。

## 交付它

本课程产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能会搭建一个具有正确工具/资源/提示划分和 SDK 迁移路径的 MCP 服务器。

## 练习

1. 运行 `code/main.py`，用手工构建的 JSON-RPC 消息驱动它。练习 `notes_create`，然后通过 `resources/read` 检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会显示确认对话框（这需要一个真实的主机；Claude Desktop 可以使用）。

3. 实现 `resources/subscribe`，使得每当笔记被修改时，服务器推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件应缩减到 80 行以内。线路行为必须完全相同；使用相同的 JSON-RPC 测试工具验证。

5. 阅读规范中的 `server/tools` 部分，找出一个本课程服务器中未实现的工具定义字段。（提示：有多个；选择一个并添加它。）

## 关键术语

| 术语 | 常说的意思 | 实际含义 |
|------|------------|----------|
| MCP server | “暴露工具的东西” | 通过 stdio 或 HTTP 运行 MCP JSON-RPC 协议进程 |
| stdio transport | “子进程模型” | 服务器由客户端启动；通过 stdin/stdout 通信 |
| Dispatcher | “方法路由器” | JSON-RPC 方法名到处理函数映射 |
| Content block | “工具结果块” | 工具响应 `content` 数组中的类型化元素 |
| `isError` | “工具级失败” | 表示工具执行失败；与 JSON-RPC 错误区分 |
| Annotations | “安全提示” | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | “Python SDK” | 基于 MCP 协议的装饰器式高级框架 |
| Resource URI | “可寻址数据” | `file://`、`db://` 或自定义方案标识资源 |
| Prompt template | “斜杠命令简报” | 服务器提供的模板，带参数槽位，供主机 UI 使用 |
| Capability declaration | “功能开关” | 在 `initialize` 中声明的每个原语的标志 |

## 进一步阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行的 TS 实现
- [FastMCP — server framework](https://gofastmcp.com/) — MCP 服务器的装饰器式 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
