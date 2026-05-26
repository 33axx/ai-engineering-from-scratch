# 构建 MCP 客户端 —— 发现、调用、会话管理

> 大多数 MCP 内容会提供服务器教程，并对客户端一笔带过。而客户端代码才是真正复杂的编排所在：进程启动、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课程将构建一个多服务器客户端，将三个不同的 MCP 服务器整合为一个扁平的工具有效空间供模型使用。

**类型：** 构建
**语言：** Python（标准库，多服务器 MCP 客户端）
**前置条件：** 阶段 13 · 07（构建 MCP 服务器）
**预计时间：** ~75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器会话的状态（能力、工具列表、最后看到的通知 ID）。
- 跨多个服务器合并工具列表到一个命名空间，并处理冲突。
- 将工具调用路由到拥有该工具的服务器，并重新组装响应。

## 问题所在

一个真实的智能体宿主（Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行一个文件系统服务器、一个 Postgres 服务器和一个 GitHub 服务器。客户端的工作是：

1. 启动每个服务器。
2. 与每个服务器独立进行握手。
3. 在每个服务器上调用 `tools/list` 并将结果扁平化。
4. 当模型发出 `notes_search` 时，在合并后的命名空间中查找，并路由到正确的服务器。
5. 处理来自任何服务器的通知（`tools/list_changed`），而不阻塞。
6. 在传输失败时重新连接。

亲手完成所有这些工作，正是区分“玩具”和“可用产品”的关键所在。官方 SDK 封装了这些，但心智模型必须是你自己的。

## 概念

### 子进程启动

`subprocess.Popen`，设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。将 `bufsize=1` 并启用文本模式以实现逐行读取。每个服务器是一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每个服务器的会话状态

每个服务器对应一个 `Session` 对象，其中包含：

- `process` —— Popen 句柄。
- `capabilities` —— 服务器在 `initialize` 时声明的能力。
- `tools` —— 上次 `tools/list` 的结果。
- `pending` —— 请求 ID 到等待响应的 Promise/Future 的映射。

请求本质上是异步的；在服务器 B 进行调用期间，向服务器 A 发送的 `tools/call` 不能阻塞。要么使用线程加队列，要么使用 asyncio。

### 合并的命名空间

当客户端看到聚合的工具列表时，名称可能会冲突。两个服务器可能都公开了 `search`。客户端有三种选择：

1. **按服务器名称添加前缀。** `notes/search`、`files/search`。清晰但略显笨拙。
2. **静默先到先得。** 后启动的服务器的 `search` 会覆盖先前的。有风险；会隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器；通知用户。对于安全敏感的宿主来说最安全。

Claude Desktop 使用按服务器名称添加前缀。Cursor 使用冲突拒绝并给出明确错误。VS Code MCP 也采用按服务器名称添加前缀。

### 路由

合并后，一个调度表将 `tool_name` 映射到 `session`。模型按名称发出调用；客户端找到会话，并向该服务器的 stdin 写入一条 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端必须：

1. 在该采样解决之前阻塞对该服务器的进一步请求，或者如果其实现支持并发则进行流水线处理。
2. 调用其 LLM 提供者。
3. 将响应发送回服务器。

第 11 课会完整介绍采样。本课程仅做存根以保持完整性。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用则重新读取。通知不得产生响应 —— 不要尝试确认它们。

一个常见的客户端错误：在 `tools/call` 上阻塞读取循环，而通知却停留在流中。使用后台读取线程，将所有消息推入队列；主线程从队列中取出并分发。

### 重连

传输可能失败：服务器崩溃、操作系统杀死进程、stdio 管道断开。客户端检测到 stdout 上的 EOF，并将该会话视为死亡。选项：

- 静默重启服务器并重新握手。对于纯只读服务器可以接受。
- 向用户报告失败。对于有用户可见会话的状态服务器可以接受。

第 13 课第 9 节涵盖了 Streamable HTTP 的重连语义；stdio 更简单。

### 保活与会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头部。Stdio 没有会话 ID —— 进程身份本身就是会话。保活 ping 是可选的；stdio 管道不会因不活动而断开。

## 使用它

`code/main.py` 将三个模拟的 MCP 服务器作为子进程启动，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。这些“服务器”实际上是运行玩具响应器的其他 Python 进程（没有真正的 LLM）。运行它可以看到：

- 三次初始化，每次都有各自的能力集。
- 三份 `tools/list` 结果合并成一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀方式防止冲突。

要点：

- `Session` 数据类干净地保存了每个服务器的状态。
- 后台读取线程在不阻塞主线程的情况下，逐行从 stdout 取出所有行。
- 调度表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明相同的名称时，后一个会被重命名并加上前缀。

## 交付产物

本课程产生 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能将生成一个启动它们、合并工具列表、并提供带有冲突解决机制的路由函数的工具。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。用 SIGTERM 杀死其中一个模拟服务器进程，观察客户端如何检测到 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用是否路由正确。

3. 为服务器重启添加连接池风格的退避策略：连续失败时指数退避，上限 30 秒，失败三次后向用户发送通知。

4. 草拟一个支持 100 个并发 MCP 服务器的客户端。什么数据结构取代简单的调度字典？（提示：用于前缀命名空间的字典树，再加上每个服务器的工具计数指标。）

5. 将客户端移植到官方的 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从大约 200 行缩减到大约 40 行，同时保留多服务器路由功能。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| MCP 客户端 | “智能体宿主” | 启动服务并编排工具调用的进程 |
| 会话 | “每个服务器的状态” | 能力、工具列表、待处理请求的记账信息 |
| 合并命名空间 | “一个工具列表” | 所有活跃服务器上工具名称的扁平集合 |
| 命名空间冲突 | “两个服务器有相同的工具” | 客户端必须添加前缀、拒绝或先到先得处理重复 |
| 路由 | “这个调用该谁处理？” | 从工具名称分发到所属服务器 |
| 后台读取器 | “非阻塞 stdout” | 从服务器 stdout 中提取数据到队列的线程或任务 |
| 采样回调 | “LLM 即服务” | 客户端处理来自服务器的 `sampling/createMessage` 的处理程序 |
| `notifications/*_changed` | “原语已变异” | 表示客户端必须重新发现或重新读取的信号 |
| 重连策略 | “服务器挂了怎么办” | 传输失败时的重启语义 |
| Stdio 会话 | “进程 = 会话” | 无会话 ID；子进程生命周期即为会话 |

## 延伸阅读

- [Model Context Protocol — 客户端规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范的客户端行为
- [MCP — 客户端快速入门指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 Hello World 客户端教程
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 对等实现
- [VS Code — 扩展中的 MCP](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
