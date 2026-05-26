# 异步任务（SEP-1686）——立即调用，稍后获取，用于长时间运行的工作

> 实际的智能体工作需要几分钟到几小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时或阻塞 UI。SEP-1686（已于 2025-11-25 合并）添加了任务（Tasks）原语：任何请求都可以扩展为任务，结果可以稍后获取，或通过状态通知流式传输。漂移风险提示：任务在 2026 年上半年仍为实验性；SDK 表面仍在围绕规范设计。

**类型：** 构建  
**语言：** Python（标准库，异步任务状态机）  
**先修课程：** 阶段 13 · 07（MCP 服务器），阶段 13 · 09（传输）  
**时间：** ~75 分钟

## 学习目标

- 识别何时将工具从同步提升为任务增强（服务器端工作 > 30 秒）。
- 遍历任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，使崩溃不会丢失正在进行的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

`generate_report` 工具运行一个耗时数分钟的提取管道。同步模型下的选项：

1. 保持连接打开三分钟。远程传输会断开连接；客户端超时；UI 冻结。
2. 立即返回一个占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 发送后不管；没有结果。

以上都不好。SEP-1686 添加了第四个选项：任务增强。任何请求（通常是 `tools/call`）都可以标记为任务。服务器立即返回一个任务 ID。客户端在完成后轮询 `tasks/status` 并获取 `tasks/result`。服务器端状态在重启后仍然存在。

## 概念

### 任务增强

请求通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定）成为任务。服务器立即返回：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；超过 ttl 后任务结果将被丢弃。

### 每个工具的选择加入

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` —— 该工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` —— 客户端可以请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

`generate_report` 工具应为 `required`。`notes_search` 工具应为 `forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是仅追加的：一旦进入 `completed`、`failed` 或 `cancelled`，任务即为终态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和进度提示。
- `tasks/result {taskId}` —— 阻塞或返回 404（如果尚未完成）。
- `tasks/cancel {taskId}` —— 幂等；终态忽略。
- `tasks/list` —— 可选；枚举活跃和最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

采用流式而非轮询的客户端获得更好的用户体验。轮询始终作为最小表面得到支持。

### 持久化状态

规范要求声明任务支持的服务器必须持久化状态。崩溃不应在 ttl 内丢失已完成的结果。存储方式从 SQLite 到 Redis 再到文件系统。课程 13 的脚手架使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器尝试停止（检查执行器协作取消）。如果任务已经是终态，则请求为无操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将进程已死亡且状态为 `working` 的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在其 ttl 内保留 `completed` / `failed` / `cancelled` 状态。

### 异步任务与采样

任务本身可以调用 `sampling/createMessage`。这就是长时间运行的研究任务的工作方式：服务器任务线程根据需要采样客户端模型，而客户端 UI 将任务显示为 `working`，并带有定期进度更新。

### 为什么这是实验性的

SEP-1686 于 2025-11-25 发布，但更广泛的路线图指出了三个未解决的问题：持久化订阅原语、子任务（父子任务关系）以及结果 TTL 标准化。预计规范将在 2026 年继续演进。生产代码应将任务视为仅在常见情况下稳定，并针对子任务未来的 SDK 变更做好防护。

## 使用它

`code/main.py` 实现了一个持久化任务存储（基于文件系统）和一个 `generate_report` 工具，该工具在后台线程中运行。客户端调用工具，立即获得任务 ID，在工作线程更新进度时轮询 `tasks/status`，完成后获取 `tasks/result`。取消功能有效；崩溃恢复通过终止工作线程并重新加载状态来模拟。

需要注意的内容：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示进度推进。
- 客户端的取消设置了一个事件；工作线程检查并提前退出。
- “崩溃”后的状态重载将进行中的任务标记为 `failed`，并带有 `CRASH_RECOVERY`。

## 交付它

本课程产生 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（研究、构建、导出），该技能设计任务存储（状态形状、ttl、持久性），选择合适的 taskSupport 标志，并勾勒进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中添加 `tasks/cancel` 调用。验证工作线程响应它，状态变为 `cancelled`。

3. 模拟崩溃恢复：终止工作线程，重启加载器，观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展到 SQLite。持久性优势相同；查询选项开放（列出会话 X 中的所有任务）。

5. 阅读 MCP 2026 路线图文章。确定一个最可能在未来一年内影响 SDK API 设计的与任务相关的未解决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Task | “长时间运行的工具调用” | 请求通过 `_meta.task` 增强以实现异步执行 |
| SEP-1686 | “任务规范” | 规范演进提案，于 2025-11-25 添加任务 |
| `_meta.task` | “任务信封” | 每个请求的元数据，包含 id、state、ttl |
| taskSupport | “工具标志” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “轮询方法” | 获取当前状态和可选的进度提示 |
| `tasks/result` | “获取结果” | 返回完成的载荷，如果尚未完成则返回 404 |
| `tasks/cancel` | “停止它” | 幂等的取消请求 |
| ttl | “保留预算” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 服务器发起的状态变更事件 |
| Durable store | “崩溃安全状态” | 文件系统 / SQLite / Redis 持久化层 |

## 扩展阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) —— 原始提案及完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) —— 设计讲解及原理
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) —— 机制和状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) —— SDK 级别的任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 未解决问题及 2026 年优先级，包括子任务
