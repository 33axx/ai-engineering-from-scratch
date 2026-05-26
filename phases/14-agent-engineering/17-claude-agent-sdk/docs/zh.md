# Claude Agent SDK：子代理与会话存储

> Claude Agent SDK 是 Claude Code 工具集的库形式。内置工具、用于上下文隔离的子代理、钩子、W3C 追踪传播、会话存储同等功能。Claude Managed Agents 是用于长期运行异步工作的托管替代方案。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**前置条件：** 阶段 14 · 01（Agent 循环），阶段 14 · 10（技能库）  
**时间：** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（工具集形态）之间的区别。
- 描述子代理——并行化与上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 使用标准库实现一个包含内置工具、可生成隔离上下文的子代理、生命周期钩子以及会话存储的工具集。

## 问题

原始 LLM API 只能完成一次往返。生产级代理需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化、追踪传播。Claude Agent SDK 将这个形态以库的形式提供——即 Claude Code 所使用的同一套工具集，专为自定义代理开放。

## 概念

### Client SDK 与 Agent SDK 对比

- **Client SDK（`anthropic`）。** 原始 Messages API。你自己控制循环、工具和状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子代理生成、会话存储。以库的形式提供 Claude Code 循环。

### 内置工具

SDK 内置了 10 多个工具：文件读写、shell、grep、glob、网页抓取等等。自定义工具通过标准的工具模式接口注册。

### 子代理

Anthropic 文档中说明了两个用途：

1. **并行化。** 并发执行独立工作。“为这 20 个模块各自查找测试文件”就是 20 个并行的子代理任务。
2. **上下文隔离。** 子代理使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子代理日志。

### 会话存储

与 TypeScript 功能对等：

- `append(session_id, message)` — 添加一个轮次。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联删除子代理会话。
- `list_subkeys(session_id)` — 列出子代理键。

`--session-mirror`（CLI 标志）在流式传输时将日志镜像到外部文件，用于调试。

### 钩子

你可以注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 门禁或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理。
- `UserPromptSubmit` — 在模型看到用户输入之前进行操作。
- `PreCompact` — 在上下文压缩之前执行。
- `Stop` — 代理退出时清理。
- `Notification` — 侧通道警报。

钩子是专业工作流（阶段 14 课程参考）以及类似系统添加横切行为的方式。

### W3C 追踪上下文

调用者上的 OTel 跨度通过 W3C 追踪上下文头传播到 CLI 子进程。整个多进程追踪在后端显示为一条追踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长期运行异步工作，内置提示缓存，内置压缩。用控制权换取托管基础设施。

### 此模式何时出错

- **子代理过度生成。** 为 100 个小任务生成 100 个子代理。开销占主导。应分批处理。
- **钩子蔓延。** 每个团队都添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话不断累积；体积增长。使用 `list_sessions` + 过期策略。

## 构建

`code/main.py` 使用标准库实现了 SDK 形态：

- `Tool`、`ToolRegistry` 包含内置的 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文，隔离运行，返回结果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — pre_tool_use、post_tool_use、session_start、session_end。
- 演示：主代理并行生成 3 个子代理（每个隔离），汇总结果，持久化会话。

运行：

```
python3 code/main.py
```

追踪显示子代理上下文隔离（编排器上下文大小保持有界）、钩子执行以及会话持久化。

## 使用

- **Claude Agent SDK** 用于以 Claude 为先的产品，希望使用 Claude Code 工具集形状。
- **Claude Managed Agents** 用于托管的长期运行异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 为先的对应物。
- **LangGraph + 自定义工具** 如果你想要图状状态机。

## 交付

`outputs/skill-claude-agent-scaffold.md` 构建了一个 Claude Agent SDK 应用的脚手架，包含子代理、钩子、会话存储、MCP 服务器挂接和 W3C 追踪传播。

## 练习

1. 添加一个子代理生成器，将 20 个任务分为每组 5 个并行子代理。测量编排器上下文大小与每个任务一个子代理的对比。
2. 实现一个 `PreToolUse` 钩子，限制 `write_file` 调用的速率（每个会话每分钟 5 次）。追踪行为。
3. 使用 `list_subkeys` 渲染子代理树。深层嵌套看起来是什么样？
4. 将玩具示例移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。何时从自托管切换到托管？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Agent SDK | “Claude Code 作为库” | 工具集形态：工具、MCP、钩子、子代理、会话存储 |
| Subagent | “子代理” | 独立上下文，自有预算；结果向上冒泡 |
| Session store | “会话数据库” | 持久化、加载、列出、删除轮次及子代理级联 |
| Hook | “生命周期回调” | 预/后工具、会话、提示提交、压缩、停止 |
| W3C trace context | “跨进程追踪” | 父跨度传播到 CLI 子进程 |
| Managed Agents | “托管工具集” | Anthropic 托管的长期运行异步工作 |
| `--session-mirror` | “日志镜像” | 在流式传输时将会话轮次写入外部文件 |
| MCP server | “工具表面” | 附加到代理的外部工具/资源源 |

## 进一步阅读

- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic，使用 Claude Agent SDK 构建代理](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应物
