# MCP 采样 — 服务器发起的 LLM 完成与 Agent 循环

> 大多数 MCP 服务器只是被动执行者：接收参数、运行代码、返回内容。采样让服务器能够反转方向：它请求客户端的 LLM 做出决策。这使得服务器可以在不拥有任何模型凭证的情况下托管 Agent 循环。2025-11-25 合并的 SEP-1577 在采样请求中增加了工具，使得循环可以包含更深层的推理。漂移风险提示：SEP-1577 的 tool-in-sampling 形态在 2026 年第一季度仍处于实验阶段，并在 SDK API 中逐步稳定。

**类型：** 构建  
**语言：** Python（标准库，采样运行器）  
**前置知识：** 阶段 13 · 07（MCP 服务器），阶段 13 · 10（资源和提示）  
**时间：** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无服务器端 API 密钥的服务器托管循环）。
- 实现一个服务器，要求客户端对多轮提示进行采样并返回完成结果。
- 使用 `modelPreferences`（成本/速度/智能优先级）来指导客户端模型选择。
- 构建一个 `summarize_repo` 工具，内部通过采样而非硬编码行为进行迭代。

## 问题

一个用于代码摘要工作流的实用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合生成摘要并返回。LLM 的推理发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API 密钥，服务端付费，对每个用户来说成本高昂。

方案 B：服务器返回原始内容；客户端的 Agent 进行推理。这种方法可行但将服务器逻辑移到了客户端提示中，不够健壮。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、进行几轮），而客户端保留计费和模型选择。服务器完全没有凭证。

采样就是方案 C。这是一种机制，使得受信任的服务器能够在没有完整 LLM 主机身份的情况下托管 Agent 循环。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好能力更强的模型。

外加 `hints`：服务器偏好的指定模型名称。客户端可以尊重也可以不尊重提示；客户端用户配置始终优先。

### `includeContext`

三个取值：

- `"none"` — 仅包含服务器提供的消息。默认值。
- `"thisServer"` — 包含来自此服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

自 2025-11-25 起，`includeContext` 被软弃用，因为它会泄露跨服务器上下文，存在安全隐患。推荐使用 `"none"` 并在消息中显式传递上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增功能：采样请求可以包含一个 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这使得服务器可以通过客户端的模型托管 ReAct 风格的 Agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样、如果调用工具则执行工具、再次采样、返回最终的助手消息。该功能在 2026 年第一季度之前仍处于实验阶段；SDK 签名可能仍有变化。实现时请参考 2025-11-25 规范的 client/sampling 部分。

### 人在回路中

客户端**必须**在运行采样前向用户展示服务器要求模型执行的内容。恶意服务器可能利用采样操控用户会话（“对用户说 X，这样他们就会点击 Y”）。Claude Desktop、VS Code 和 Cursor 将采样请求作为用户可拒绝的确认对话框呈现。

2026 年的共识：无人工确认的采样是危险信号。网关（阶段 13 · 17）可以自动批准低风险采样，并自动拒绝任何可疑内容。

### 无需 API 密钥的服务器托管循环

典型用例：一个没有自身 LLM 访问权限的代码摘要 MCP 服务器。它执行以下操作：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，内容为“选择最有可能描述此仓库用途的五个文件。”
3. 读取这些文件。
4. 调用 `sampling/createMessage`，内容为文件内容和“用 3 段话总结该仓库。”
5. 将摘要作为 `tools/call` 结果返回。

服务器从未接触 LLM API。客户端用户使用自己的凭证为这些完成结果付费。

### 安全风险（Unit 42 披露，2026 Q1）

- **隐蔽采样。** 一种始终调用采样，要求“从会话上下文中回复用户的电子邮件”的工具。阶段 13 · 15 涵盖了攻击向量。
- **通过采样的资源窃取。** 服务器要求客户端总结攻击者的负载，并向用户收费。
- **循环炸弹。** 服务器在紧密循环中调用采样。客户端**必须**对每个会话强制执行速率限制。

## 使用它

`code/main.py` 提供了一个模拟的服务器到客户端采样运行器。一个模拟的“summarize_repo”工具执行两轮采样（选择文件，然后摘要），模拟客户端返回预设响应。该运行器展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回完成结果。
- 服务器继续其循环。
- 速率限制器限制每次工具调用中的总采样次数。

需要注意什么：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都在采样调用中完成。
- 模型优先级影响客户端的模型选择；提示列出偏好的模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 的限制可捕获失控循环。

## 构建它

本课程产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用（研究、摘要、规划）的服务器端算法，该技能将设计一个基于采样的实现，包含合适的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断。

2. 实现 SEP-1577 的 tool-in-sampling 变体：采样请求携带一个 `tools` 数组。验证客户端循环在返回最终完成结果之前执行了这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍有变化。

3. 添加人在回路中的确认：在服务器的第一个 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回类型化的拒绝信息。

4. 添加基于客户端会话密钥的按用户速率限制器。同一用户的同服务器循环应共享一个预算。

5. 设计一个使用采样选择要包含的块的 `summarize_pdf` 工具。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时如何改变行为？

## 关键术语

| 术语 | 常说的意思 | 实际含义 |
|------|----------------|------------------------|
| 采样 | "服务器到客户端的 LLM 调用" | 服务器请求客户端的模型进行完成 |
| `sampling/createMessage` | "那个方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上名称提示 |
| `includeContext` | "跨会话泄露" | 软弃用的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 允许在采样中包含工具以实现服务器托管的 ReAct |
| 人在回路中 | "用户确认" | 客户端在运行前向用户展示采样请求 |
| 循环炸弹 | "失控采样" | 服务器端无限采样循环；客户端必须进行速率限制 |
| 隐蔽采样 | "隐藏推理" | 恶意服务器在采样提示中隐藏意图 |
| 资源窃取 | "使用用户的 LLM 预算" | 服务器强制客户端为其不希望的采样付费 |
| `stopReason` | "生成停止的原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层概述
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范的 `sampling/createMessage` 形状
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 关于采样中工具的规范演进提案（实验性）
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样和资源窃取模式
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 附带客户端代码示例的讲解
