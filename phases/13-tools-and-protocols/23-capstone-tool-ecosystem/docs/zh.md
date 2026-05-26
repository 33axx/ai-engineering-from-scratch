# 顶石项目 — 构建一个完整的工具生态系统

> 第13阶段讲解了每一块拼图。这个顶石项目将它们整合成一个面向生产环境的系统：一个包含工具、资源、提示和任务以及 UI 的 MCP 服务器，边缘层使用 OAuth 2.1，一个 RBAC 网关，一个多服务器客户端，一个 A2A 子代理调用，将 OTel 追踪数据送入收集器，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 捆绑包。完成本课后，你能够为每一项架构选择进行辩护。

**类型：** 构建
**语言：** Python（stdlib，端到端生态系统框架）
**先决条件：** 第13阶段 · 01 至 21
**时间：** 约120分钟

## 学习目标

- 编排一个 MCP 服务器，公开工具、资源、提示以及一个带有 `ui://` 应用的任务。
- 为该服务器配置一个 OAuth 2.1 网关，该网关强制实施 RBAC 和固定哈希。
- 编写一个多服务器客户端，使用 OTel GenAI 属性进行端到端追踪。
- 将部分工作负载委派给一个 A2A 子代理；验证不透明性得以保持。
- 将整个栈打包为 AGENTS.md + SKILL.md，以便其他代理能够驱动它。

## 问题描述

交付“研究与报告”系统：

- 用户询问：“总结2026年arXiv上关于代理协议引用最多的三篇论文。”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文总结委派给专门的写手代理；汇总结果；以 MCP Apps `ui://` 资源的形式呈现交互式报告；将每一步记录到 OTel。

第13阶段的所有基本元素都会出现。这不是一个玩具——Anthropic（Claude Research 产品）、OpenAI（带有 Apps SDK 的 GPTs）以及第三方在2026年交付的生产级研究助手系统正是这种形态。

## 概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层次结构

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个追踪 ID。每个跨度都带有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭据；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read` 和 `research:write` 权限，可以调用所有工具。`bob` 拥有 `research:read` 权限，不能调用 `generate_report`。
- 固定描述清单：丢弃任何工具哈希发生变化的服务器。
- 双重规则审计：没有工具同时结合不可信输入、敏感数据和关键操作。

### 渲染

最终的 `generate_report` 任务返回内容块以及一个 `ui://report/current` 资源。客户端的主机（如 Claude Desktop 等）在沙盒 iframe 中渲染交互式仪表板。该仪表板包含一个排序后的论文列表、引用计数以及一个按钮，该按钮对用户点击的任何论文调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下形式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动该系统。

### 每个第13阶段课程的贡献

| 课程 | 顶石项目使用的内容 |
|--------|------------------------|
| 01-05 | 工具接口、提供者可移植性、并行调用、模式、lint |
| 06-10 | MCP 基本元素、服务器、客户端、传输、资源 + 提示 |
| 11-14 | 采样、根 + 启发、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子代理委派 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 将前面课程的模式合并为一个可运行的演示。全部使用 stdlib，全部在进程中运行，以便你可以端到端阅读。它运行了研究和报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为任务、对写手的 A2A 调用、返回 ui:// 资源、发出 OTel 跨度。

需要关注的内容：

- 整个系统使用同一个追踪 ID。
- 网关策略阻止第二个用户进行写入操作。
- 任务生命周期从 working 变为 completed，并返回文本和 ui:// 内容。
- A2A 调用的内部状态对协调器是不透明的。
- AGENTS.md 和 SKILL.md 是其他代理重现工作流所需的唯一文件。

## 交付它

本课程生成 `outputs/skill-ecosystem-blueprint.md`。针对一个产品需求（研究、总结、自动化），该技能会生成完整的架构：使用哪些 MCP 基本元素、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包。

## 练习

1. 运行 `code/main.py`。注意单个追踪 ID 以及跨度如何嵌套。统计演示使用的第13阶段基本元素数量。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到同一个命名空间中。

3. 将虚假的 A2A 写手代理替换为在子进程中运行的真实代理。使用第19课框架实现。

4. 在协调器和 LLM 之间的路由网关中添加一个 PII 脱敏步骤。确认用户查询中的电子邮件被清除。

5. 为负责维护该系统的团队成员编写一个 AGENTS.md。阅读时间应少于五分钟，并为他们提供在 Cursor 或 Codex 中驱动顶石项目所需的一切。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|----------------|------------------------|
| 顶石项目 | “第13阶段集成演示” | 使用所有基本元素的端到端系统 |
| 研究和报告 | “场景” | 搜索、总结、渲染模式 |
| 生态系统 | “把所有拼图放在一起” | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 包 |
| 追踪层次结构 | “单个追踪 ID” | 每个跳点的跨度共享该追踪；通过跨度 ID 建立父子关系 |
| 网关颁发的令牌 | “传递式认证” | 客户端只看到网关的令牌；网关持有上游凭据 |
| 合并命名空间 | “所有工具在同一个扁平列表中” | 在网关处进行多服务器合并，冲突时添加前缀 |
| 不透明边界 | “A2A 调用隐藏内部细节” | 子代理的推理对协调器不可见 |
| 三层栈 | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | “多层安全” | 固定哈希、OAuth、RBAC、双重规则、审计日志 |
| 规范合规矩阵 | “我们交付的符合规范要求的内容” | 将交付物映射到2025-11-25要求的清单 |

## 进一步阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的未来方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK 概述](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级代理运行时模式
