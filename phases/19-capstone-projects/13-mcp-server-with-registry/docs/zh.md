# 顶点项目 13 — 带注册中心和治理的 MCP 服务器

> 2026 年，模型上下文协议不再只是未来，而是成为默认的工具使用规范。Anthropic、OpenAI、Google 以及所有主流 IDE 都内置了 MCP 客户端。Pinterest 发布了其内部的 MCP 服务器生态系统。AAIF 注册中心在 `.well-known` 中正式定义了能力元数据。AWS ECS 发布了参考无状态部署模式。Block 的 goose-agent 将同一协议集成到了托管助手中。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 作用域、OPA 策略门控，以及一个让平台团队能够发现、验证和启用服务器的注册中心。请完整构建这套体系。

**类型：** 顶点项目
**语言：** Python（服务器，使用 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）
**前置知识：** 阶段 11（LLM 工程）、阶段 13（工具与 MCP）、阶段 14（智能体）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**时间：** 25 小时

## 问题

MCP 已成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有托管智能体现在都使用 MCP 服务器。生产环境的挑战不再是如何编写服务器（FastMCP 使这变得简单），而是如何以企业级规模部署它们：租户级别的 OAuth 作用域、用于破坏性工具的 OPA 策略、StreamableHTTP 无状态扩展、用于发现的注册中心、每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统和 AAIF 注册中心规范设定了 2026 年的标准。

你将构建一个 MCP 服务器，暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）、一个用于平台发现的注册中心 UI，以及一个用于破坏性工具的人工审批门控。负载测试将展示 StreamableHTTP 的水平扩展能力。审计跟踪将满足企业安全审查的要求。

## 概念

MCP 2026 修订版将 StreamableHTTP 作为默认传输方式。与早期的 stdio 和 SSE 形态不同，StreamableHTTP 默认是无状态的：一个单一的 HTTP 端点接收 JSON-RPC 请求、流式返回响应，并支持用于通知的长连接。无状态意味着可以在负载均衡器后面进行水平扩展。

授权采用 OAuth 2.1，并带有每个工具的作用域。令牌携带诸如 `jira:read`、`s3:list`、`postgres:query:readonly` 等作用域。MCP 服务器在工具调用时（而非仅在会话开始时）检查作用域。对于高风险工具，服务器会拒绝任何其作用域在最近 N 分钟内未提升为 `approved:by:human` 的调用——这种提升来自 Slack 审查卡片。

注册中心是一个独立服务。每个 MCP 服务器暴露一个 `.well-known/mcp-capabilities` 文档，其中包含其工具清单、传输 URL 和认证要求。注册中心进行轮询、验证和索引。平台团队通过注册中心 UI 查看哪些工具可用、需要哪些作用域以及由哪个团队拥有。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：基于 HTTPS 的 StreamableHTTP（无状态）
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 实现工作负载身份
- 策略：每个工具的 OPA / Rego 规则；每次请求的策略决策服务
- 注册中心：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：针对破坏性工具的 Slack 交互消息
- 部署：AWS ECS Fargate 或 Fly.io，每个租户一个服务器或共享服务器但进行租户范围限定
- 审计：按租户存储的 JSONL 格式结构化日志，包含每次调用的完整链路

## 构建步骤

1. **工具表面。** 暴露 10 个内部工具：Postgres 只读查询、S3 列出对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具都有类型化模式和作用域标签。

2. **FastMCP 服务器。** 挂载这些工具。配置 StreamableHTTP 传输。添加用于 OAuth 令牌内省和作用域强制执行的中间件。

3. **OPA 策略。** 每个工具的 Rego 策略：哪些作用域允许调用、应用哪些 PII 脱敏规则、应用哪些负载大小上限。每次工具调用时调用决策服务。

4. **注册中心服务。** 一个独立的 Go 或 TS 服务，用于轮询已注册服务器的 `.well-known/mcp-capabilities`，使用 JSON Schema 进行验证，并提供一个列表/搜索/验证/启用-禁用的 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、所属团队、SLO。

6. **破坏性工具分离。** 变更状态的工具（Jira 创建、Linear 创建、Postgres 写入）运行在第二个 MCP 服务器上，并采用更严格的认证流程：令牌必须具有通过 Slack 卡片在 15 分钟内提升的 `approved:by:human` 作用域。

7. **审计日志。** 按租户的追加型 JSONL 日志：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。在写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** 在 StreamableHTTP 上模拟 100 个并发客户端。展示通过添加第二个副本进行水平扩展；演示负载均衡器在无需会话粘性的情况下重新分配请求。

9. **一致性测试。** 对两个服务器运行官方的 MCP 一致性测试套件。通过所有必选部分。

## 使用方式

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付要求

`outputs/skill-mcp-server.md` 描述交付物。一个生产级的 MCP 服务器 + 注册中心 + 审计层，用于内部工具，支持 OAuth 2.1 作用域和 OPA 门控。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 作用域强制、所有工具的 OPA 覆盖、密钥卫生 |
| 20 | 可观测性 | 每次工具调用的审计日志，含 PII 脱敏 |
| 20 | 扩展性 | 100 客户端负载测试的水平扩展演示 |
| 15 | 注册中心用户体验 | 发现/验证/启用-禁用工作流 |
| **100** | | |

## 练习

1. 添加一个新工具（Confluence 搜索）。在不触及核心服务器的情况下，通过注册中心验证流程将其发布。

2. 编写一个 OPA 策略，对包含名为 `email`、`ssn` 或 `phone` 的列的 Postgres 查询结果进行脱敏。使用探测查询进行测试。

3. 对 StreamableHTTP 与 stdio 在本地延迟方面进行基准测试。报告每次调用的 p50/p95。

4. 实现每个租户的配额：每个租户每个工具每分钟的最大调用次数 N。通过第二条 OPA 规则进行强制。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 一致性测试套件，并修复所有失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------|----------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式传输；取代网络服务器的 SSE 和 stdio |
| 能力清单 | "知名文档" | `.well-known/mcp-capabilities`，包含工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | 开放策略代理，用于根据外部规则授权工具调用 |
| 作用域提升 | "人工批准" | 通过 Slack 审批授予的短期作用域，对破坏性工具是必需的 |
| 注册中心 | "工具发现" | 从能力清单中索引 MCP 服务器的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于 OAuth 令牌颁发的加密服务身份 |
| 一致性测试套件 | "规范测试" | 用于 StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [模型上下文协议 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心
- [AAIF MCP 注册中心规范](https://github.com/modelcontextprotocol/registry) — 2026 注册中心规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP 使用](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [开放策略代理](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考
