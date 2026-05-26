# MCP 网关与注册中心 —— 企业控制平面

> 企业不能让每个开发者随意安装各种 MCP 服务器。网关集中管理认证、RBAC、审计、限速、缓存及工具投毒检测，然后将合并后的工具面暴露为单个 MCP 端点。官方 MCP 注册中心（Anthropic + GitHub + PulseMCP + Microsoft，命名空间已验证）是规范的上游来源。本课将说明网关的定位，演示一个最小实现，并概述 2026 年的厂商格局。

**类型：** 学习
**语言：** Python（标准库，最小网关）
**前置知识：** 阶段 13 · 15（工具投毒），阶段 13 · 16（OAuth 2.1）
**时长：** ~45 分钟

## 学习目标

- 解释 MCP 网关的位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证、RBAC、审计、限速、策略。
- 在网关层强制实施固定工具哈希清单。
- 区分官方 MCP 注册中心与元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家财富 500 强企业拥有 30 个已批准的 MCP 服务器、5000 名开发者、合规与审计要求，以及一个希望集中实施策略的安全团队。让每位开发者在 IDE 中随意安装任意服务器是不可行的。

网关模式：

1. 网关作为单个 Streamable HTTP 端点运行，开发者连接至此。
2. 网关持有每个后端 MCP 服务器的凭据。
3. 每个开发者请求都通过网关自身的 OAuth 进行身份验证和范围限定。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用均记录以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway —— 所有厂商均在 2025-2026 年推出了网关或网关功能。

同时，官方 MCP 注册中心作为规范上游启动：经过策划、命名空间验证、使用反向 DNS 命名的服务器，网关可直接从中拉取。元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合来自多个来源的服务器。

## 概念

### 网关的五大职责

1. **认证。** OAuth 2.1 用于标识开发者；映射到用户角色。
2. **RBAC。** 按用户策略：哪些服务器、哪些工具、哪些范围。
3. **审计。** 每个调用均记录谁、做了什么、何时、结果如何。
4. **限速。** 按用户/按工具/按服务器设置上限以防止滥用。
5. **策略。** 拒绝被投毒的描述、强制执行 Two 原则、脱敏 PII。

### 网关作为单一端点

对开发者而言，网关看起来像一个 MCP 服务器。内部它路由到 N 个后端。会话 ID（阶段 13 · 09）在边界处被重写。

### 凭据保险库

开发者永远看不到后端令牌。网关持有它们（或代理到处理此事的身份提供商）。网关上的开发者若拥有 `notes:read` 权限，可传递式地使用网关自身的后端凭据访问 notes MCP 服务器 —— 但仅当策略约束该传递式访问时方可。

### 网关处的工具哈希固定

网关持有一份已批准的工具描述清单（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除任何描述已变更的工具。这是在中央层面上应用阶段 13 · 15 中的地毯式抽取防御。

### 策略即代码

高级网关通过 OPA/Rego、Kyverno 或 Styra 表达策略。诸如“用户 `alice` 只能在组织 `acme` 的仓库中调用 `github.open_pr`”之类的规则以声明方式编码。简单网关使用手写 Python 实现。两种形式均有效。

### 会话感知路由

当用户会话包含多个服务器时，网关进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个对应一个服务器。来自任何后端的通知都通过网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常在冲突时添加前缀。例如 `github.open_pr`、`notes.search`。这样路由便清晰无疑。

### 注册中心

- **官方 MCP 注册中心（`registry.modelcontextprotocol.io`）。** 由 Anthropic、GitHub、PulseMCP、Microsoft 共同推出。命名空间已验证（反向 DNS：`io.github.user/server`）。已预过滤以满足基本质量。
- **Glama。** 以搜索为中心的元注册中心，聚合多个来源。
- **MCPMarket。** 偏向商业的目录，包含供应商列表。
- **MCP.so。** 社区目录；接受公开提交。
- **Smithery。** 类似包管理器的安装流程。
- **LobeHub。** 在其 LobeChat 应用中集成了 UI 注册中心。

企业网关默认从官方注册中心拉取，允许管理员从元注册中心策划添加，并拒绝任何未固定的内容。

### 反向 DNS 命名

官方注册中心要求公开服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间可防止抢占并使信任委托更清晰。

### 2026 年 4 月厂商概览

| 厂商 | 优势 |
|------|------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层级 |
| Kong AI Gateway | 原生 K8s；细粒度策略；日志输出至 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏向 DevOps；以指标为先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可定制过滤器 |

阶段 17（生产基础设施）将更深入探讨网关运维。

## 动手实现

`code/main.py` 实现了一个约 150 行的最小网关：通过伪造的 Bearer 令牌认证用户，持有按用户的 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，实施限速，并拒绝任何描述哈希与固定清单不匹配的后端工具。

需关注的点：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 限速使用每个用户的令牌桶。
- 固定清单是以 `server::tool -> hash` 为结构的字典。

## 交付

本课生成 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 计划（用户、后端、合规要求），该技能将生成一个网关配置规范。

## 练习

1. 运行 `code/main.py`。分别以允许用户、禁止用户以及超过限速的爆发式调用进行测试。验证所有三种流程。

2. 添加一个策略，在返回结果给客户端之前脱敏 PII。对类 SSN 的字符串使用简单正则；注意缺口（电子邮件、电话号码）。

3. 扩展审计日志以输出 OpenTelemetry GenAI 跨度。阶段 13 · 20 涵盖了确切的属性。

4. 为一个 50 人的开发团队设计 RBAC 策略，包含五个后端（notes、github、postgres、jira、slack）。谁对每个后端只有读权限？谁拥有写权限？

5. 从头到尾阅读 Cloudflare 企业 MCP 博文。识别一个 Cloudflare 提供但此 stdlib 网关没有的功能。

## 关键术语

| 术语 | 大家常说的含义 | 实际含义 |
|------|----------------|----------|
| 网关 | "MCP 代理" | 位于客户端和后端之间的集中式服务器 |
| 凭据保险库 | "后端令牌留在服务器端" | 开发者永远看不到上游令牌 |
| 会话感知路由 | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| 工具哈希固定 | "已批准的清单" | 每个已批准工具描述的 SHA256；在中央层面阻止地毯式抽取 |
| RBAC | "按用户策略" | 用于工具和服务器的基于角色的访问控制 |
| 策略即代码 | "声明式规则" | 在网关处强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志 | "谁、做了什么、何时" | 用于合规的仅追加事件日志 |
| 限速 | "每用户令牌桶" | 每分钟上限以防止滥用 |
| 官方 MCP 注册中心 | "规范上游" | `registry.modelcontextprotocol.io`，命名空间已验证 |
| 反向 DNS 命名 | "注册中心命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [官方 MCP 注册中心](https://registry.modelcontextprotocol.io/) —— 规范上游，命名空间已验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) —— 包含 OAuth 和策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) —— 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) —— 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) —— IBM 的企业级网关
