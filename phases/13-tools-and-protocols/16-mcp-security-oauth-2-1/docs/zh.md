# MCP 安全 II — OAuth 2.1、资源指示器、增量作用域

> 远程 MCP 服务器需要授权，而不仅仅是身份认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器 (RFC 8707) + 受保护资源元数据 (RFC 9728) 保持一致。SEP-835 通过基于 403 WWW-Authenticate 的逐步授权增加了增量作用域同意。本课将逐步授权流程实现为一个状态机，以便你观察每一个跳转。

**类型：** 构建
**语言：** Python（标准库，OAuth 状态机模拟器）
**前提条件：** 第 13 阶段 · 09（传输），第 13 阶段 · 15（安全 I）
**时长：** ~75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 走通受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理攻击。
- 实现逐步授权：服务器返回 403 并携带要求更高作用域的 WWW-Authenticate；客户端重新提示用户同意并重试。

## 问题

早期的 MCP（2025 年之前）为远程服务器提供了临时的 API 密钥甚至无认证。2025-11-25 规范通过完整的 OAuth 2.1 配置文件填补了这一空白。

三个现实需求：

- **普通远程服务器。** 用户安装一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 + PKCE 是合适的形式。
- **作用域升级。** 一个被授予 `notes:read` 的笔记服务器后续可能需要 `notes:write` 来执行某个特定操作。无需重新走完整流程，逐步授权（SEP-835）请求额外的作用域。
- **混淆代理防护。** 客户端持有一个受众限定为服务器 A 的令牌。服务器 A 是恶意的，试图将该令牌出示给服务器 B。资源指示器（RFC 8707）将令牌固定到其预期受众。

OAuth 2.1 并不新鲜。新鲜的是 MCP 的配置文件：特定的必需流程（仅授权码 + PKCE；无隐式流，默认无客户端凭证），每个令牌请求都必须包含资源指示器，以及发布受保护资源元数据以便客户端知道去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 签发令牌。可以是与资源服务器相同的服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源服务器和授权服务器可以是同一主机，但应按 URL 加以区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证验证器的哈希值与存储的挑战值是否匹配，并签发访问令牌。
6. 客户端在对资源服务器的每个请求中使用令牌：`Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数将令牌的预期受众固定下来。签发的令牌包含 `aud: "https://notes.example.com"`。另一个收到此令牌的 MCP 服务器会检查 `aud` 并拒绝。

### 作用域模型

作用域是以空格分隔的字符串。常见的 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份

作用域选择应遵循最小权限原则：只请求当前需要的，需要更多时再逐步升级。

### 逐步授权（SEP-835）

用户授予 `notes:read`。随后他们要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 `insufficient_scope` 错误，为用户显示一个针对额外作用域的同意对话框，执行一个针对该作用域的迷你 OAuth 流程，然后用新令牌重试请求。

### 令牌受众验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配则返回 401。这阻止了跨服务器令牌重用。

### 短生命周期令牌与轮换

访问令牌应为短生命周期（默认 1 小时）。刷新令牌每次刷新时轮换。客户端在后台处理静默刷新。

### 禁止令牌透传

采样服务器（第 13 阶段 · 11）不得将客户端的令牌透传给其他服务。采样请求是边界。

### 混淆代理防护

令牌绑定到 `aud`。客户端绑定到 `client_id`。每个请求都对两者进行验证。该规范明确禁止旧的“传递令牌”模式，这种模式在早期的 MCP 远程工具生态系统中很常见。

### 客户端 ID 发现

每个 MCP 客户端在其固定 URL 发布元数据。授权服务器可以获取客户端的元数据文档，以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

第 13 阶段 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，颁发给客户端的令牌是网关签发的，上游令牌从不离开网关。这翻转了信任模型——用户只需向网关认证一次；网关处理 N 个服务器授权。

## 使用它

`code/main.py` 将完整的 OAuth 2.1 逐步授权流程模拟为一个状态机。它实现了：

- PKCE 代码验证器 / 挑战值生成。
- 带有资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带有受众检查的令牌验证。
- 针对 `insufficient_scope` 的逐步升级。

本课中没有 HTTP 服务器；状态机在内存中运行，以便你可以追踪每一步。第 13 阶段 · 17 的网关课程将把它连接到实际的传输。

## 交付

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个包含工具的远程 MCP 服务器，该技能将设计作用域集合、固定规则和逐步授权策略。

## 练习

1. 运行 `code/main.py`。追踪两个作用域的逐步升级流程。注意逐步升级时哪些跳转会重复。

2. 添加刷新令牌轮换：每次刷新都签发一个新的刷新令牌并使旧令牌失效。模拟一个被盗用的刷新令牌在轮换后被使用，并确认其失败。

3. 使用 stdlib http.server 将受保护资源元数据端点实现为一个真实的 HTTP 响应。仿照第 09 课的 /mcp 端点。

4. 为一个 GitHub MCP 服务器设计作用域层次结构：读取仓库、编写 PR、批准 PR、合并 PR、管理。在每个层级之间使用逐步升级。

5. 阅读 RFC 8707 和 RFC 9728。识别 9728 中 MCP 使用方式与 RFC 示例不同的一个字段。（提示：与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| OAuth 2.1 | “现代 OAuth” | 整合后的 RFC，强制要求 PKCE 并禁止隐式流 |
| PKCE | “持有证明” | 用于防御授权码拦截的代码验证器 + 挑战值 |
| 资源指示器 | “令牌受众” | RFC 8707 的 `resource` 参数，将令牌固定到一个服务器 |
| 受保护资源元数据 | “发现文档” | RFC 9728 的 `.well-known/oauth-protected-resource` |
| 逐步授权 | “增量同意” | SEP-835 中按需添加作用域的流程 |
| `insufficient_scope` | “带 WWW-Authenticate 的 403” | 服务器信号，提示需要重新同意以获得更大作用域 |
| 混淆代理 | “跨服务令牌重用” | 受信任持有者不适当地转发令牌的攻击 |
| 短生命周期令牌 | “访问令牌 TTL” | 快速过期的持有者令牌；通过刷新令牌续期 |
| 作用域层次结构 | “最小权限栈” | 逐级递增的作用域集合，层级间通过逐步升级 |
| 客户端 ID 元数据 | “客户端发现文档” | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范的 MCP OAuth 配置文件
- [den.dev — MCP 十一月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更的详细说明
- [RFC 8707 — OAuth 2.0 的资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定的 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档的 RFC
- [Aembit — MCP OAuth 2.1、PKCE 与 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的逐步授权流程讲解
