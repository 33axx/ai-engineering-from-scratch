# MCP 传输层——stdio、可流式 HTTP 与 SSE 迁移

> stdio 仅适用于本地部署，无法用于其他场景。可流式 HTTP（2025-03-26）是远程连接的标准方案。旧的 HTTP+SSE 传输方式已被弃用，将于 2026 年中期移除。选错传输方式将付出迁移代价；选对传输方式则可获得支持远程部署、会话连续性及 DNS 重绑定防护的 MCP 服务器。

**类型：** 学习
**语言：** Python（标准库，可流式 HTTP 端点骨架）
**前置条件：** 第 13 阶段 · 07、08（MCP 服务器与客户端）
**预计时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程，单进程 vs 集群）在 stdio 与可流式 HTTP 之间做出选择。
- 实现可流式 HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制执行 `Origin` 验证与会话 ID 语义，以抵御 DNS 重绑定攻击。
- 在 2026 年中期移除截止日期前，将旧版 HTTP+SSE 服务器迁移至可流式 HTTP。

## 问题所在

首个 MCP 远程传输方案（2024-11）是 HTTP+SSE：包含两个端点，一个用于客户端的 POST 请求，另一个用于服务器向客户端推送事件的服务器发送事件（SSE）通道。该方案确实可行，但也存在缺陷：每个会话需两个端点、部分 CDN 前的缓存会失效、严重依赖某些 WAF 会激进终止的长连接。

2025-03-26 规范用可流式 HTTP 取代了它：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。此后构建或迁移的每个服务器都采用可流式 HTTP。旧版 SSE 模式正在被弃用——Atlassian Rovo 于 2026 年 6 月 30 日将其移除；Keboola 于 2026 年 4 月 1 日移除；大多数剩余企业服务器将在 2026 年底前完成移除。

而 stdio 对于本地服务器仍然重要。Claude Desktop、VS Code 以及所有 IDE 形态的客户端都通过 stdio 启动服务器。正确的思维模型是：stdio 用于“本机”，可流式 HTTP 用于“通过网络”。两者不可交叉使用。

## 概念

### stdio

- 子进程传输方式。客户端启动服务器，通过标准输入/输出进行通信。
- 每行一个 JSON 对象，以换行符分隔。
- 无需会话 ID；进程身份即代表会话。
- 无需认证（子进程继承了父进程的信任边界）。
- 切勿用于远程服务器——否则需要通过 SSH 或 socat 建立隧道，此时应直接使用可流式 HTTP。

### 可流式 HTTP

单个端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp**：客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或包含一个或多个响应的 SSE 流（适用于批量响应及与该请求相关的通知）。
- **GET /mcp**：客户端打开一个长连接的 SSE 通道。服务器利用该通道发送服务器到客户端的请求（采样、通知、启发式交互）。
- **DELETE /mcp**：客户端显式终止会话。

会话通过服务器在首次响应中设置、客户端在每次后续请求中回传的 `Mcp-Session-Id` 头来标识。会话 ID 必须为加密随机生成（至少 128 位）；出于安全考虑，服务器会拒绝客户端自行选择的 ID。

### 单端点 vs 双端点

旧规范中的双端点模式在 2026 年仍可调用——规范将其声明为“遗留兼容”。但所有新服务器都应采用单端点模式。官方 SDK 默认输出单端点；仅在与未迁移的远程服务器通信时才使用遗留模式。

### `Origin` 验证与 DNS 重绑定

浏览器目前并非 MCP 客户端，但攻击者可以构造一个网页，诱使浏览器向 `localhost:1234/mcp` 发送 POST 请求——该地址是用户本地 MCP 服务器的监听端口。如果服务器不检查 `Origin`，浏览器的同源策略将无法保护，因为 `Origin: http://evil.com` 在同源策略下属于有效的跨域来源。

2025-11-25 规范要求服务器拒绝 `Origin` 不在白名单中的请求。白名单通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）以及本地 UI 所用的 localhost 变体。

### 会话 ID 生命周期

1. 客户端在首次请求中不携带 `Mcp-Session-Id`。
2. 服务器分配一个随机 ID，在响应头中设置 `Mcp-Session-Id`。
3. 客户端在后续所有请求及 `GET /mcp` 流请求中回传该头。
4. 服务器可撤销会话；客户端在后续请求中收到 404，必须重新初始化。
5. 客户端可显式 DELETE 会话以实现干净关闭。

### 保活与重连

SSE 连接可能断开。客户端通过使用相同的 `Mcp-Session-Id` 重新发起 GET 请求来重建连接。服务器必须将在中断期间错过的事件进行排队（在合理的窗口期内），并通过客户端回传的 `last-event-id` 头进行重放。

第 13 阶段 · 13 介绍了任务（Tasks），它允许长时间运行的工作在甚至整个会话重连后仍然存活。

### 向后兼容探测

希望同时支持新旧服务器的客户端：

1. 向 `/mcp` 发送 POST 请求。
2. 如果返回 `200 OK` 且内容为 JSON 或 SSE，则为可流式 HTTP。
3. 如果返回 `200 OK` 且 `Content-Type: text/event-stream`，并伴有指向辅助端点的 `Location` 头，则为旧版 HTTP+SSE；请跟随该 `Location`。

### Cloudflare、ngrok 与托管

2026 年生产环境中的远程 MCP 服务器运行在 Cloudflare Workers（使用其 MCP Agents SDK）、Vercel Functions 或容器化 Node/Python 上。关键点：托管平台必须支持用于 SSE GET 的长连接 HTTP 连接。Vercel 的免费套餐限制为 10 秒，不适合使用。Cloudflare Workers 支持无限流。

### 网关组合

当您通过网关（第 13 阶段 · 17）前置多个 MCP 服务器时，网关是一个单一的可流式 HTTP 端点，它重写会话 ID 并将请求多路复用到上游。工具在网关层合并；客户端看到的只是一个逻辑服务器。

### 传输故障模式

- **stdio SIGPIPE**：子进程在写入过程中死亡会触发 SIGPIPE；服务器应干净退出。客户端应检测到 EOF 并将该会话标记为死亡。
- **HTTP 502 / 504**：Cloudflare、nginx 及其他代理在上游故障时会返回这些状态码。可流式 HTTP 客户端应在短暂回退后重试一次。
- **SSE 连接断开**：TCP RST、代理超时或客户端网络变化会关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **会话撤销**：服务器使会话 ID 失效；客户端在下一次请求中收到 404。客户端必须重新握手。
- **时钟偏差**：客户端的资源 TTL 计算与服务器不一致。客户端应将服务器时间戳视为权威。

### 何时绕过可流式 HTTP

部分企业在自己网络内部通过 gRPC 或消息队列传输方式部署 MCP 服务器。这并非标准做法——MCP 规范并未正式定义这些传输方式。网关可以在为 MCP 客户端暴露可流式 HTTP 接口的同时，内部使用 gRPC。保持外部接口符合规范；网关负责进行转换。

## 使用它

`code/main.py` 使用 `http.server`（标准库）实现了一个最小的可流式 HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在首次响应中设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非白名单来源的请求。该处理器复用了第 07 课笔记服务器的调度逻辑。

需要关注的重点：

- POST 处理器读取 JSON-RPC 体，进行调度，并写入 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查拒绝了默认的 `http://evil.example` 探测，但接受了 `http://localhost`。
- 会话 ID 是随机的 128 位十六进制字符串；服务器在内存中维护每个会话的状态。

## 输出产物

本课程生成 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP 服务器，该技能将生成一份迁移计划，涵盖会话 ID 连续性、Origin 检查以及向后兼容探测支持，最终迁移至可流式 HTTP。

## 练习

1. 运行 `code/main.py`。使用 `curl` 发送一个 `initialize` 的 POST 请求，观察 `Mcp-Session-Id` 响应头。发送第二个 POST 请求并回传该头，验证会话连续性。

2. 添加一个打开 SSE 流的 GET 处理器。每五秒发送一个 `notifications/progress` 事件。通过使用相同的会话 ID 重新发送 GET 请求来重连，并确认服务器接受该请求。

3. 实现 `last-event-id` 重放逻辑。在重连时，重放从该 ID 之后生成的所有事件。

4. 扩展 `Origin` 验证，支持通配符模式（`https://*.example.com`），确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中选取一个旧的 HTTP+SSE 服务器（有多个），草拟迁移方案：端点处理、会话 ID 生成和头部语义方面需要哪些更改。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| stdio 传输 | “本地子进程” | 通过 stdin/stdout 进行 JSON-RPC 通信，以换行符分隔 |
| 可流式 HTTP | “远程传输方式” | 单端点 POST + GET + 可选 SSE，基于 2025-03-26 规范 |
| HTTP+SSE | “遗留方案” | 将在 2026 年中期移除的双端点模型 |
| `Mcp-Session-Id` | “会话头” | 服务器分配的随机 ID，每次后续请求都需回传 |
| `Origin` 白名单 | “DNS 重绑定防御” | 拒绝来源不在白名单中的请求 |
| 单端点 | “一个 URL” | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 用于恢复已断开的流而不丢失事件的头 |
| 向后兼容探测 | “新旧检测” | 客户端根据响应形态检查自动选择传输方式 |
| 长连接 HTTP | “SSE 流式传输” | 服务器在单个 TCP 连接上持续推送事件数分钟或数小时 |
| 会话撤销 | “强制重新初始化” | 服务器使会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — 基本传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和可流式 HTTP 的权威参考
- [MCP — 基本传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入可流式 HTTP 的修订版
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管下的可流式 HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态下的对比
- [Atlassian — HTTP+SSE 弃用通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体的迁移截止日期示例
