# 生产环境中的MCP认证——DCR、JWKS轮换、基于iii原语的受众固定令牌

> 第16课在内存中搭建了OAuth 2.1状态机。到2026年，你交付给真实组织的每一个MCP服务器都将位于生产认证之后：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、不会在凌晨3点破坏令牌验证的JWKS轮换，以及拒绝混乱副手复用的受众固定令牌。本课通过iii原语——`iii.registerTrigger`用于HTTP和cron、`iii.registerFunction`用于认证逻辑、`state::set/get`用于缓存密钥——来连接所有这些，使认证表面像引擎中的其他工作负载一样可观察、可重启、可重放。

**类型：** 构建
**语言：** Python（标准库，iii原语模拟用于本课环境）
**前置条件：** 阶段13 · 16（OAuth 2.1状态机），阶段13 · 17（网关）
**时间：** ~90分钟

## 学习目标

- 通过RFC 8414元数据发现授权服务器并验证契约。
- 实现RFC 7591动态客户端注册，使MCP客户端无需管理员干预即可注册。
- 使用cron触发器缓存和轮换JWKS密钥，使签名验证在密钥滚动期间仍然有效。
- 使用RFC 8707资源指示器将令牌固定到单个MCP资源，并拒绝混乱副手复用。
- 将每个端点和后台作业作为iii原语——HTTP触发器、cron触发器、命名函数和`state::*`读取——连接，以便单次重启即可重建认证表面。
- 读取IdP能力矩阵，并在IdP无法满足MCP的认证配置文件时拒绝部署。

## 问题

第16课的模拟器在内存中运行OAuth 2.1。生产环境存在三个操作空白，内存模拟器看不到它们。

第一个空白是注册。真实组织运行数百个MCP服务器和数千个MCP客户端。操作员不会手动将每个Cursor用户注册为OAuth客户端。RFC 7591动态客户端注册允许客户端向授权服务器`POST /register`并立即收到`client_id`（以及可选的`client_secret`）。服务器在其RFC 8414元数据中发布`registration_endpoint`；客户端无需带外配置即可发现它。

第二个空白是密钥轮换。JWT验证依赖于授权服务器的签名密钥，以JSON Web密钥集（JWKS）形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，在事件响应期间可能更快）。MCP服务器在启动时获取一次JWKS，在轮换窗口之前验证正常——然后每个请求都会失败，直到重启。生产环境将JWKS作为缓存值，并配合一个刷新作业在先前密钥过期前覆盖缓存，再加上缓存未命中时的回退获取，以应对使用比缓存更新的密钥签名的令牌到达的情况。

第三个空白是受众绑定。第16课介绍了RFC 8707资源指示器。在生产环境中，该指示器成为每个请求上的硬声明检查。MCP服务器将`token.aud`与其自身规范资源URL进行比较，并在不匹配时以HTTP 401拒绝。这是防止上游MCP服务器（或持有针对某个服务器的令牌的恶意客户端）在同一个信任网格中针对另一个服务器重放该令牌的唯一防御。

本课将每个空白都视为一个iii原语。元数据文档是一个返回函数输出的HTTP触发器。JWKS轮换是一个调用`auth::rotate-jwks`的cron触发器，后者写入`state::set("auth/jwks/<issuer>", ...)`。JWT验证是一个函数，其他人通过`iii.trigger("auth::validate-jwt", token)`调用。MCP服务器本身只是另一个HTTP触发器，在调度前调用验证。重启引擎：触发器注册表重建；状态存活；认证表面无需手动协调即可运行。

## 概念

### RFC 8414 — OAuth授权服务器元数据

`/.well-known/oauth-authorization-server`处的文档描述了客户端所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端获得一个MCP资源URL后可以链式发现：`oauth-protected-resource`（来自RFC 9728，资源服务器的文档）指定了发行者，然后`oauth-authorization-server`（本RFC）指定了每个端点。客户端永远不需要硬编码授权URL。

在信任IdP用于MCP之前需要验证的契约：

- `code_challenge_methods_supported`包含`S256`（RFC 7636 PKCE）。
- `grant_types_supported`包含`authorization_code`并拒绝`password`和`implicit`。
- `registration_endpoint`存在（支持RFC 7591）。
- `response_types_supported`恰好是`["code"]`（OAuth 2.1）。

如果缺少这些中的任何一个，MCP服务器将拒绝针对此IdP部署。部署清单有问题，而不是代码有问题。

### RFC 9728（回顾）— 受保护资源元数据

第16课涵盖了RFC 9728。生产环境中的差异：该文档是客户端查找*此*MCP服务器信任的授权服务器的唯一位置。单个MCP服务器可以接受来自多个IdP的令牌（一个用于员工，一个用于合作伙伴）。RFC 9728声明了该集合；RFC 8414记录了每个IdP支持的内容。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 动态客户端注册

没有DCR，每个MCP客户端（Cursor、Claude Desktop、自定义代理）都需要与IdP管理员进行带外交换。有了DCR，客户端可以提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应`client_id`和一个`registration_access_token`用于后续更新：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`是运行在用户设备上的MCP客户端的正确默认值。它们只获得一个`client_id`——没有`client_secret`可被窃取。PKCE提供了公共客户端需要的持有证明。

三个生产陷阱：

- 注册端点必须按源IP进行速率限制。没有它，恶意行为者可以编写数百万个虚假注册的脚本并耗尽`client_id`命名空间。iii使这变得简单：注册HTTP触发器在调度到注册器之前调用一个`auth::rate-limit`函数。
- 某些企业IdP需要`software_statement`（一个为客户端担保的签名JWT）。本课的模拟跳过了它；生产环境需要连接一个验证步骤，拒绝来自localhost重定向URI以外的未签名注册。
- `registration_access_token`必须存储为哈希值，而不是明文。该令牌被盗意味着攻击者可以重写客户端的重定向URI。

### RFC 8707（回顾）— 资源指示器

第16课确立了形状。生产规则：每个令牌请求包含`resource=<canonical-mcp-url>`，并且MCP服务器在每次调用时验证`token.aud`是否与其自身资源URL匹配。如果MCP服务器在`https://notes.example.com/mcp`可达，则规范URL是`https://notes.example.com`——路径组件被排除，以便单个服务器可以在一个受众下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE在OAuth 2.1中是强制性的。本课的授权码流程始终携带`code_challenge`和`code_verifier`。服务器拒绝任何没有验证器或验证器哈希值与存储的挑战不匹配的令牌请求。

### MCP规范2025-11-25认证配置文件

MCP规范（2025-11-25）精确规定了MCP服务器的授权层必须做什么：

- 发布`/.well-known/oauth-protected-resource`（RFC 9728）。
- 仅通过`Authorization: Bearer ...`接受令牌。
- 根据请求验证`aud`、`iss`、`exp`和所需作用域。
- 对于每个401和403，响应时携带包含`Bearer error=...`的`WWW-Authenticate`，并包含`scope=`和`resource=`参数（如适用）。
- 拒绝`aud`与规范资源不匹配的令牌。
- 拒绝`iss`不在受保护资源元数据的`authorization_servers`列表中的令牌。

OAuth 2.1草案是基础；RFC 8414/7591/8707/9728 + RFC 7636是表面；MCP规范是配置文件。

### IdP能力矩阵

并非所有IdP都支持完整的MCP配置文件。下表记录了截至2025-11-25规范的事实性能力声明。这是一个*部署门控*，而不是推荐。

| IdP类别 | RFC 8414元数据 | RFC 7591 DCR | RFC 8707资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管（Keycloak） | 支持 | 支持 | 支持（自24.x起） | 支持 | 本课中MCP配置文件的参考IdP；端到端支持所有RFC。 |
| 企业SSO（Microsoft Entra ID） | 支持 | 支持（高级层） | 支持 | 支持 | DCR可用性因租户层而异；在目标租户中验证后才部署。 |
| 企业SSO（Okta） | 支持 | 支持（Okta CIC / Auth0） | 支持 | 支持 | DCR在Auth0（现为Okta CIC）上可用；经典Okta组织需要管理员预注册。 |
| 社交登录IdP（通用） | 不同 | 很少 | 很少 | 支持 | 大多数社交IdP将客户端视为静态合作伙伴；不要依赖DCR。仅用作身份来源，在其上部署自己的MCP感知授权服务器。 |
| 自定义/自研 | 取决于 | 取决于 | 取决于 | 取决于 | 如果你自己发布，请发布完整的配置文件。跳过上述四个RFC中的任何一个都会破坏MCP认证契约。 |

部署清单的拒绝规则：如果选择的IdP不返回`registration_endpoint`且不在`code_challenge_methods_supported`中列出`S256`，则MCP服务器拒绝启动。没有降级模式。

### 使用iii的JWKS轮换模式

生产故障模式是过时的JWKS缓存。使用cron触发器和`state::*`缓存来解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron触发器调用`auth::rotate-jwks`，它从`<issuer>/.well-known/jwks.json`获取并写入`state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从`state::get`读取。一个`kid`在缓存中缺失的令牌会触发同步的`auth::rotate-jwks`调用作为回退。这同时处理了两种情况：计划轮换（cron）和密钥重叠窗口（同步回退）。

状态形状：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时有两个密钥是稳态。授权服务器通过引入下一个密钥（`k_2026_04`）然后在退休上一个密钥（`k_2026_03`）之前轮换，因此在旧密钥下颁发的令牌在过期前仍然有效。缓存保存并集；验证器通过`kid`选择。

### iii原语连接（本课真正关注的部分）

五个原语组成了认证表面：

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP服务器本身从不直接调用验证。它这样做：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接性就是iii的赌注。明天你可以将验证器替换为一个同时咨询两个IdP的扇出，或者添加一个跨度发射器，或者缓存阳性验证结果。MCP服务器无需更改。

### 带受众绑定的混乱副手演练

服务器A（`notes.example.com`）和服务器B（`tasks.example.com`）都向同一个授权服务器注册。服务器A被攻破。攻击者获取用户的笔记令牌并针对服务器B重放。

服务器B的验证器：

1. 解码JWT，通过`kid`获取JWKS，验证签名。
2. 检查`iss`是否在其受保护资源元数据的`authorization_servers`中。（通过——相同IdP。）
3. 检查`aud == "https://tasks.example.com"`。（失败——令牌的`aud`是`https://notes.example.com`。）
4. 返回401并附带`WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众声明是协议层对此攻击的唯一防御。为了性能跳过它是生产中最常见的错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。

### 故障模式

- **过时的JWKS。** 验证器在密钥轮换后拒绝有效令牌。修复方法是上述的cron+回退模式。永远不要在没有刷新作业的情况下缓存JWKS。
- **缺少`aud`声明。** 某些IdP默认省略`aud`，除非令牌请求中存在`resource`。验证器必须拒绝缺少`aud`的令牌，而不是将缺失视为通配符。
- **作用域升级竞争。** 同一用户的两个并发升级流程可能都成功，并产生两个具有不同作用域的访问令牌。验证器必须使用请求中呈现的令牌，而不是查找“用户的当前作用域”——这会产生TOCTOU窗口。
- **注册令牌被盗。** 泄露的`registration_access_token`使攻击者可以重写重定向URI。在存储时对其哈希；要求客户端在每次更新时提供明文；在怀疑泄露时轮换。
- **未固定`iss`。** 接受任何`iss`的验证器允许攻击者建立自己的授权服务器，为目标受众注册客户端，并颁发令牌。受保护资源元数据的`authorization_servers`列表是允许列表；强制执行它。

## 使用它

`code/main.py`使用标准库Python和一个小的`iii_mock`注册表（模拟`iii.registerFunction`、`iii.registerTrigger`、`iii.trigger`和`state::set/get`）运行完整的生产流程。流程：

1. 授权服务器在`/.well-known/oauth-authorization-server`发布RFC 8414元数据。
2. MCP客户端调用元数据端点，发现注册端点。
3. MCP客户端向`/register`（RFC 7591）提交并收到一个`client_id`。
4. MCP客户端运行PKCE保护的授权码流程（RFC 7636），并带有`resource`指示器（RFC 8707）。
5. MCP客户端使用`Authorization: Bearer ...`调用MCP服务器上的一个工具。
6. MCP服务器触发`auth::validate-jwt`，它从`state::get`读取JWKS。
7. cron触发器触发`auth::rotate-jwks`，替换状态中的JWKS。
8. 下一次调用使用新密钥进行验证，无需重启。
9. 针对不同MCP资源的混乱副手尝试得到401，并附带受众不匹配。

这里的模拟JWT使用HS256和共享密钥（以便本课仅依赖标准库）。生产环境使用RS256或EdDSA以及上述JWKS模式；验证逻辑在其他方面相同。

## 交付它

本课生成`outputs/skill-mcp-auth-iii.md`。给定MCP服务器配置和IdP能力集，该技能输出要注册的iii原语、JWKS轮换计划、作用域映射，以及在IdP不支持完整RFC配置文件时应用的拒绝规则。

## 练习

1. 运行`code/main.py`。追踪9步流程。注意`state::get`在`auth::rotate-jwks`覆盖它之前立即返回过时数据的位置，以及下一次请求如何验证新密钥。

2. 向受保护资源元数据的`authorization_servers`列表添加一个新的IdP。颁发一个由新IdP签名的令牌，并确认验证器接受它。颁发一个由未列出IdP签名的令牌，并确认验证器拒绝并返回`WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 实现`auth::rate-limit`作为一个iii函数，并在注册HTTP触发器内部的注册器运行之前调用它。使用每个源IP的令牌桶，保存在`state::set("auth/ratelimit/<ip>", ...)`中。

4. 阅读RFC 7591，并确定本课`/register`处理器未验证的两个字段。添加验证。（提示：`software_statement`和`redirect_uris`的URI方案。）

5. 阅读MCP规范2025-11-25的授权部分。找到本课验证器当前未发出的关于`WWW-Authenticate`头的一个规范性要求。添加它。

## 关键术语

| 术语 | 人们说的话 | 实际含义 |
|------|------------|----------|
| ASM | "OAuth元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程 |
| JWKS | "用于JWT验证的公钥" | JSON Web密钥集，从`jwks_uri`获取，通过`kid`索引 |
| 资源指示器 | "受众参数" | RFC 8707 `resource`参数，将令牌固定到一个服务器 |
| `aud`声明 | "受众" | 验证器与规范资源URL比较的JWT声明 |
| 混乱副手 | "令牌重放" | 攻击，其中为服务器A颁发的令牌被呈现给服务器B |
| `iss`允许列表 | "受信任的授权服务器" | 受保护资源元数据的`authorization_servers`中命名的集合 |
| 密钥轮换 | "滚动JWKS" | 定期替换签名密钥，带有重叠窗口 |
| 公共客户端 | "原生或浏览器客户端" | 没有`client_secret`的OAuth客户端；PKCE进行补偿 |
| `WWW-Authenticate` | "401/403响应头" | 携带`Bearer error=...`指令，驱动客户端恢复 |

## 进一步阅读

- [MCP — 授权规范 (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的MCP认证配置文件
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — 用于代码交换的证明密钥 (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 合并的OAuth基础
