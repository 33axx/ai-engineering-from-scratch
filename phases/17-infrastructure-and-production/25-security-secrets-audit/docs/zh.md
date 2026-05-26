# 安全 — 机密、API 密钥轮换、审计日志、护栏

> 通过集中式凭据库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除凭据散乱。绝不在配置文件中存储凭据，也不在 VCS 中的环境变量文件或电子表格中存储。使用 IAM 角色替代静态密钥；在 CI/CD 中使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关在运行时从凭据库拉取凭据。在凭据库中轮换密钥后，所有应用在数分钟内即可获取新密钥——无需重新部署，无需在 Slack 中询问“谁有新的密钥”。轮换策略 ≤ 90 天；在每次提交中使用 TruffleHog / GitGuardian / Gitleaks 进行扫描。零信任：MFA、SSO、RBAC/ABAC、短生命周期令牌、设备合规性。PII 脱敏使用实体识别在转发前对 PHI/PII 进行掩码处理；一致的令牌化（Mesh 方法）将敏感值映射为稳定的占位符，从而让 LLM 保留代码/关系语义。网络出站：将 LLM 服务部署在专用 VPC/VNet 子网中，仅放行 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站流量。2026 年事件驱动：Vercel 供应链攻击通过被攻陷的 CI/CD 凭据泄露了数千个客户部署的环境变量。

**类型：** 学习
**语言：** Python（标准库，玩具级 PII 脱敏器 + 审计日志写入器）
**前置条件：** 阶段 17 · 19（AI 网关）、阶段 17 · 13（可观测性）
**时长：** ~60 分钟

## 学习目标

- 列举四种机密管理反模式（VCS 中的配置文件、硬编码的环境变量、电子表格、静态密钥）并说出其替代方案。
- 将 AI 网关从凭据库拉取的模式解释为 2026 年生产标准。
- 实现带有一致令牌化（相同值 → 相同占位符）的 PII 脱敏器，确保语义不被破坏。
- 说出 2026 年 Vercel 供应链事件及其对 CI/CD 凭据卫生的教训。

## 问题

一名实习生提交了包含 API 密钥的 `.env` 文件。他们很快将其删除。但密钥已经进入 git 历史——GitGuardian 扫描会捕捉到它，而你的轮换流程是“在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务。”8 小时后，一半服务已上线，另一半则在等待部署窗口。

另外，用户提示中包含“我的 SSN 是 123-45-6789”。该提示被发送至 OpenAI。你们有 BAA，但内部政策要求在转发前掩码 PII。你没有做到。

另外，你的 EKS 集群中的 LLM  Pod 可以访问任何互联网主机。有人通过向攻击者控制的域发送 DNS 查询来泄露数据。没有任何流量被阻断。

LLM 服务的安全必须解决所有三个向量：基于凭据库的凭据、PII 脱敏、网络出站过滤、审计日志。

## 概念

### 集中式凭据库 + IAM 角色拉取

**凭据库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份进行身份验证，而非使用静态密钥。凭据库在令牌生命周期内返回机密。

**AI 网关模式**：网关在请求时从凭据库拉取 `OPENAI_API_KEY`。在凭据库中轮换密钥后，下一个请求将获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、凭据库根令牌、CI/CD 凭据。尽可能自动化轮换。手动轮换需记录并跟踪。

### 机密扫描

- **TruffleHog** — 基于正则表达式 + 熵对提交进行扫描。
- **GitGuardian** — 商业产品，高精度。
- **Gitleaks** — 开源，可在 CI 中运行。

在每次提交时运行。如果检测到新机密，则阻止 PR。

### 零信任姿态

- 所有账户要求 MFA。
- 通过 SAML/OIDC 实现 SSO。
- 基于 RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问。
- 短生命周期令牌（小时，而非天）。
- 设备合规性——仅限启用了磁盘加密的企业设备。

### PII / PHI 脱敏

在提示离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业产品）。
2. 对匹配的实体进行掩码处理：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致的令牌化（Mesh 方法）：相同的值映射到相同的占位符，以便 LLM 保留关系。
4. （可选）对 LLM 响应进行反向映射。

静态正则过滤器可以捕获常见模式；NER 可以捕获更多。两者结合使用。

### 输入 + 输出护栏

输入：阻止已知的越狱提示、禁止话题；按用户进行速率限制。

输出：使用正则表达式对泄露的机密（API 密钥模式、拒绝上下文中的电子邮件模式）进行脱敏；使用分类器检测策略违规。

### 网络出站白名单

将 LLM 服务部署在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、凭据库端点。
- 其余所有流量：丢弃。
- DNS 仅通过允许列表解析器（避免 DNS 隧道泄露）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示哈希（为保护隐私，不记录原始提示）。
- 模型 + 版本。
- 令牌数。
- 成本。
- 响应哈希。
- 任何触发的护栏。

根据合规要求保留（SOC 2 保留 1 年，HIPAA 保留 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被攻陷的 CI/CD 凭据泄露了数千个客户部署的环境变量。教训：CI/CD 凭据与生产环境凭据同等重要。存放在凭据库中。严格限定范围。积极轮换。

### 你应该记住的关键数字

- 轮换策略：≤ 90 天。
- 在每次提交时扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭据被攻陷 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 使用它

`code/main.py` 实现了一个玩具级 PII 脱敏器，包含一致的令牌化和仅追加的审计日志。

## 交付它

本课程产出 `outputs/skill-llm-security-plan.md`。根据合规范围和当前状态，规划凭据库迁移、脱敏器、出站策略、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用了相同 SSN 的提示。确认两个提示获得相同的占位符。
2. 为一个部署在 EKS 上、调用 OpenAI + Anthropic + Weaviate 的 vLLM 服务设计网络出站策略。
3. 你在 git 历史中发现了一个密钥（已有 2 年历史）。正确的应对措施是什么——轮换密钥、清理历史记录，还是两者都做？说明理由。
4. 你的审计日志每天增长 10 GB。设计保留层级（热存储 30 天、温存储 12 个月、冷存储 6 年）。
5. 讨论反向令牌化（将真实值替换回 LLM 响应中）是否值得添加的复杂性，还是让占位符可见更好。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Vault | "机密存储" | 集中式凭据管理服务 |
| IAM role | "基于身份的身份验证" | 应用扮演的角色；返回短期凭据 |
| OIDC for CI/CD | "云颁发的令牌" | CI 中无静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | "机密扫描器" | 提交时检测机密 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII scrubbing | "数据掩码" | 移除或令牌化敏感实体 |
| Consistent tokenization | "稳定占位符" | 相同值 → 每次相同的令牌 |
| Mesh approach | "Mesh 令牌化" | 保留语义的令牌化模式 |
| Egress whitelist | "出站允许列表" | 仅可访问允许的域名 |
| Audit log | "不可变历史记录" | 仅追加的记录，用于合规 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
