# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用与模型提供商之间。核心功能包括提供商路由、回退、重试、限流、密钥引用、可观测性、护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源项目，支持 100+ 提供商，兼容 OpenAI，但在约 2000 RPS 时出现瓶颈（8 GB 内存，已发布的基准测试中存在级联故障）；最适合 Python、<500 RPS、开发/原型设计场景。**Portkey** 定位于控制平面（护栏、PII 脱敏、越狱检测、审计追踪），于 2026 年 3 月转为 Apache 2.0 开源，延迟开销 20-40 ms，生产版 $49/月。**Kong AI Gateway** 构建于 Kong Gateway 之上——Kong 在相同 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 版最多 5 个模型）；如果你已在用 Kong，则是企业级选择。**Bifrost**（Maxim AI）——自动重试，可配置退避，在 OpenAI 返回 429 时回退到 Anthropic。**Cloudflare / Vercel AI Gateways**——托管式、零运维、基础重试。数据驻留决定了是否选择自托管；Portkey 和 Kong 介于两者之间，提供开源 + 可选的托管服务。

**类型：** 学习
**语言：** Python（标准库，简易网关路由模拟器）
**先修知识：** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）
**时间：** ~60 分钟

## 学习目标

- 列举六大网关核心功能（路由、回退、重试、限流、密钥、可观测性、护栏）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）与规模上限和使用场景对应。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释为何它对 >500 RPS 场景重要。
- 根据数据驻留和运维预算选择自托管 vs 托管。

## 问题

你的产品调用了 OpenAI、Anthropic 和自托管的 Llama。每个提供商有不同的 SDK、错误模型、限流和认证方案。你需要故障转移（如果 OpenAI 返回 429，则尝试 Anthropic）、统一的凭据存储、统一的可观测性以及按租户的限流。

在应用层重新实现这一切会导致每个服务与每个提供商耦合。网关层将其整合到一个进程中，通过一个统一的 API（通常兼容 OpenAI）向外分发到各提供商。

## 概念

### 六大核心功能

1. **提供商路由** — 将 OpenAI、Anthropic、Gemini、自托管等统一在一个 API 后面。
2. **回退** — 在 429、5xx 或质量失败时，重试其他提供商。
3. **重试** — 指数退避，有限次数。
4. **限流** — 按租户、按密钥、按模型。
5. **密钥引用** — 运行时从密钥库获取凭据（绝不放在应用中）。
6. **可观测性** — OTel + GenAI 属性（阶段 17 · 13）+ 成本归属。
7. **护栏** — PII 脱敏、越狱检测、允许话题过滤器。

### LiteLLM — MIT 开源，Python

- 支持 100+ 提供商，兼容 OpenAI，路由配置，回退，基础可观测性。
- 在 Kong 基准测试中大约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下出现级联故障。
- 最佳适用场景：Python 应用，<500 RPS，开发/测试网关，实验性路由。
- 成本：开源免费；存在免费云层。

### Portkey — 控制平面定位

- 截至 2026 年 3 月为 Apache 2.0 开源。护栏、PII 脱敏、越狱检测、审计追踪。
- 每次请求延迟开销 20-40 ms。
- 生产版 $49/月，包含数据保留 + SLA。
- 最佳适用场景：需要护栏+可观测性捆绑的受监管行业。

### Kong AI Gateway — 规模之选

- 基于 Kong Gateway（成熟的 API 网关产品，lua+OpenResty）构建。
- Kong 在 12 CPU 等价环境上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 版最多 5 个。
- 最佳适用场景：已在用 Kong；>1000 RPS；愿意付费许可。

### Bifrost（Maxim AI）

- 自动重试，可配置退避。
- 在 OpenAI 返回 429 时回退到 Anthropic 是经典配置。
- 较新的参与者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管式，零运维。基础重试和可观测性。
- 最佳适用场景：基于 Cloudflare/Vercel 的边缘 JavaScript 应用。
- 在护栏和限流方面功能有限，不如 Kong/Portkey。

### 自托管 vs 托管

数据驻留是关键驱动因素。医疗和金融领域默认自托管（LiteLLM 或 Portkey 开源版或 Kong）。消费者产品默认使用托管（Cloudflare AI Gateway）或中端托管（Portkey 托管版）。混合模式：受监管租户自托管，其他租户使用托管。

### 延迟预算

- LiteLLM：典型开销 5-15 ms。
- Portkey：开销 20-40 ms。
- Kong：开销 3-8 ms。
- Cloudflare/Vercel：开销 1-3 ms（边缘优势）。

网关延迟直接增加 TTFT。对于 TTFT P99 < 100 ms 的 SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 ms，任何网关均可。

### 限流语义很重要

简单的令牌桶适用于中等规模。多租户需要滑动窗口 + 突发许可 + 按租户层级。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供层级限流。

### 网关 + 可观测性 + 路由组合

阶段 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖全部三项的工具，或仔细地组合它们：2026 年大多数部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）组合使用，实现分工。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：开销 20-40 ms；2026 年 3 月起为 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 版最多 5 个模型。
- Cloudflare/Vercel：边缘开销 1-3 ms。

## 使用它

`code/main.py` 模拟了包含回退的网关路由，在三个提供商之间注入 429/5xx 错误。报告延迟、重试率和回退命中率。

## 交付它

本课程生成 `outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规要求、延迟预算，选择一个网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI → Anthropic → 自托管的回退。在提供商错误率 5% 的情况下，预期的命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基础延迟为 300 ms。哪些网关保持在预算内？
3. 一家医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey 开源版还是 Kong？
4. 比较 LiteLLM 与 Kong：团队在什么 RPS 上限下应该迁移？
5. 为多租户 SaaS 设计一个限流策略：免费层、试用层、付费层。使用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|--------------|----------|
| 网关 | "API 代理" | 位于应用和提供商之间的进程 |
| LiteLLM | "那个 MIT 许可的" | Python 开源，支持 100+ 提供商，在 2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI 网关 | "规模型网关" | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 回退方案 |
| Cloudflare AI 网关 | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII 脱敏 | "数据清理" | 在发送给模型之前，用正则 + NER 进行遮盖 |
| 越狱检测 | "提示注入防护" | 对用户输入进行分类 |
| 审计追踪 | "受监管的日志" | 每个 LLM 调用的不可变记录 |
| 令牌桶 | "简单的限流器" | 基于补充的限流器 |
| 滑动窗口 | "精确的限流器" | 基于时间窗口的限流器，更公平 |

## 延伸阅读

- [Kong AI 网关基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 年 AI 网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年最佳 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI 网关文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
