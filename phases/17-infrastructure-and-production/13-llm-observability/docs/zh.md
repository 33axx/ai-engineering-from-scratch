# 大语言模型可观测性栈的选择

> 2026年的可观测性市场分为两大类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示管理、会话回放捆绑在一起。网关/检测工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则专注于遥测。Langfuse 拥有 MIT 许可的核心代码，开源与商业的平衡做得很好（免费云服务每月 50K 事件）。Phoenix 是基于 OpenTelemetry 的原生工具，使用 Elastic License 2.0 — 在漂移/RAG 可视化方面表现出色，但不适合作为持久化的生产后端。Arize AX 采用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 领域领先，价格为 $39/用户/月，仅企业版支持自托管。Helicone 是基于代理的工具，设置只需 15-30 分钟，免费额度为每月 100K 请求，但在智能体追踪方面深度不足。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习
**语言：** Python（标准库，玩具级追踪采样模拟器）
**前置知识：** 阶段 17 · 08（推理指标），阶段 14（智能体工程）
**时长：** ~60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 对照六种主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）的许可、定价和最佳应用场景。
- 解释 OpenTelemetry 粘合模式，该模式允许你将网关工具与独立的评估平台结合使用。
- 指出 2026 年的成本差异化因素（Arize AX 的零拷贝方法与单体数据摄入的对比），并说明大约 100 倍的倍率。

## 问题

你发布了一个 LLM 功能。它运行正常。但你无法洞察提示失败、工具循环、延迟回退、成本激增或提示缓存命中率。你在谷歌上搜索"LLM 可观测性"，得到了八个工具，它们都声称以三种不同的价格解决了同一个问题。

但它们解决的不是同一个问题。LangSmith 回答"这个 LangGraph 运行为什么失败了？" Phoenix 回答"我的 RAG 流水线是否在漂移？" Helicone 回答"哪个应用在消耗令牌？" Langfuse 回答"我能自托管整个东西吗？" 不同的工具，不同的受众。

选择涉及四个维度：技术栈（LangChain？原生 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可以接受？商业许可没问题？）、预算（免费层级？$100/月？$1000/月？）、是否自托管（必须？有更好？从不？）。

## 概念

### 两个类别

**开发平台**将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑在一起。你可以运行实验，查看哪个提示有效，对旧赢家的提示进行数据集回归测试。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行检测——提示、响应、令牌、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简主义。可以通过 OpenTelemetry 与独立的评估工具结合使用。

### Langfuse — 开源与商业的平衡

- 核心代码采用 Apache / MIT 许可；通过 Docker 自托管。
- 云端免费层级：每月 50K 事件。付费：团队版 $29/月。
- 提供评估、提示管理、追踪、数据集。全面覆盖开发平台的各项功能。
- 最佳场景：你想要类似 LangSmith 的功能，但需要自托管或保持开源许可。

### Phoenix (Arize) — 遥测优先，原生 OpenTelemetry

- Elastic License 2.0；自托管非常简单。
- 在 RAG 和漂移可视化方面表现出色。嵌入空间散点图作为一等公民提供。
- 不设计为持久化生产后端——主要是开发阶段的可观测性。
- 最佳场景：RAG 流水线开发、漂移调试，可与独立的网关工具配合使用。

### Arize AX — 规模化的玩法

- 商业许可。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在规模化情况下比单体可观测性（如 Datadog 级别）便宜约 100 倍。原理：你将追踪数据存储在自己的 S3 上的 Parquet 中；Arize 直接读取。
- 最佳场景：每日超过 1000 万追踪、已有数据湖、希望获得 LLM 专属仪表板但不想承受 Datadog 的定价。

### LangSmith — LangChain/LangGraph 优先

- 商业许可，$39/用户/月。仅企业版支持自托管。
- 在 LangChain 和 LangGraph 栈中表现最佳。如果你不在这两个框架上，它的吸引力会降低。
- 最佳场景：团队已投入 LangChain，愿意付费。

### Helicone — 基于代理的最小化可行方案

- 只需 15-30 分钟设置，将你的 `OPENAI_API_BASE` 切换到 Helicone 代理即可。
- MIT 许可；每月 100K 免费请求，付费 $20/月起。
- 包含故障切换、缓存、速率限制——同时作为网关使用。
- 在智能体/多步骤追踪方面深度不足。
- 最佳场景：快速启动、单一栈应用、需要网关与可观测性合二为一。

### Opik (Comet) — 开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，继承了 Comet 的血统。
- 最佳场景：ML 团队已经使用 Comet，希望在同一个面板中查看 LLM 可观测性。

### SigNoz — 原生 OpenTelemetry 的全栈 APM

- Apache 2.0。处理通用 APM 以及通过 OpenTelemetry 接入 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年末发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。支持 OTel 的工具可以互操作。正在形成的生产模式：

1. 每次 LLM 调用都发出带有 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常监控。
3. 双发送到评估平台（Phoenix / Langfuse）用于回归测试。
4. 归档到数据湖（Iceberg）以便通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层进行检测

在你的智能体框架内部进行检测（例如添加 LangSmith 追踪）会将你与那个框架耦合。在 HTTP/OpenAI-SDK 层（通过 OpenLLMetry 或你的网关）进行检测是可移植的。

### 采样——你无法保留所有数据

当每日请求超过 100 万时，全量追踪保留的成本比 LLM 调用本身还高。按规则采样：100% 错误、100% 高成本、5% 成功。始终保持聚合数据；原始数据保留在长尾中。

### 你应该记住的数字

- Langfuse 免费云端：每月 50K 事件。
- LangSmith：$39/用户/月。
- Helicone 免费：每月 100K 请求。
- Arize AX 声称：规模化时比单体方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 使用它

`code/main.py` 模拟了一天的 100 万条追踪，涵盖不同的保留策略（100% 摄入、采样、采样+错误）。报告每种策略下的存储成本以及丢失了什么。

## 交付它

本课程产出 `outputs/skill-observability-stack.md`。根据栈、规模、预算、许可立场，选择适用的工具。

## 练习

1. 你的团队使用 LangChain，希望使用开源自托管可观测性。选择 Langfuse 或 Opik 并证明理由。
2. 在每天 500 万条追踪的情况下，Datadog 报价 $150K/月，计算 Arize AX 的盈亏平衡点。
3. 设计一组 OpenTelemetry GenAI 属性，作为你所在组织在每次 LLM 调用中必须强制执行的指南。
4. 论证仅靠 Phoenix 是否足以用于生产。它在哪些情况下不适用？
5. Helicone 带来 20ms 的代理开销。在 P99 TTFT 为 300 ms 的情况下，这是否可以接受？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| OpenLLMetry | "用于 LLM 的 OTel" | 用于 LLM 的开源 OpenTelemetry 检测 |
| GenAI conventions | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态系统捆绑的商业平台 |
| Langfuse | "开源的 LangSmith" | 具有类似功能集的 MIT 开源方案 |
| Phoenix | "Arize 开发工具" | 原生 OpenTelemetry 的开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测数据并具备网关功能的 HTTP 代理 |
| Opik | "Comet LLM" | Comet 提供的 Apache 2.0 开源开发平台 |
| Session replay | "追踪重放" | 重放包含工具调用的完整智能体会话 |
| Eval | "离线测试" | 在标注数据集上运行候选模型/提示 |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
