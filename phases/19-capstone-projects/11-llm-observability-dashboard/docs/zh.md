# 顶点项目 11 — 大型语言模型可观测性与评估仪表盘

> Langfuse 转向了开放核心模式。Arize Phoenix 发布了 2026 年 GenAI 语义约定映射。Helicone 和 Braintrust 都加倍投入了按用户成本归因。Traceloop 的 OpenLLMetry 成为了事实上的 SDK 仪表化标准。生产架构是：ClickHouse 用于追踪数据，Postgres 用于元数据，Next.js 用于 UI，以及一小批运行在采样追踪数据上的评估任务（DeepEval、RAGAS、LLM-judge）。你需要构建一个自托管的仪表盘，从至少四个 SDK 家族摄取数据，并演示在五分钟内捕捉到注入的回归问题。

**类型：** 顶点项目  
**语言：** TypeScript（UI）、Python / TypeScript（摄取 + 评估）、SQL（ClickHouse）  
**前置要求：** 阶段 11（大型语言模型工程）、阶段 13（工具）、阶段 17（基础设施）、阶段 18（安全）  
**涉及的阶段：** P11 · P13 · P17 · P18  
**时间：** 25 小时  

## 问题

到了 2026 年，每个运行生产流量的 AI 团队都会在模型旁边维护一个可观测性平面。成本归因、幻觉检测、漂移监控、越狱信号、SLO 仪表盘、PII 泄露警报。开源参考实现——Langfuse、Phoenix、OpenLLMetry——都已统一到 OpenTelemetry GenAI 语义约定上作为摄取模式。现在，你可以使用一个 SDK 来仪表化 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 Span。

你将构建一个自托管的仪表盘，从至少四个 SDK 家族摄取数据，在采样追踪上运行一小批评估任务，检测漂移并发出警报。衡量标准：在故意注入一个回归（一个开始产生 PII 的提示词）后，仪表盘能在五分钟内捕捉到它并触发警报。

## 概念

摄取使用 OTLP HTTP。SDK 生成 GenAI 语义约定 Span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 进入 ClickHouse 进行列式分析；元数据（用户、会话、应用）进入 Postgres。

评估作为批处理作业在采样追踪上运行。DeepEval 对忠实度、毒性和答案相关性进行评分。RAGAS 在追踪携带检索上下文时对检索指标进行评分。自定义 LLM 评判器运行领域特定的检查（PII 泄露、违反策略的响应）。评估运行结果作为评估 Span 写回同一个 ClickHouse，并与父追踪关联。

漂移检测监控嵌入空间随时间的分布变化（对提示词嵌入进行 PSI 或 KL 散度分析）以及评估分数的趋势。警报发送给 Prometheus Alertmanager，然后转发到 Slack / PagerDuty。UI 使用 Next.js 15 和 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄取：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- 收集器：OpenTelemetry Collector，带有尾采样处理器（用于成本控制）
- 存储：ClickHouse 用于 Span，Postgres 用于元数据，S3 用于原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器包、自定义 LLM 评判器
- 漂移：每周对池化的提示词嵌入（sentence-transformers）计算 PSI / KL 散度
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + 服务端操作
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建步骤

1. **收集器配置。** OpenTelemetry Collector，配置 OTLP HTTP 接收器、尾采样器（保留 100% 的错误追踪和 10% 的成功追踪），以及导出到 ClickHouse 和 S3。

2. **ClickHouse 模式。** `spans` 表，列映射 GenAI 语义约定：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加上用于长负载的 JSON 袋。添加按 `user_id` 和 `app_id` 的二级索引。

3. **SDK 覆盖测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小型客户端应用，并使用 OpenLLMetry 自动仪表化。验证每个 SDK 都能生成规范的 GenAI Span 并落入 ClickHouse。

4. **评估任务。** 一个定时任务读取过去 15 分钟的采样追踪，并运行 DeepEval 忠实度、毒性和答案相关性测试。输出作为评估 Span 链接到父追踪。

5. **自定义 LLM 评判器。** 一个 PII 泄露评判器：给定一个响应，调用一个守卫 LLM 来评分 PII 泄露的可能性。高分的响应进入一个分诊队列。

6. **漂移检测。** 每周任务计算本周池化提示词嵌入与过去 4 周基线的 PSI。如果 PSI 超过阈值，则发出警报。

7. **仪表盘。** Next.js 15 页面：概览（每秒 Span、每个用户的成本、P95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（随时间变化的 PSI）、警报。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 将警告路由到 Slack，将严重违规路由到 PagerDuty。

9. **回归探针。** 注入一个 bug：被评估的聊天机器人开始以 1% 的概率泄露虚假社保号。测量平均修复时间（MTTR）：从 bug 部署到 Slack 发出警报。

## 使用方式

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付要求

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表盘应能摄取其追踪、运行评估、对漂移发出警报，并在 Next.js 中显示按用户成本细分。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪模式覆盖率 | 能生成规范 GenAI Span 的 SDK 家族数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与手动标注集的对比 |
| 20 | 仪表盘用户体验 | 注入回归的平均修复时间（目标：5 分钟以内） |
| 20 | 成本/规模 | 持续以每秒 1000 个 Span 的速度摄入且无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端运行 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义仪表化。验证规范 Span 带着正确的 `gen_ai.*` 属性落入 ClickHouse。

2. 在同样的追踪上，用 Phoenix 评估器替换 DeepEval。测量两个评估引擎之间的分数漂移。

3. 优化漂移检测器：按 app-id（而非全局）计算 PSI。显示每个应用的漂移轨迹。

4. 添加一个“用户影响”页面：按用户的成本、按用户的失败率，附带迷你趋势图。

5. 构建一个尾采样策略：保留 100% 毒性大于 0.5 的追踪，以及剩余追踪中 10% 的分层样本。测量引入的采样偏差。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM 属性" | 2025 年 OpenTelemetry 关于 LLM Span 属性的规范（系统、模型、令牌） |
| Tail sampling | "追踪后采样" | 收集器在追踪完成后决定保留或丢弃（可以检查错误） |
| PSI | "群体稳定性指标" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "模型评估" | 一个 LLM 根据规则对另一个 LLM 的输出进行评分（忠实度、毒性、PII） |
| Tail-sampling policy | "保留规则" | 决定保留还是丢弃追踪的规则；错误率 + 采样率 |
| Eval span | "链接的评估追踪" | 子 Span，携带评估分数，链接到原始 LLM 调用 Span |
| Cost per user | "单位经济" | 在一个时间窗口内归属于某个 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考的开放核心可观测性平台  
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 另一个参考实现，支持强大的漂移检测  
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动仪表化 SDK 家族  
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄取模式  
- [Helicone](https://www.helicone.ai) — 替代的托管可观测性服务  
- [Braintrust](https://www.braintrust.dev) — 替代的评估优先平台  
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 Span 存储  
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
