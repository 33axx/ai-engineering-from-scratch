# Agent 可观测性：Langfuse、Phoenix、Opik

> 三大开源 Agent 可观测性平台主导 2026 年。Langfuse（MIT）—— 月安装量 600 万+，涵盖追踪 + 提示管理 + 评估 + 会话回放。Arize Phoenix（Elastic 2.0）—— 深度 Agent 专项评估、RAG 相关性、OpenInference 自动埋点。Comet Opik（Apache 2.0）—— 自动提示优化、护栏、LLM 裁判幻觉检测。

**类型：** 学习
**语言：** Python（stdlib）
**前置条件：** 阶段 14 · 23（OTel GenAI）
**时间：** 约 45 分钟

## 学习目标

- 说出三大顶级开源 Agent 可观测性平台及其许可证。
- 区分各平台的最强项：Langfuse（提示管理 + 会话）、Phoenix（RAG + 自动埋点）、Opik（优化 + 护栏）。
- 解释为何到 2026 年，89% 的组织已部署 Agent 可观测性。
- 使用 stdlib 实现一个从追踪到看板、带 LLM 裁判评估的管线。

## 问题

OTel GenAI（第 23 课）提供了模式。你仍然需要一个平台来摄取跨度、运行评估、存储提示版本、并暴露回归问题。这三个竞争者各自强调生命周期的不同部分。

## 概念

### Langfuse（MIT）

- 月 SDK 安装量 600 万+，GitHub 星级 19000+。
- 特性：追踪、提示管理（含版本控制 + 试验场）、评估（LLM 作为裁判、用户反馈、自定义）、会话回放。
- 2025 年 6 月：原商业模块（LLM 作为裁判、标注队列、提示实验、试验场）在 MIT 许可下开源。
- 最强项：端到端可观测性与紧密的提示管理闭环。

### Arize Phoenix（Elastic License 2.0）

- 更深入的 Agent 专项评估：追踪聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动埋点。
- 可与托管版 Arize AX 配对用于生产。
- 不支持提示版本控制 —— 定位为漂移/行为回归工具，配合更广泛的平台使用。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验自动优化提示。
- 护栏（PII 脱敏、主题约束）。
- LLM 裁判幻觉检测。
- 来自 Comet 自身测量的基准：Opik 日志 + 评估用时 23.44 秒，而 Langfuse 为 327.15 秒（约 14 倍差距）—— 厂商基准请视为方向性参考。
- 最强项：优化闭环、自动实验、护栏执行。

### 行业数据

根据 Maxim（2026 年领域分析）：89% 的组织已部署 Agent 可观测性；质量问题是首要生产障碍（32% 的受访者提及）。

### 如何选择

| 场景 | 选择 |
|------|------|
| 一体化方案，含提示管理 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开源许可，无 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任意 —— 它们都导出 OTel |

### 该模式的常见错误

- **缺乏评估策略。** 没有评估的追踪只是昂贵的日志。
- **自行实现 LLM 裁判但缺乏依据。** CRITIC 模式（第 5 课）适用 —— 裁判需要外部工具进行事实核查。
- **提示版本未与追踪关联。** 当生产出现回归时，无法二分定位到是哪个提示导致的。

## 动手构建

`code/main.py` 实现了一个 stdlib 追踪收集器 + LLM 裁判评估器：

- 摄取 GenAI 形状的跨度。
- 按会话分组，标记失败运行（护栏触发、低置信度评估）。
- 一个脚本化的 LLM 裁判，根据评分标准对 Agent 响应打分。
- 类似看板的摘要：失败率、主要失败原因、评估分数分布。

运行它：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，与 Langfuse/Phoenix/Opik 显示的内容一致。

## 如何使用

- **Langfuse**：自托管或云；通过 OTel 或其 SDK 接入。
- **Arize Phoenix**：自托管；自动埋点 OpenInference。
- **Comet Opik**：自托管或云；自动优化闭环。
- **Datadog LLM 可观测性**：适用于已运行 Datadog 的运维+ML 混合团队。

## 交付

`outputs/skill-obs-platform-wiring.md` 选择一个平台，将追踪、评估和提示版本接入现有 Agent。

## 练习

1. 导出一周的 OTel 追踪到 Langfuse 云（免费层级）。哪些会话失败？原因是什么？
2. 为你的领域编写一个 LLM 裁判评分标准（事实准确性、语气、范围遵守）。在 50 个追踪上测试。
3. 比较 Langfuse 的提示版本控制与 Phoenix 的追踪聚类。哪个能更快告诉你哪里出了问题？
4. 阅读 Opik 的护栏文档。将 PII 脱敏护栏接入你的一个 Agent 运行。
5. 在你的语料库上对三个平台进行基准测试。忽略厂商发布的数字；测量你自己的。

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|------------------------|
| 追踪 | "跨度收集器" | 摄取 OTel / SDK 跨度；按会话索引 |
| 提示管理 | "提示内容管理系统" | 版本化的提示，与追踪关联 |
| LLM 作为裁判 | "自动评估" | 单独的 LLM 根据评分标准对 Agent 输出打分 |
| 会话回放 | "追踪回放" | 逐步回放过去运行以进行调试 |
| RAG 相关性 | "检索质量" | 检索到的上下文是否与查询匹配 |
| 追踪聚类 | "行为分组" | 将相似运行聚类以检测漂移 |
| 护栏执行 | "日志时的策略" | 对记录的内容执行 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) —— 追踪、评估、提示管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) —— 自动埋点、漂移
- [Comet Opik](https://www.comet.com/site/products/opik/) —— 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 三者共同使用的模式
