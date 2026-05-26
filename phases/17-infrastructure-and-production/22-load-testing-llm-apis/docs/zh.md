# 对 LLM API 进行负载测试——为什么 k6 和 Locust 会撒谎

> 传统的负载测试工具并非为流式响应、可变输出长度、token 级指标或 GPU 饱和而设计。有两个陷阱常让团队栽跟头。**GIL 陷阱**：Locust 在客户端侧进行 token 级测量，其 tokenization 操作位于 Python GIL 之下；在高并发场景下，tokenization 会与请求生成争抢资源，导致 tokenization 积压，从而虚报 token 间延迟——瓶颈变成了你的客户端，而非服务端。**提示一致性陷阱**：在循环中使用完全相同的提示只测试了 token 分布上的一个点；真实流量包含可变长度和多样化的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这一问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式、Kubernetes 原生分布式测试（通过 TestRun/PrivateLoadZone CRD），最适合 CI/CD 门控；Vegeta 用于 Go 恒定速率饱和；Locust 2.43.3 仅在与 LLM-Locust 扩展配合时才支持流式。负载模式：稳态、斜坡、尖峰（自动缩放测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库，简易真实提示生成器 + 延迟收集器）
**前置条件：** 阶段 17 · 08（推理指标），阶段 17 · 03（GPU 自动缩放）
**时间：** ~75 分钟

## 学习目标

- 解释两种反模式（GIL 陷阱、提示一致性陷阱），它们导致通用负载测试工具对 LLM API 给出错误结果。
- 根据特定用途选择合适的工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、斜坡、尖峰、浸泡），并说出每种模式能捕获的故障类型。
- 使用输入 token 的均值 + 标准差（而非固定长度）构建真实的提示分布。

## 问题

你用 k6 以 500 并发用户测试了你的 LLM 端点。它挺住了。你发布了。生产环境只有 200 个实际用户时服务就崩溃了——P99 TTFT 飙升，GPU 满负荷。

发生了两件事。首先，k6 发送了 500 个完全相同的提示——你的请求合并和前缀缓存使得它看起来像在处理 500 个并发解码，实际上只处理了一个。其次，k6 无法像人眼感知那样跟踪流式响应的 token 间延迟；它看到的是一个 HTTP 连接，而不是以不同间隔到达的 500 个 token。

LLM 的负载测试是一门独立的学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下执行客户端侧的 tokenization。高并发时 tokenizer 会在请求生成后面排队。报告的 token 间延迟包含了客户端侧的 tokenization 积压。你以为服务器慢，其实是测试工具本身的问题。

解决方法：LLM-Locust 扩展将 tokenization 移到独立进程中，或者使用编译语言编写的测试工具（k6、LLMPerf 使用 tokenizers.rs）。

### 提示一致性陷阱

所有已知的负载测试工具都允许配置一个提示。在 10,000 次迭代的循环测试中，每次都会发送完全相同的提示。服务端每次都看到相同的前缀——前缀缓存命中率趋近 100%，吞吐量看起来很好。

解决方法：从提示分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样，内容多样。

### 四种负载模式

1. **稳态**——恒定 RPS 持续 30-60 分钟。捕获：基准性能回归。
2. **斜坡**——在 15 分钟内将 RPS 从 0 线性增加到目标值。捕获：容量拐点、预热异常。
3. **尖峰**——RPS 突然增加 3-10 倍持续 2 分钟，然后恢复。捕获：自动缩放延迟、队列饱和、冷启动影响。
4. **浸泡**——稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）——Python，但 tokenization 基于 Rust。支持均值和标准差提示。支持流式。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf**——NVIDIA 的参考工具。使用 Triton 客户端；指标覆盖全面。注意它的 ITL 不包含 TTFT；LLMPerf 的 ITL 包含。两台工具对同一服务器会给出不同的 TPOT。

**LLM-Locust**（TrueFoundry）——Locust 扩展，修复了 GIL 陷阱。熟悉的 Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译语言，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 进行 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门控和 SLA 测试。

**Vegeta**——Go 语言，比 k6 简单。用于恒定速率 HTTP 饱和测试。不具备 LLM 感知能力，但适合网关/限速测试。

**Locust 2.43.3 原生版本**——对于 LLM 存在 GIL 陷阱。只有在 LLM-Locust 扩展下才能使用。

### CI 中的 SLA 门控

在 PR 上运行 k6：

- 以基线 RPS 运行 30-50 次迭代。
- 门控条件：P50/P95 TTFT、5xx 错误率 < 5%、TPOT 低于阈值。
- 违反即阻断构建。

### 真实提示分布

从真实流量样本中构建（如果有的话），或从公开分布中获取（例如 ShareGPT 提示用于聊天，HumanEval 用于代码）。将均值和标准差输入给 LLMPerf。无论如何都要避免使用单提示循环。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支持流式感知指标。
- 典型 LLMPerf 运行：100-1000 个请求，并发数为 X。
- 典型 CI 门控：每个 PR 30-50 次迭代。
- 四种模式：稳态、斜坡、尖峰、浸泡。

## 使用它

`code/main.py` 模拟了一个具有真实提示分布的负载测试，测量有效 TPOT，并演示了统一提示陷阱。

## 交付它

本节课产生 `outputs/skill-load-test-plan.md`。根据工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较统一分布与真实分布——差距在哪里？
2. 为 CI 门控编写 k6 脚本：TTFT P95 < 800 ms，并发数 100，运行时间 5 分钟。
3. 你的浸泡测试显示内存以 50 MB/小时增长。说出三种可能的原因以及用于区分它们的监控手段。
4. 尖峰测试从 10 RPS 突增到 100 RPS。如果 Karpenter + vLLM 生产栈已就位（阶段 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务器上报告 TPOT=11ms。解释。

## 关键术语

| 术语 | 大家说的 | 实际含义 |
|------|----------|----------|
| LLMPerf | "那个 LLM 测试工具" | Anyscale 基准测试工具，支持流式 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "面向 LLM 的 Locust" | Locust 扩展，修复 GIL 陷阱 |
| guidellm | "合成基准测试" | 大规模合成工具 |
| k6 Operator | "K8s 上的 k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | tokenization 积压导致报告延迟虚高 |
| 提示一致性陷阱 | "单提示谎言" | 循环使用相同提示，命中缓存，虚高吞吐量 |
| 稳态 | "恒定负载" | 固定 RPS 持续 N 分钟 |
| 斜坡 | "线性上升" | 从 0 到目标值随时间上升 |
| 尖峰 | "突发测试" | 突然倍增后恢复 |
| 浸泡 | "长时间测试" | 持续数小时用于检测泄漏 |

## 扩展阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
