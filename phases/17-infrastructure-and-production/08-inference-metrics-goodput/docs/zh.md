# 推理指标——TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否正常。TTFT 是预填充（prefill）加上排队（queue）和网络（network）。TPOT（等价于 ITL）是受内存限制的逐词元解码成本。端到端延迟等于 TTFT + TPOT × 输出长度。吞吐量是整个集群每秒处理的词元数。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求比例。高吞吐量但低 goodput 意味着你处理了大量无法及时到达用户的词元。以 2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数据为例：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均端到端延迟 1,093 ms。始终报告 P50、P90、P99——不要只报告平均值。注意测量陷阱：GenAI-Perf 在 ITL 计算中排除了 TTFT，而 LLMPerf 则包含 TTFT；同一个运行中，两个工具对 TPOT 的数值会不一致。

**类型：** 学习  
**语言：** Python（标准库，玩具级别的百分位计算器和 goodput 报告器）  
**前置知识：** 阶段 17 · 04（vLLM 服务内部机制）  
**时间：** ~60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、端到端延迟、吞吐量和 goodput，并说明每个指标衡量的组件。
- 解释为什么平均值是 LLM 服务中错误的统计量，并说明如何解读 P50/P90/P99。
- 构建一个多约束的 SLO（例如 TTFT<500 ms 且 TPOT<15 ms 且端到端延迟<2 s），并据此计算 goodput。
- 列举两个在同一个运行中对 TPOT 给出不同结果的基准测试工具，并解释原因。

## 问题所在

“我们的吞吐量是每秒 15,000 个词元。”所以呢？如果 40% 的请求端到端延迟超过了 2 秒，用户已经离开了会话。仅凭吞吐量无法判断产品是否正常。

推理具有多个延迟维度，每个维度的失效方式不同。预填充受计算限制，随提示长度扩展。解码受内存限制，随批处理大小扩展。排队延迟是运维问题。网络是物理距离问题。你需要针对每个维度的独立指标，还需要百分位数，还需要一个单一的复合指标来回答“用户是否得到了他们期望的结果”——那就是 goodput。

## 概念

### TTFT——首词元时间（Time To First Token）

`TTFT = 排队延迟 + 网络请求时间 + 预填充时间`

当提示很长时，预填充占主导。在 H100 上以 FP8 运行 Llama-3.3-70B 时，一个 32k 的提示需要约 800 ms 的纯预填充时间。排队时间是调度器在负载下的行为。网络请求时间包括 TLS 在内的线路时间。TTFT 是用户在收到任何流式响应之前感受到的延迟。

### TPOT / ITL——词元间延迟

同一个量有许多名称。TPOT（每输出词元时间）、ITL（词元间延迟）、解码延迟——都是同一个概念。它是第一个词元之后连续流式词元之间的时间。

`TPOT = (解码前向时间 + 调度器开销) / 产生的词元数`

在同一套 Llama-3.3-70B H100 堆栈上，使用分块预填充时，TPOT 平均值约 7 ms。如果未使用分块预填充，相邻序列长时间预填充期间，TPOT 可能飙升至 50 ms。关注 P99，而不是平均值。

### 端到端延迟（E2E Latency）

`端到端延迟 = TTFT + TPOT × 输出词元数 + 网络响应时间`

对于长输出（>500 词元），端到端延迟受 TPOT 主导。对于短输出长提示，端到端延迟受 TTFT 主导。报告按输出长度条件下的端到端延迟。

### 吞吐量（Throughput）

`吞吐量 = 总输出词元数 / 经过时间`

聚合指标。反映集群效率。不反映单个请求的健康状况。

### Goodput——你真正关心的指标

`goodput = 满足 (TTFT <= a) 且 (TPOT <= b) 且 (端到端延迟 <= c) 的请求比例`

SLO 是一个多约束条件。一个请求只有在所有约束条件都成立时才被认为是“好的”。goodput 就是这一比例。高吞吐量但 goodput 只有 60% 是失败；低吞吐量但 goodput 为 99% 才是目标。

到 2026 年，goodput 已经成为 MLPerf Inference v6.0 提交和人工智能平台提供商内部 SLA 追踪中使用的指标。

### 为什么平均值是错误的统计量

LLM 延迟分布是右偏的。一个解码批次如果有一个长预填充邻接序列，可能会发送 500 个 TPOT 约 7 ms 的词元和 20 个 TPOT 约 60 ms 的词元。平均 TPOT 是 9 ms，但 P99 TPOT 是 65 ms。用户会经常遇到 P99 的情况——这就是他们离开的原因。

始终报告三元组（P50, P90, P99）。对用户体验而言，P99 是你需要优化的重点。

### 参考数据——Llama-3.1-8B-Instruct on TRT-LLM, 2026

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均端到端延迟：1,093 ms
- P99 TPOT：根据分块预填充配置不同，在 10-25 ms 之间变化

这些是 NVIDIA 公布的参考点。它们会随着模型大小（70B 会显示 3-5 倍差异）、硬件（H100 与 B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年最常用的两个基准测试工具对同一个运行的 TPOT 结果不一致：

- **NVIDIA GenAI-Perf**：从 ITL 计算中排除 TTFT。ITL 从第 2 个词元开始计算。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个词元开始计算。

对于一个 TTFT 为 500 ms、输出 100 个词元且总解码时间为 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，而 LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具的选择会改变数值。

务必说明使用的工具。务必发布定义。

### 构建 SLO

一个面向消费者的 70B 聊天模型在 2026 年的合理 SLO 示例：

- TTFT P99 <= 800 ms
- TPOT P99 <= 25 ms
- 对于输出 <300 个词元，端到端延迟 P99 <= 3 s
- Goodput 目标 >= 99%

企业 SLO 会收紧 TTFT（200-400 ms）并放宽端到端延迟。关键是要将它们写下来，同时测量所有三个指标，并将 goodput 作为一个单一复合指标进行追踪。

### 如何测量

- 运行真实流量或逼真的合成流量（LLMPerf 使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准运行时目标并发度为峰值的 2 倍。
- 运行 30-50 次迭代，取合并样本的百分位数。
- 报告时附上工具名称、工具版本、模型、硬件、并发度、提示分布。

## 使用它

`code/main.py` 是一个玩具级的 goodput 计算器。它生成一个合成的延迟分布，应用 SLO，并计算 goodput。它还展示了在同一条追踪记录上 GenAI-Perf 与 LLMPerf 在 TPOT 上的差异。

## 交付它

本课时生成 `outputs/skill-slo-goodput-gate.md`。给定一个工作负载和 SLO，它会生成一个 CI/CD 就绪的基准测试配方，根据 goodput 而非吞吐量来设定部署门禁。

## 练习

1. 运行 `code/main.py`。生成一个带有 1% 尾部尖峰（tail spike）的分布。当将 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 供应商报价“在 Llama 3.3 70B H100 上每秒 15,000 个词元”。在相信这个数字之前，请列出三个需要提出的问题。
3. 为什么分块预填充能保护 P99 TPOT，却不能保护平均 TPOT？
4. 为语音助手构建一个消费级 SLO（第一个词元是听到的，而不是读到的）。哪个指标对用户最可见？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出另外三个两个工具不一致的指标。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|------------------------|
| TTFT | “首词元时间” | 排队 + 网络 + 预填充；长提示时预填充主导 |
| TPOT | “每输出词元时间” | 第一个词元之后受内存限制的解码成本 |
| ITL | “词元间延迟” | 大多数工具中与 TPOT 相同（并非全部——参见 GenAI-Perf） |
| 端到端 | “端到端” | TTFT + TPOT × 输出长度；再加上响应侧的网络 |
| 吞吐量 | “词元/秒” | 集群效率；没有延迟百分位数则毫无用处 |
| Goodput | “SLO 达标率” | 同时满足每个 SLO 约束条件的请求比例 |
| P99 | “尾部” | 1% 最差情况下的延迟；用户体验指标 |
| SLO 多约束 | “联合条件” | 三个延迟界限的 AND；任何一个违反则请求失败 |
| GenAI-Perf vs LLMPerf | “工具陷阱” | 工具在 ITL 是否包含 TTFT 上存在分歧 |

## 进一步阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 实际部署中的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业界公认的基于 goodput 的基准测试。
