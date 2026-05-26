# 顶点项目14 — 推测解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在实际流量下实现了 2.5–3 倍的吞吐量提升。P-EAGLE (AWS 2026) 将并行推测推向了更深层次。SGLang 的 SpecForge 大规模训练草稿头。Red Hat 的 Speculators 中心为常见开源模型发布了对齐的草稿。TensorRT-LLM 将推测解码作为 NVIDIA 上的一流特性。2026 年的生产服务栈是 vLLM 或 SGLang 搭配 EAGLE 系列草稿、FP8 或 INT4 量化以及基于队列等待的 HPA。本顶点项目的目标是服务两个开源模型，达到基线吞吐量的 2.5 倍以上，并提供完整的尾部延迟报告。

**类型：** 顶点项目
**语言：** Python (服务), C++ / CUDA (内核检查), YAML (配置)
**前置要求：** 阶段 3 (深度学习), 阶段 7 (Transformer), 阶段 10 (从头构建 LLM), 阶段 17 (基础设施)
**覆盖阶段：** P3 · P7 · P10 · P17
**预计时间：** 30 小时

## 问题

推测解码在 2026 年已成为商品化技术。EAGLE-3 草稿头在目标模型的隐藏状态上训练，并预测后续 N 个 token；目标模型在单次前向传递中完成验证。60–80% 的接受率可转化为 2–3 倍的端到端吞吐量。vLLM 0.7 原生集成了该技术。SGLang + SpecForge 提供了训练管线。Red Hat 的 Speculators 发布了针对 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 等模型的对齐草稿。

关键在于服务运维，而非模型本身。接受率随流量分布（ShareGPT 对比代码对比领域数据）而漂移。无推测时的尾部延迟反而更差 —— 你必须报告多个批次大小下的 p99 延迟，而不仅仅是稳态 token/秒。每百万 token 成本与 Anthropic / OpenAI API 的对比是可信度的杠杆。

## 概念

推测解码包含两层。一个**草稿**模型（EAGLE-3 头、n-gram 或较小的目标对齐模型）每步提出 k 个候选 token。**目标**模型在一次前向传递中验证所有 k 个 token；任何被接受的前缀替换掉贪心路径。接受率取决于草稿与目标的对齐程度以及输入分布。

EAGLE-3 在大多数流量上优于 n-gram 草稿。P-EAGLE 通过并行推测实现更深的草稿树。权衡点在于：拒绝时的 p99 延迟更高，因为验证前向传递更大。服务配置必须报告按批次大小分桶的延迟，以揭示这一点。

部署环境为 Kubernetes。vLLM 0.7 在每个 GPU 或张量并行分片上运行一个副本。HPA 基于队列等待时间而非 CPU 进行自动伸缩。FP8 (Marlin) 和 INT4 (AWQ) 量化将 GPU 内存限制在 H100 / H200 的容量内。端到端报告包括吞吐量、接受率、批次 1/8/32 下的 p50/p99 以及 $/1M token。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务: vLLM 0.7 或 SGLang 0.4
- 推测方法: EAGLE-3 草稿头, P-EAGLE 并行推测, n-gram 回退
- 草稿训练: SpecForge (SGLang) 或 Red Hat Speculators
- 目标模型: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- 量化: FP8 (Marlin), INT4 AWQ
- 部署: Kubernetes + NVIDIA 设备插件; 基于队列等待时间的 HPA
- 评估: ShareGPT, MT-Bench-v2, GSM8K, HumanEval — 用于衡量领域跨度下的接受率
- 参考: TensorRT-LLM 推测解码 — 作为供应商基线

## 构建步骤

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 vLLM 0.7 上部署在 1×H100（或 2×张量并行）上。

2. **草稿来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的推测解码配置中。

3. **基线数据。** 推测前：批次 1/8/32 下的 token/秒、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 切换配置；重新运行相同基准测试。报告加速比、接受率、p99 尾部延迟变化。

5. **P-EAGLE。** 启用并行推测；测量更深草稿树与串行 EAGLE-3 的对比。报告 P-EAGLE 在哪些情况下有帮助、在哪些情况下有害。

6. **领域流量。** 通过同一服务器运行 ShareGPT、HumanEval 以及特定领域流量。测量每种分布下的接受率。识别草稿何时漂移。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同管线。草稿更棘手（MoE 路由噪声）。报告结果。

8. **K8s HPA。** 在 Kubernetes 下部署，HPA 监控 `queue_wait_ms`。演示当负载翻三倍时如何自动扩展。

9. **成本对比。** 计算每百万 token 成本，与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在相同评估集上对比。发布结果。

## 使用方法

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付物

`outputs/skill-inference-server.md` 描述交付内容。包含一个经过实测的推测解码服务栈、一份完整的基准测试报告以及一个 Kubernetes 部署。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对于基线的实测加速比 | 两个模型上以匹配质量实现 2.5 倍以上的吞吐量 |
| 20 | 真实流量下的接受率 | 每种分布下的接受率报告 |
| 20 | p99 尾部延迟纪律 | 有无推测时批次 1/8/32 下的 p99 延迟 |
| 20 | 运维 | Kubernetes 部署、基于队列等待的 HPA、平滑上线 |
| 15 | 文档与方法论 | 清晰解释做了什么以及为什么 |
| **100** | | |

## 练习

1. 测量当草稿比目标落后一个版本时（例如 Llama 3.3 → 3.4 漂移）接受率的下降。构建一个监控告警。

2. 实现 n-gram 回退：如果 EAGLE-3 接受率低于某个阈值，切换到 n-gram 草稿。报告可靠性提升。

3. 运行受控的 MoE 实验：对同一个 Qwen3-Coder-30B，分别在有路由噪声和无路由噪声的情况下运行。测量草稿接受率的敏感性。

4. 扩展到 H200（141 GB）。报告每个副本可容纳的模型大小裕度，以及能否服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上对 TensorRT-LLM 推测解码进行基准测试。报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | "Speculator" | 小型模型，提出 N 个 token 供目标模型验证 |
| EAGLE-3 | "2026 年草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行推测" | 在单次目标前向传递中验证的草稿分支树 |
| Acceptance rate | "命中率" | 被接受的草稿 token 比例（无需重新采样） |
| Quantization | "FP8 / INT4" | 低精度权重，将更多模型放入 GPU 内存 |
| Queue wait | "HPA 指标" | 请求在待处理队列中等待推理开始的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 发布的常见开源模型 EAGLE 草稿中心 |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考服务栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行推测解码论文及集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练管线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐的草稿中心
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) — 代码与基准
