# EAGLE-3 推测解码：生产环境实践指南

> 推测解码将一个快速的草稿模型与目标模型配对使用。草稿模型提出 K 个 token；目标模型通过一次前向传播完成验证；被接受的 token 相当于“免费”获得。到 2026 年，EAGLE-3 已成为生产级版本——它在目标模型的隐藏状态而非原始 token 上训练一个草稿头，在通用对话场景下将接受率 alpha 推高至 0.6–0.8 区间。关键问题不是“草稿模型有多快”，而是“我的流量上的 alpha 是多少？”如果 alpha 低于约 0.55，在高并发下推测解码将是净负收益，因为每次拒绝的草稿都会导致第二次目标前向传播。本课程教您先测量 alpha，再决定是否开启该功能。

**类型：** 学习
**语言：** Python（标准库，简易接受率模拟器）
**前置条件：** 阶段17 · 04（vLLM 服务内部原理），阶段10 · 18（多 token 预测）
**时间：** 约 60 分钟

## 学习目标

- 列出推测解码的三代演进，解释 EAGLE-3 相对于 EAGLE-2 和经典草稿模型的变化。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算预期加速比，并确定目标并发下的盈亏平衡 alpha。
- 解释为什么在 vLLM 2026 中推测解码是 opt-in（非默认），以及为什么不测量 alpha 就开启它属于生产反模式。
- 编写测量计划：使用哪个基准测试、哪种提示分布、哪个并发点、以哪个指标作为门控。

## 问题所在

解码受限于内存带宽。在运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并产生一个 token。解码期间 GPU 计算几乎空闲——瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

推测解码利用了这一差距。用一个廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证所有 K 个 token。每个被验证的 token 实际上是“免费”的（平摊到目标模型本就需要做的 K 批次前向传播中）。

经典的草稿模型方法使用同一系列的小型模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。该方法有效，但接受率一般——小模型的分布与目标模型存在偏差。EAGLE，然后是 EAGLE-2，再到 EAGLE-3，直接在目标模型的内部状态上训练一个轻量级的草稿头，因此草稿的分布与目标模型更接近。这就是为什么 alpha 从草稿模型方法的 0.4 提升到 EAGLE-3 的 0.6–0.8。

但注意：EAGLE-3 在 vLLM 2026 中是 opt-in 的，必须显式设置 `speculative_config`。没有标记，就没有加速。团队在不测量其实际流量的 alpha 时就开启它，常常会看到尾部延迟变得更差，而不是更好。

## 概念

### 推测解码真正带来的好处

没有推测解码时，每个 token 的成本是一次目标前向传播。使用推测解码，在草稿长度 K 和接受率 alpha 下，每个目标前向传播预期的 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。当 K=5，alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数值集中在 2–3 倍，因为 alpha 在生产流量中很少那么高，而且 epsilon 在大批次下会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 并不会消失——它们会迫使目标模型对被拒绝的第一个 token 进行第二次前向传播。在工作负载中，如果 alpha 降至 0.4，你将支付草稿开销、验证开销以及重新生成的开销。在高并发（例如 256 个并发）下，解码批次已经足够大，以至于“仅有目标模型”和“目标模型加验证”之间的内存带宽差距缩小。在大多数 2026 年硬件上，当 alpha 低于 0.55 时，推测解码为净负收益。

Alpha 随工作负载变化。在 ShareGPT 风格的通用对话上，基于 ShareGPT 训练的 EAGLE-3 达到 0.6–0.8。对于领域特定流量（代码、医疗、法律），在通用数据上训练的草稿头会降至 0.4–0.6。训练领域特定的草稿头可以恢复 alpha——与目标模型微调相比，这是一个轻量且快速的训练任务。

### EAGLE 代际概览

- **经典草稿模型**：同一系列的小模型。Alpha 0.3–0.5。基础设施简单——加载两个模型，每个目标前向传播中草稿运行 K 次前向。
- **EAGLE-1 (2024)**：在目标模型隐藏状态（最后一层）上训练的单个草稿头。Alpha 约 0.5–0.6。在目标模型之上增加少量参数开销。
- **EAGLE-2 (2025)**：自适应草稿长度和基于树的草稿（在一次目标前向传播中验证多条分支）。Alpha 约 0.6–0.7。草稿调度器更复杂。
- **EAGLE-3 (2025–2026)**：在目标模型多个层（不仅是最后一层）上训练的草稿头，对齐更好。通用对话上 Alpha 0.6–0.8。

### 2026 年生产方案

1. 单独部署目标模型。测量目标并发下的基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 通过 `spec_decode_metrics.accepted_tokens_per_request` 报告该值。除以请求的草稿长度得到 alpha。
4. 如果生产流量分布下的 alpha < 0.55，禁用推测解码或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 未变差。

### 生产陷阱：P99 尾部

平均 ITL 因推测解码而下降。但如果不进行调优，P99 可能会变差。被拒绝的草稿会触发两阶段序列（草稿 + 验证失败 + 重新生成）。在满批次下，这两个阶段会串行化。请关注 P99 ITL，而非 P50。

### EAGLE-3 已部署的场景

谷歌在 2025 年将推测解码部署到 AI Overviews（相同质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档化的接口提供；V1 中的 N-gram GPU 推测解码是与分块预填充兼容的变体。SGLang 支持 EAGLE-3，并将其作为前缀密集型工作负载的推荐草稿路径。

### 盈亏平衡公式一行总结

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 可解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead ≈ 0.15，K=5：`alpha_breakeven = 0.03`。但这是原始解码的计算。在高并发下，验证开销增加，解码批次已经将内存读取平摊到多个序列上，因此有效盈亏平衡 alpha 在实践中会上升到约 0.45–0.55。

### 何时不应使用推测解码

- 单批次离线生成，延迟不重要时。使用普通目标模型。
- 输出非常短（少于 50 个 token）。草稿开销和验证成本占主导。
- 专业领域没有经过领域训练的草稿头。Alpha 太低。
- vLLM v0.18.0 加上草稿模型推测解码再加上 `--enable-chunked-prefill`。此组合无法编译。文档中记录的例外是 V1 中的 N-gram GPU 推测解码。

## 使用它

`code/main.py` 模拟了在不同 alpha 值和草稿长度 K 下，有无推测解码的解码循环。它打印出盈亏平衡 alpha、测量到的加速比以及尾部行为。在多个 (alpha, K) 组合上运行它，以精确了解推测解码何时停止带来收益。

## 交付它

本课程生成 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它将输出一个分阶段的 EAGLE-3 部署计划——基准测试基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。当 K=5 时，要实现 2 倍加速比需要多大的 alpha？3 倍加速比呢？该结果对 verify_overhead 的敏感度如何？
2. 假设生产流量为 70% 通用对话，30% 代码。通用对话在使用 ShareGPT 训练的 EAGLE-3 下 alpha 为 0.7；代码 alpha 为 0.4。混合 alpha 是多少？推测解码是否为净正收益？
3. 阅读 vLLM `speculative_config` 文档。列出三种模式（草稿模型、EAGLE、N-gram），并指出哪一种与分块预填充兼容。
4. 你发现启用 EAGLE-3 后平均 ITL 下降了 25%，但 P99 ITL 上升了 15%。请诊断并提出缓解措施。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。与使用 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 日常说法 | 实际含义 |
|------|---------|--------|
| 推测解码 (Speculative decoding) | “draft plus verify” | 用一个廉价模型提出 K 个 token，在一次目标前向传播中验证所有 K 个 |
| 接受率 alpha (Acceptance rate alpha) | “spec accept rate” | 目标模型接受的草稿 token 比例；唯一重要的指标 |
| 草稿长度 K (Draft length K) | “spec k” | 每个目标前向传播中草稿提出的 token 数；典型值 4-8 |
| 验证开销 epsilon (Verify overhead epsilon) | “spec overhead” | 相对于普通目标前向传播，验证并重新生成的额外成本；随批次增大而增加 |
| EAGLE-3 | “latest EAGLE” | 2025-2026 变体；在目标模型的多个层上训练草稿头；通用对话 alpha 0.6-0.8 |
| `speculative_config` | “vLLM spec config” | vLLM V1 中的显式 opt-in；无默认值意味着无加速 |
| N-gram 推测解码 (N-gram spec decode) | “N-gram draft” | 利用提示中的 N-gram 查找进行 GPU 端草稿；兼容分块预填充 |
| 盈亏平衡 alpha (Break-even alpha) | “no-op alpha” | 推测解码带来零加速时的 alpha；请在生产并发下关注此值 |
| 拒绝草稿两阶段 (Rejected-draft two-pass) | “reroll cost” | 当草稿被拒绝时发生两次目标前向传播；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 `speculative_config` 和 V1 中分块预填充兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 确切的字段集合。
- [EAGLE 论文 (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE 草稿头公式。
- [EAGLE-2 论文 (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应草稿与树结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 使用推测解码的高效 LLM 系统。
- [BentoML — 推测解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产部署检查清单。
