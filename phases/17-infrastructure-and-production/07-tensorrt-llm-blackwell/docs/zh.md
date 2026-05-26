# TensorRT-LLM 在 Blackwell 上使用 FP8 与 NVFP4

> TensorRT-LLM 是 NVIDIA 专有技术，但在 Blackwell 上表现优异。在 GB200 NVL72 上通过 Dynamo 编排，SemiAnalysis InferenceX 测得在 2026 年 Q1-Q2 期间，120B 模型的每百万 token 成本为 0.012 美元，而 H100 + vLLM 为 0.09 美元/百万 token——相差 7 倍的经济效益。这一技术栈由三种浮点模式组合而成：FP8 仍然是 KV cache 和注意力内核的关键，因为它具有所需的动态范围；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）与分离式预填充/解码又将性能提升 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需训练后转换。对于 2026 年的工程团队而言，难点在于：TRT-LLM 是一个闭源的 NVIDIA 技术栈，采用它意味着用可移植性换取吞吐量。在做出承诺之前，请根据你的模型和硬件组合来核算成本。

**类型：** 学习
**语言：** Python（标准库，简易 FP8/NVFP4 内存与成本计算器）
**先修知识：** Phase 17 · 04（vLLM Serving 内部原理），Phase 10 · 13（量化）
**预计用时：** 约 75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 仍然对 KV cache 和注意力至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省空间的原因。
- 列举 TRT-LLM 利用的 Blackwell 特定功能（Day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断何时 TRT-LLM 的 NVIDIA 锁定值得其相对于 Hopper 上 vLLM 的 7 倍成本差距。

## 问题

2026 年推理经济的前沿问题是“每美元能获得多少 token”。答案取决于四个层层叠加的选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）以及编排方式（标准 vs 分离式 vs Dynamo）。

在 Hopper 上使用 vLLM，一个 120B MoE 模型的运行成本约为每百万 token 0.09 美元。在 Blackwell 上使用 TRT-LLM + Dynamo，相同模型每百万 token 仅需 0.012 美元——便宜了 7 倍。部分差距来自硬件（Blackwell 每 GPU LLM 吞吐量比 Hopper 高 11-15 倍），部分来自技术栈：FP4 权重、MTP 草稿、分离式预填充/解码以及用于 MoE 专家通信的 NVLink 5 all-to-all。

这些效果在 NVIDIA 技术栈之外无法复现。这就是权衡——用可移植性换取经济效益。理解哪些技术栈选择贡献了多少份额的差距，正是本课的重点。

## 概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年常见错误：认为 NVFP4 适用于所有场景。事实并非如此。KV cache 需要 FP8（8 位浮点），因为它存储的注意力键和值具有广泛的动态范围。将 KV 量化为 FP4 会导致灾难性的精度损失——分布尾部丢失，注意力分数崩溃。FP8 的指数位为 KV cache 提供了所需的动态范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每个权重块拥有自己的缩放因子，因此小块可以在不损失每张量缩放的情况下覆盖不同的动态范围。对于激活来说，FP4 能够胜任，因为激活值在层内范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV cache：FP8。
- 注意力累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特定原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。不需要 AWQ/GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（Phase 17 · 05）相同的思想，但集成在 TRT-LLM 构建中。
- **分离式服务**：预填充和解码位于不同的 GPU 池，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）思想相同。
- **All-to-all 通信原语**：NVLink 5 相比 Hopper 将 MoE 专家通信延迟降低了 3 倍。TRT-LLM 的 MoE 内核为此进行了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上硬件加速的缩放因子处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到每百万 token 0.02 美元。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到每百万 token 0.012 美元。
- H100 + vLLM 在类似工作负载下约为每百万 token 0.09 美元。
- TRT-LLM 在三个月内更新带来 2.8 倍吞吐量提升（2026 年）。
- Blackwell 相比 Hopper 每 GPU LLM 吞吐量提升 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交任务中占主导地位。

### FP4 实际带来的质量代价

NVFP4 是激进的。在推理密集型工作负载（链式思维、数学、长上下文代码生成）上，FP4 权重会明显降低质量。逐块校准可以缓解但无法消除。发布推理模型的团队通常采用 FP8 权重 + FP4 激活作为折中方案，或者坚持在 H200 上使用全 FP8。

规则：在将 NVFP4 权重投入生产前，务必在你的评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要针对特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 对于使用 TRT-LLM 服务的层级来说是不可行的——你仍然可以在混合硬件上使用 vLLM 提供服务。如果你是纯 NVIDIA 用户，7 倍的差距足以抵消锁定成本。

### 2026 年实用配方

对于年推理账单超过 1 亿美元的用户，在 Hopper + vLLM 上运行会浪费 7-10 倍的成本。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以保持模型迭代速度。每个经过 NVFP4 转换的模型在投入生产前都要验证质量。

### 分离式服务的额外收益

TRT-LLM 的分离式服务（分离的预填充和解码池）在 Phase 17 · 20 中有深入介绍。在 Blackwell 上，乘数效应叠加：FP4 权重 × MTP 加速 × 分离式放置 × 缓存感知路由。7 倍数字假设了完整技术栈。

## 使用

`code/main.py` 计算一个模型在三种技术栈（H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM）下的 HBM 占用、解码吞吐量（内存受限模式）和每百万 token 成本。运行它来观察复合效应以及每个变化贡献的差距份额。

## 交付

本课程生成 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年 token 量，该文件会判断 Blackwell + TRT-LLM 技术栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于一个 120B MoE 模型，活跃参数占 30%，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 三种情况下内存带宽受限的解码吞吐量。哪个最大的提升来自何处？
2. 一个客户每年在 H100 + vLLM 上花费 200 万美元。鉴于 7 倍经济差距，他们需要购买多少块 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. 你在 NVFP4 权重转换后观察到 MATH 精度下降 3 个百分点。给出两种恢复路径：一个以质量优先（保留 FP8 权重），一个以成本优先（使用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务在 Blackwell 和 Hopper 之间的差距最小？为什么？
5. 计算一个 405B 模型在使用 NVFP4 权重 + FP8 KV cache（128k 上下文）时所需的 HBM。它能否放入单个 GB200 NVL72 节点？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| FP8 | "八位浮点" | 8 位浮点；用于 KV cache 和注意力，因其动态范围 |
| NVFP4 | "四位微" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上用于权重和激活 |
| MXFP8 | "MX 八" | 微缩放 FP8 变体；Blackwell Tensor Core 硬件加速 |
| Day-0 FP4 | "直接发布 FP4 权重" | 模型提供商以 FP4 形式发布权重；无需训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码草稿（Phase 17 · 05） |
| 分离式服务 | "分离预填充/解码" | 预填充和解码位于不同 GPU 池；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍延迟 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业公认的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 与 MoE 内核。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排框架。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数据的基准套件。
