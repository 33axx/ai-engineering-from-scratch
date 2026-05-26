# 分离式 Prefill/Decode — NVIDIA Dynamo 与 llm-d

> Prefill 是计算密集型的；decode 是内存密集型的。在同一 GPU 上同时运行两者会浪费一种资源。分离式架构将它们拆分到不同的池中，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）传输 KV 缓存。NVIDIA Dynamo（GTC 2025 宣布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler 和 SLA Planner 自动匹配 prefill:decode 比率以满足 SLO。NVIDIA 公布了大约的吞吐量提升——developer.nvidia.com（2025-06）显示在中等延迟场景下，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上获得约 6 倍的提升；Dynamo 产品页面（developer.nvidia.com，日期不详）宣称在 GB300 NVL72 + Dynamo 上相比 Hopper 最高可达 50 倍的 MoE 吞吐量。社区汇总的“30倍”数字涵盖全栈 Blackwell + Dynamo + DeepSeek-R1 的报告；我们尚未找到明确指出恰好 30 倍的单一原始来源，因此应将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生方案：prefill / decode / router 作为独立服务，每个角色配有 HPA。llm-d 0.5 增加了层次化 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、伸缩至零。经济性：内部汇总多个客户披露信息表明，在恒定 SLA 下，从同地部署切换到使用 Dynamo 的分离式部署，在 200 万美元规模的推理支出中可节省 30–40%（即每年 60-80 万美元）；这个 200 万 → 60-80 万的具体数字是内部综合数据，而非单一已发布案例研究——请将其作为数量级锚点使用，而非引用来源。短提示（<512 token，短输出）无法证明传输成本合理。

**类型：** 学习  
**语言：** Python（标准库，玩具版分离式与同地部署模拟器）  
**先修知识：** Phase 17 · 04（vLLM Serving Internals）、Phase 17 · 08（推理指标）  
**时间：** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 具有不同的最优 GPU 分配，并量化同地部署下的浪费。
- 绘制分离式架构图：prefill 池、decode 池、通过 NIXL 进行的 KV 传输、路由器。
- 指出分离式架构无法带来回报的条件（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上层）与 llm-d（Kubernetes 原生），并将每种方案匹配到相应的运营环境。

## 问题

你在 8 块 H100 上运行 Llama 3.3 70B。在混合工作负载下（长提示 + 短输出），GPU 在 decode 期间闲置，因为大部分算力花在了 prefill 上。在不同工作负载下（短提示 + 长输出），情况相反。同地部署 prefill + decode 意味着你同时过度配置了两者。

预算影响：20-40% 的 GPU 时间浪费在了错误的资源上。你购买 H100 算力来运行内存密集型的 decode，或者购买 H100 HBM 带宽来运行计算密集型的 prefill。两者都是昂贵的浪费。

分离式架构将 prefill 和 decode 拆分到各自针对瓶颈规模定制的独立池中。KV 缓存通过高带宽互连从 prefill 池传输到 decode 池。

## 概念

### 为什么瓶颈不同

**Prefill** — 在单次前向传播中运行整个输入提示的 transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批处理效率高 — 一次前向传播可处理许多 token。

**Decode** — 每次生成一个 token，每次迭代读取全部权重。内存带宽受限。HBM3 提供约 3 TB/s。仅在高并发时批处理效率高 — 权重读取在整个批次中摊销。

将它们同地部署：你购买的 GPU 同时针对两者进行了优化。H100 在两者上都表现出色，但成本相同。在规模上，你希望 prefill 池使用 H100 / 计算密集型；decode 池使用 H200 / 内存密集型，或采用激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输。在有条件时使用 RDMA/InfiniBand，否则回退到 TCP。传输延迟确实存在 — 对于 70B FP8 上 4K token 提示的 KV 缓存，通常为 20-80 ms。这就是为什么短提示不适用于分离式架构：传输开销超过了节省。

### Dynamo 与 llm-d 对比

**NVIDIA Dynamo**（GTC 2025 宣布，1.0 GA）：
- 作为编排器位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比率。
- Rust 核心，Python 可扩展性。
- 吞吐量提升：NVIDIA 报告在中等延迟场景下，DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上获得 6 倍提升（developer.nvidia.com，2025-06）；社区报告的全栈 Blackwell + Dynamo + DeepSeek-R1 上的“最高 30 倍”缺乏单一原始来源，应视为方向性信息。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE 吞吐量最高达 50 倍，源自 Dynamo 产品页面（developer.nvidia.com，日期不详）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立的 Kubernetes 服务。
- 使用队列深度（prefill）/ KV 利用率（decode）信号的每个角色 HPA。
- `topologyConstraint packDomain: rack` 将 prefill 和 decode 组打包到同一机架，以实现高带宽 KV 传输。
- llm-d 0.5（2026）：层次化 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、伸缩至零。

如果你想要托管式的栈上层编排器，请使用 Dynamo。如果你想要 Kubernetes 原生原语并且致力于 CNCF 生态系统，请使用 llm-d。

### 经济性

内部综合数据（非单一已发布案例研究 — 数量级锚点）：

- 每年 200 万美元的同地部署推理支出。
- 切换到使用 Dynamo 的分离式架构。
- 相同的请求量，相同的 P99 延迟 SLA。
- 报告节省：每年 60-80 万美元（减少 30-40%）。
- 无新硬件。

我们将这个数字综合自多个客户披露的信息，而非单个可引用的案例研究；最接近的已发布数据点是 Baseten 在使用 Dynamo KV 路由后 TTFT 提升 2 倍 / 吞吐量提升 61%（baseten.co，2025-10），以及 VAST + CoreWeave 在 40-60% KV 命中率下预测的 token/$ 提升 60-130%（vastdata.com，2025-12）。节省来自对每个池进行恰当规模调整；prefill 繁重的工作负载（RAG，8K+ 前缀）比均衡负载受益更多。

### 何时不应该使用分离式架构

- 提示 < 512 token 且输出 < 200 token：传输开销超过节省。
- 小集群（< 4 个 GPU）：没有足够的池多样性。
- 团队无法操作两个 GPU 池并支持每个角色的扩缩容：Dynamo 有帮助，但并不简单。
- 没有 RDMA 网络：TCP 传输开销更大。

### 路由器与 Phase 17 · 11 集成

分离式路由器是 KV 缓存感知的（Phase 17 · 11）。请求落在持有其前缀的 decode 池上 — 如果不匹配，则流向 prefill → decode。命中率与分离式架构叠加 — 缓存感知路由器决定是否甚至需要新的 prefill。

### MoE on Blackwell 才是真正的数字所在

GB300 NVL72 + Dynamo 相比 Hopper 基准显示 MoE 吞吐量提升 50 倍。MoE 专家路由在 prefill 上计算密集，但在 decode 上内存密集（专家缓存），因此分离式架构是双重胜利。2026 年前沿模型服务将以 MoE 为主导（DeepSeek-V3，未来的 GPT-5 变体）。

### 你应该记住的数字

基准数字会变动 — NVIDIA 和推理栈每个季度都会发布更新结果。在引用前请重新检查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中等延迟场景下基准吞吐量提升约 6 倍（developer.nvidia.com，2025-06）；社区声称的全栈 Blackwell + Dynamo 上的“最高 30 倍”是方向性汇总，没有单一原始来源。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE 吞吐量最高达 50 倍（developer.nvidia.com，日期不详）。
- 节省锚点（内部综合数据，非单一案例研究）：在恒定 SLA 下，每年 200 万美元支出中节省 60-80 万美元。
- 分离式阈值：提示 > 512 token + 输出 > 200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示的 KV 传输时间为 20-80 ms。

## 使用它

`code/main.py` 模拟同地部署与分离式服务。报告吞吐量、每请求成本以及提示长度交叉点。

## 交付

本课程产生 `outputs/skill-disaggregation-decider.md`。根据工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在什么提示长度下，分离式架构胜过同地部署？
2. 为 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：为一个纯 Kubernetes 环境且没有 Python 运行时偏好的团队选择一种方案。
4. 计算 KV 传输成本：70B FP8 上的 4K prefill ≈ 500 MB KV。在 RDMA 100 GB/s 下，传输 = 5 ms。在 TCP 10 GB/s 下 = 50 ms。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。对于每个 token 激活不同专家的 MoE，分离式架构表现如何？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-------------|----------|
| 分离式服务 (Disaggregated serving) | “拆分 prefill/decode” | 为每个阶段配备独立的 GPU 池 |
| NIXL | “NVIDIA 传输” | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | “编排器” | vLLM/SGLang/TRT-LLM 的栈上层协调器 |
| llm-d | “Kubernetes 原生” | Red Hat + AWS 的 K8s 分离式栈 |
| Planner Profiler | “Dynamo 自动配置” | 测量工作负载，配置池比率 |
| SLA Planner | “Dynamo 策略” | 自动匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | “llm-d 拓扑” | 将 prefill+decode 打包到同一机架以实现快速 KV |
| UCCL | “统一集合通信” | llm-d 0.5 的网络层，用于伸缩至零 |
| MoE 专家路由 | “每个 token 的专家” | DeepSeek-V3 模式；分离式架构有助于优化 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
