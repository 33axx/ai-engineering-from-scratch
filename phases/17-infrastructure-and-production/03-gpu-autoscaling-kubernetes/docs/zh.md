# Kubernetes上的GPU自动扩缩容 —— Karpenter、KAI Scheduler、组调度

> 三层，而非一层。Karpenter动态配置节点（一分钟以内，比Cluster Autoscaler快40%）。KAI Scheduler负责组调度、拓扑感知和层级队列——它防止了7/8部分分配陷阱：七个节点等待一个缺失的GPU，白白耗钱。应用层自动扩缩容工具（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理专用信号扩缩——队列深度、KV缓存利用率——而非CPU/DCGM占空比。经典的HPA陷阱在于：`DCGM_FI_DEV_GPU_UTIL`是一个占空比测量值：100%可能对应10个请求或100个。vLLM预分配KV缓存内存，因此内存永远不会触发缩容。本课程教你组合这三个层次，并避开默认的Karpenter `WhenEmptyOrUnderutilized`策略——该策略会在推理中途终止正在运行的GPU任务。

**类型：** 学习
**语言：** Python（stdlib，简易队列深度自动扩缩容模拟器）
**前置知识：** 阶段17·02（推理平台经济学），阶段17·04（vLLM服务内部原理）
**时长：** 约75分钟

## 学习目标

- 画出三层自动扩缩容架构图（节点配置、组调度、应用层），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 不适合作为vLLM的HPA信号，并说出两个替代信号（队列深度、KV缓存利用率）。
- 描述组调度以及KAI Scheduler所防止的部分分配失败模式（8个GPU闲置7个）。
- 说出会导致正在运行的GPU任务被终止的Karpenter整合策略（`WhenEmptyOrUnderutilized`），并指出2026年的安全替代方案。

## 问题

你的团队在Kubernetes上部署了一个LLM服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了HPA。服务在工作时间一直固定在100%利用率。HPA从未扩容——它已经认为你满载了。你手动增加一个副本；TTFT下降。HPA仍然不扩容。这个信号欺骗了你。

另外，你使用Cluster Autoscaler来管理节点。凌晨2点一个100万token的提示到达；集群花了3分钟配置节点，请求超时。

另外，你部署了一个需要8个GPU、分布在2个节点上的70B模型。集群有7个空闲GPU，但第8个分散在3个节点上。Cluster Autoscaler为缺失的那个GPU配置了一个节点。七个节点等待4分钟烧钱，而Kubernetes才把最后一个GPU准备好。

三层，三种不同的失败模式。2026年的GPU感知自动扩缩容不是“开启HPA”，而是组合节点配置、组调度和应用信号自动扩缩容。

## 概念

### 第一层——节点配置（Karpenter）

Karpenter监控待处理Pod，并在约45-60秒内配置节点（Cluster Autoscaler通常需要90-120秒来配置GPU节点）。它根据 `NodePool` 约束动态选择实例类型——如果你的Pod需要8个H100，而集群没有匹配的节点，Karpenter直接配置一个，而不是扩展现有的实例组。

**整合陷阱**：Karpenter的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对GPU池是危险的。它会终止正在运行的GPU节点，以便将Pod迁移到更便宜的合适实例上。对于推理工作负载，这意味着驱逐正在运行的请求，并在新节点上重新加载70B模型。损失是数分钟的容量加上请求失败。

GPU池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许Karpenter在一小时后整合真正空的节点，但从不驱逐正在运行的任务。

### 第二层——组调度（KAI Scheduler）

KAI Scheduler（项目曾用名“Karp”，后更名）处理默认kube-scheduler做不到的事情：

**组调度**——全有或全无调度。一个需要8个GPU的分布式推理Pod要么全部8个同时启动，要么一个都不启动。没有这个机制，你就会遇到部分分配陷阱：7个Pod启动，无限期等待，烧钱。

**拓扑感知**——了解哪些GPU共享NVLink，哪些位于同一机架，哪些之间有InfiniBand。相应地放置Pod。一个DeepSeek-V3 67B的张量并行工作负载必须停留在一个NVLink域内；KAI Scheduler尊重这一点。

**层级队列**——多个团队竞争同一个GPU池，带有优先级和配额。A团队的生产环境插队只有在优先级规则允许时才会被B团队的训练任务抢占。

KAI作为辅助调度器与kube-scheduler一起部署；你可以通过注解让工作负载使用它。Ray和vLLM生产栈都集成了它。

### 第三层——应用级信号

**HPA陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一个占空比指标——它测量GPU在每个采样间隔是否在做工作。100%利用率可能意味着10个并发请求或100个；无论如何GPU都是忙的。根据占空比扩缩容相当于盲目扩缩。

更糟的是，vLLM及类似引擎会预分配KV缓存内存（最多到 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也保持在90%左右。基于内存的HPA永远不会缩容。

**2026年替代信号**：

- 队列深度（等待预填充的请求数）。
- KV缓存利用率（分配给活跃序列的块的比例）。
- 每副本P99 TTFT（你的SLA信号）。
- 优质吞吐量（每秒满足所有SLO的请求数）。

NVIDIA Dynamo Planner和llm-d Workload Variant Autoscaler消费这些信号并调整副本数量。它们完全取代了LLM服务中的HPA。

### 什么时候用什么

| 扩缩容决策 | 工具 |
|------------|------|
| 添加/移除节点 | Karpenter |
| 调度多GPU任务 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义HPA） |
| 选择GPU类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler队列 |

### 分离式预填充/解码使一切复杂化

如果运行分离式预填充/解码（阶段17·17），你会得到两类Pod，它们具有不同的扩缩容触发器：预填充Pod根据队列深度扩缩，解码Pod根据KV缓存压力扩缩。llm-d将这些暴露为独立的 `Service`，每个角色有自己的HPA。不要试图在两者前面放一个单一的HPA。

### 冷启动在这里同样重要

冷启动缓解（阶段17·10）正是节点配置时间变得用户可见的地方。Karpenter 45-60秒的预热加上20GB模型加载加上引擎初始化意味着从零开始的请求需要2-5分钟。为SLO关键路径保留一个热池（`min_workers=1`），或者在应用层使用Modal风格的检查点。

### 你应该记住的数字

- Karpenter节点配置：约45-60秒 vs Cluster Autoscaler约90-120秒（GPU节点）。
- KAI Scheduler防止部分分配浪费——7/8陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为HPA信号：不可靠；使用队列深度或KV利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的GPU任务。对于推理，使用 `WhenEmpty + consolidateAfter: 1h`。

## 使用它

`code/main.py` 模拟了一个三层自动扩缩容器在突发GPU工作负载下的行为。比较了朴素HPA（占空比）、队列深度HPA以及KAI组调度扩缩容。报告了未满足的请求、闲置GPU分钟数以及一个综合得分。

## 交付它

本课程产出 `outputs/skill-gpu-autoscaler-plan.md`。根据集群拓扑、工作负载形状和SLO，设计一个三层自动扩缩容计划。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比HPA会丢弃多少个队列深度HPA能捕获的请求？差异来自哪里？
2. 为一个在H100 SXM5上运行Llama 3.3 70B FP8的集群设计Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个阻止非GPU工作负载进入这些节点的taint。
3. 你的团队报告部署一直处于Pending状态，因为“GPU可用但Pod无法调度”。诊断——这是Karpenter、kube-scheduler还是KAI Scheduler的问题？哪些指标可以确认？
4. 为分离式预填充Pod选择一个扩缩容信号，为解码Pod选择另一个不同的信号。证明两者的合理性。
5. 在一个24x7的生产服务上，计算 `WhenEmptyOrUnderutilized` 整合陷阱的成本，该服务平均每天发生60次P99 TTFT > 10s的请求丢弃事件。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Karpenter | “节点配置器” | Kubernetes节点自动扩缩容工具；亚分钟级配置 |
| Cluster Autoscaler | “旧扩缩容器” | Kubernetes节点自动扩缩容前身；更慢、基于组 |
| KAI Scheduler | “GPU调度器” | 用于组调度+拓扑+队列的辅助调度器 |
| 组调度 | “全有或全无” | 原子性调度N个Pod，否则全部推迟 |
| 拓扑感知 | “机架感知” | 基于NVLink/IB/机架放置Pod |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU利用率” | 占空比指标；不是LLM的扩缩容信号 |
| 队列深度 | “等待请求” | 预填充绑定的正确HPA信号 |
| KV缓存利用率 | “内存压力” | 解码绑定的正确HPA信号 |
| 整合 | “Karpenter整合” | 终止节点以迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | “安全整合” | 不驱逐正在运行的GPU任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) —— 设计文档和配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) —— 整合策略语义和GPU安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) —— Dynamo Planner扩缩容信号。
- [Ray文档 — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) —— Ray集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) —— 托管Kubernetes特定指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) —— Workload Variant Autoscaler设计。
