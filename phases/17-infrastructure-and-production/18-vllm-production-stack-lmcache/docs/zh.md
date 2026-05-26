# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的生产栈是参考性的 Kubernetes 部署——路由器、引擎和可观测性集成在一起。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来并在多个查询和引擎之间复用（先到 CPU DRAM，再到磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（2026 年 1 月）通过连接器 API（v0.9.0+）使其异步且可插拔。卸载延迟对用户无感知。即使没有共享前缀，LMCache 也很有价值——当 GPU 的 KV 槽位用尽时，被抢占的请求可以从 CPU 恢复，而无需重新计算预填充。已发布的基准测试在 16 块 H100（80GB HBM）横跨 4 台 a3-highgpu-4g 上：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都能显著提升吞吐量；在低 KV 占用下，所有配置与基线匹配，只有微小开销。

**类型：** 学习
**语言：** Python（stdlib，模拟 KV 溢出的小玩具）
**先修条件：** Phase 17 · 04（vLLM 服务内部机制），Phase 17 · 06（SGLang/RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM 生产栈的各层：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 定量评估 LMCache CPU-DRAM 何时有帮助（KV > HBM）何时增加开销（KV 小到足以放入 HBM）。
- 根据部署约束，在原生 vLLM CPU 卸载和 LMCache 连接器之间做出选择。

## 问题

你的 vLLM 服务显示 GPU HBM 占用率 100%，并且每当并发升高时就会出现抢占事件。请求被驱逐、重新排队，你会在一分钟内对同一个 2K token 的提示进行四次预填充。GPU 算力浪费在冗余的预填充上；有效吞吐量远低于原始吞吐量。

增加更多 GPU 的成本是线性的。无法增加更多 HBM。但 CPU DRAM 很便宜——单个插槽有 512 GB 以上，延迟虽然比 HBM 差几个数量级，但对于“临时温” KV 缓存来说已经足够。

LMCache 将 KV 缓存提取到 CPU DRAM，使得被抢占的请求能够快速恢复，并且不同引擎之间的重复前缀可以共享缓存，无需每个引擎重新预填充。

## 概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考性的 Kubernetes 部署：

- **路由器** —— 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **引擎** —— vLLM 工作节点。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载** —— LMCache 部署或原生连接器。
- **可观测性** —— Prometheus 抓取、Grafana 仪表盘、OTel 追踪。
- **控制平面** —— 服务发现、配置、滚动更新。

以 Helm chart + operator 形式提供。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的连接器 API。引擎将块卸载到连接器；连接器存储它们（RAM、磁盘、对象存储、LMCache）。请求需要某一块时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步卸载路径——卸载可以在后台进行，因此引擎在常见情况下不会因此阻塞。端到端的延迟和吞吐量仍然取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出，自定义内核卸载可能在低命中率时降低吞吐量，并且异步调度与推测解码存在已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。将 KV 块存储在主机 RAM 中。实现简单，零网络跳转。不跨引擎。

**LMCache 连接器**：集群规模。将块存储在共享的 LMCache 服务器中（CPU DRAM + Ceph/S3 层）。任何引擎都可以访问这些块。已发布 16 块 H100 的基准测试。

当单个引擎面临 HBM 压力时，选择原生卸载。当多个引擎共享前缀时（RAG 中使用公共系统提示，多租户中使用共享模板），选择 LMCache。

### 基准测试行为

16 块 H100（80 GB HBM）分布在 4 台 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示、低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 的开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提升吞吐量；LMCache 收益更大，因为跨引擎共享。

### 何时 LMCache 是关键选择

- 多租户服务，其中系统提示在租户之间共享。
- RAG 场景，文档块在查询中重复。
- 基于相同基座的微调变体（LoRA），基座模型的 KV 复用减少了冗余计算。
- 抢占较重的工作负载：从 CPU 恢复比重新预填充更便宜。

### 何时不应启用

- HBM 压力很小——你支付了开销而没有收益。
- 短上下文（<1K tokens）——传输时间超过重新预填充。
- 单租户单提示工作负载——没有复用可捕捉。

### 与解耦服务的集成

Phase 17 · 17 解耦服务 + LMCache 叠加：从预填充池到解码池的 KV 传输在未被使用时落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 缓存感知路由器可以将请求路由到其本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（不是绝对保证）。
- 16 块 H100 基准测试：当 KV 占用超过 HBM 时 LMCache 有帮助。
- 低 HBM 压力：3-5% 的开销，没有收益。

## 使用它

`code/main.py` 模拟了一个抢占较重的工作负载，分别带有和不带 LMCache。报告避免的重新预填充次数、吞吐量增益以及盈亏平衡的 HBM 利用率。

## 交付它

本课程会生成 `outputs/skill-vllm-stack-decider.md`。根据工作负载形状和 vLLM 部署，决定原生卸载、LMCache 还是两者都不用。

## 练习

1. 运行 `code/main.py`。LMCache 在 HBM 利用率达到多少时开始产生收益？
2. 一个租户在所有 200 个查询/小时中共享一个 6K token 的系统提示。计算每个租户预期的 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、故障回退到原生）。
4. LMCache 将数据存储在旋转磁盘的 Ceph 上。对于一个 4K token 的 KV（70B FP8，500 MB），读取时间与重新预填充相比如何？
5. 论证 vLLM 0.11.0 的异步路径是否“免费”——开销隐藏在何处？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 生产栈 | “参考部署” | vLLM 的 Kubernetes Helm chart + operator |
| 连接器 API | “KV 后端接口” | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载 | “引擎本地溢出” | 将 KV 存储在相同引擎的主机 RAM 中 |
| LMCache | “集群 KV 缓存” | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 异步 | “非阻塞卸载” | 卸载隐藏在引擎流之后 |
| 抢占 | “逐出以腾出空间” | HBM 满时 KV 缓存的重新洗牌 |
| 前缀复用 | “相同系统提示” | 多个查询共享开头；缓存命中 |
| Ceph 层 | “磁盘层” | 缓存层级中 DRAM 之下的持久存储 |

## 进一步阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径细节。
