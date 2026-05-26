# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在各节点之间。跨节点专家全对全通信每消耗 1 GPU 小时的计算就需要 1 GPU 小时的通信。GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它将前向和反向计算与它们触发的全对全通信重叠。气泡减少，吞吐量提升，而保留两份模型参数副本（即名称中的“dual”）的代价在专家并行已经将专家分散到各 rank 的情况下微不足道。本课程以学习型（Learn）讲解方式介绍 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进版以略微收紧的气泡为代价，去掉了 2 倍参数开销。

**类型：** 学习型  
**语言：** Python（标准库，调度模拟器）  
**先修条件：** 阶段 10 · 05（分布式训练，FSDP，DeepSpeed），阶段 10 · 14（开源模型架构和 MoE）  
**时间：** ~60 分钟

## 学习目标

- 列举 DualPipe 前向-反向块的四个组成部分，并解释每个部分为何拥有自己的重叠窗口。
- 解释大规模下的流水线气泡问题，以及实践中“无气泡”与营销上的含义有何不同。
- 手动为 8 个 PP rank 和 16 个微批次追踪 DualPipe 调度，确认前向流和反向流彼此填充空闲槽。
- 说明 DualPipeV（Sea AI Lab，2025）所做的权衡：在专家并行未激活时，以略大的气泡为代价，去掉了 2 倍参数复制。

## 问题

在 2k 块 H800 GPU 上训练 671B MoE 模型会遇到三个相互叠加的瓶颈：

1. **内存压力。** 每块 GPU 持有模型的一部分。序列长度 8k、61 层、128 头的激活内存巨大。
2. **流水线气泡。** 传统的流水线并行（GPipe，1F1B）使得 GPU 在等待其阶段的输入或梯度时空闲。在 8 阶段时，即使采用 1F1B 调度，大约 12% 的 GPU 时间可能仍是气泡。
3. **跨节点全对全。** 采用专家并行的 MoE 将专家分散到各节点。每次前向传播都会触发一次全对全将 token 分发到其专家，以及另一次全对全进行合并。在 2k GPU 规模下，这很容易导致 1:1 的计算与通信比率。

每一个瓶颈都有独立的解决方案：梯度检查点用于内存，Zero Bubble（Sea AI Lab，2023）用于流水线气泡，专家并行通信内核用于全对全。DualPipe 所做的就是让它们协同工作。该调度在单个前向-反向块内重叠计算和通信，同时从流水线两端注入微批次，并利用得到的调度将全对全隐藏在计算窗口内。

报告结果：几乎消除流水线气泡，在 DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 概念

### 流水线并行复习

将一个 N 层模型拆到 P 个设备上。设备 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 层。一个微批次向前流过设备 0 到 P-1，然后向后从 P-1 到 0。每个设备只有在前一个设备发送其输出后才能开始其前向阶段，并且只有在下游设备发送上游梯度后才能开始反向。

GPipe（Huang et al., 2019）一次调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan et al., 2021）为多个微批次交错执行前向和反向。Zero Bubble（Qi et al., 2023）将反向传播拆分为两部分——反向计算输入梯度（B）和反向计算权重梯度（W）——并安排它们填充气泡。在 Zero Bubble 之后，流水线几乎被填满。

DualPipe 是下一步。它在上述基础上增加了两个想法：

### 想法 1：块分解

每个前向块被拆分为四个部分：

- **注意力（Attention）。** Q/K/V 投影、注意力、输出投影。
- **全对全分发（All-to-all dispatch）。** 跨节点通信，将 token 发送到其专家。
- **MLP。** MoE 专家计算。
- **全对全合并（All-to-all combine）。** 跨节点通信，将专家输出带回。

反向块则包含每个部分的梯度版本。DualPipe 安排它们使得全对全分发现象与下一个块的注意力计算同时发生，全对全合并则与后续块的 MLP 计算同时发生。

### 想法 2：双向调度

大多数流水线调度从阶段 0 注入微批次，流向阶段 P-1。DualPipe 从两端注入微批次。阶段 0 看到源自那里前向微批次；阶段 P-1 同样看到源自那里的前向微批次。两个流在中间相遇。

为此，设备 `i` 必须同时持有流水线早期层 `i` 和流水线晚期层 `P - 1 - i`。这就是 DualPipe 中“dual”的部分：每个设备保留两份它需要服务的模型层副本（每个方向一份）。在 DeepSeek-V3 的规模下，这是一个 2 倍参数复制代价。之所以可以承受，是因为专家并行已经将 MoE 专家分散得足够稀薄，复制非专家层两次只是小开销。

关键的是，一个方向的前向流与另一个方向的反向流恰好重叠在单向调度中本会出现气泡的位置。气泡消失了。

### 手动追踪的调度

考虑 P = 4 个 rank，8 个微批次，分为 4 个前向 / 4 个反向。时间从左向右移动；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 "F4/F5R" 符号：rank 1 在同一时间槽内同时运行微批次 4 的前向（在流水线中从左向右）和微批次 5 的前向（从右向左）。这就是“双向”的操作含义。

在 rank 2 处，交叉流更早重叠；在 rank 0 和 P-1 处，重叠最晚。在调度的稳定中间阶段，每个 rank 都运行与反向 of-Y-direction 重叠的前向 of-X-direction。计算繁忙。前向传播的全对全分发隐藏在反向计算中。全对全合并隐藏在前向计算中。气泡被挤压出去。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 的改进将其降低，但未到零。DualPipe 在稳定阶段，如果微批次数量是流水线深度的两倍的整数倍，则气泡为零。在稳定阶段之外（预热和冷却），存在一些气泡，但不会随着微批次数量增长——这是论文强调的一个关键属性。

用营销术语说：“无气泡”。用技术术语说：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，只有当专家并行不是瓶颈时才能完全无气泡；在 EP 驱动的全对全情况下，总是存在某种调度折中。

### DualPipeV——改进

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠成一种“V 形”调度，运行在单个参数副本上。气泡比 DualPipe 稍大，但内存节省显著。DeepSeek 在其开源 DualPipe 实现中采用 DualPipeV 作为 EP 关闭模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|------|----------|-----------|------|-------------|
| 每个设备的参数副本数 | 2 | 1 | 1 | 1 |
| 气泡 vs 微批次 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 使用时机 | EP 繁重的 MoE | 密集或 EP 较轻 | 基线 | 任何流水线 |

### 对 14.8T token 训练的意义

DeepSeek-V3 的预训练消耗了 14.8T token，在 2,048 块 H800 GPU 上耗时约 280 万 GPU 小时。如果采用朴素的 1F1B，他们将损失 12-15% 的流水线气泡——34万-42万 GPU 小时，足以训练一个完整的 70B 模型。DualPipe 恢复了其中大部分。如果没有内部日志，很难直接量化其贡献，但论文声称训练平均 GPU 利用率超过 95%。

对于较小的训练（不到 1k GPU），DualPipe 是大材小用——流水线气泡相对于总成本较小，且密集模型训练很少遇到全对全瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在技术栈中的位置

- 与 **FSDP**（阶段 10 · 05）互补。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。它们可以结合。
- 与 **ZeRO-3** 梯度分片兼容。两份副本复制的记账需要与 ZeRO 的分片梯度协作。
- 需要针对特定集群拓扑调优的**自定义全对全内核**。DeepSeek 的开源内核是参考实现。

## 使用

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 参数，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段的利用率。这是一个教学工具——数字与论文中的定性声称一致，并不代表生产环境实测加速比。

该模拟器的价值：用不同的 P 和微批次数量运行它，观察 1F1B 的气泡分数如何增长，而 DualPipe 不增长。

实际训练运行中的集成注意事项：

- 选择能整除微批次数量的流水线并行深度。
- 确保你的专家并行网格支持双向全对全。DeepSeek 的内核是参考。
- 第一次使用该调度时，预计会花一周时间调试。记账工作很繁琐。
- 监控每个 rank 的 GPU 利用率，而不是聚合值。DualPipe 的优势在于收紧掉队者。

## 部署

本课程生成 `outputs/skill-dualpipe-planner.md`。给定一个训练集群规格（GPU 数量、拓扑、互连、模型形状），它会推荐流水线并行策略、要使用的调度算法，以及在目标规模下的预期气泡分数。

## 练习

1. 运行 `code/main.py`，参数为 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)`。计算 GPU 利用率差异，并将其表示为每百万 token 训练恢复的 GPU 小时数。

2. 手动绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时间槽。识别第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。识别 DualPipe 前向块中全对全分发的重叠窗口。解释计算调度如何隐藏它。

4. 计算 DualPipe 的 2 倍参数开销：对于 P=8 流水线阶段的 70B 密集模型，以及 P=16 流水线阶段的 671B MoE 模型。说明为什么 MoE 情况的开销比例更小（大部分参数是专家，分布在一个大的 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的竞争性双向调度器）进行比较。以论文第 3.4 节为参考，确定 DualPipe 添加而 Chimera 没有的两个具体属性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 流水线气泡（Pipeline bubble） | “每个 rank 的空闲时间” | GPU 周期浪费，因为流水线阶段正在等待其输入或梯度 |
| 1F1B | “默认流水线调度” | 一个前向 / 一个反向交错调度；DualPipe 超越的基线 |
| Zero Bubble | “Sea AI Lab 2023” | 将反向拆分为 B（输入梯度）和 W（权重梯度）；几乎完全填满流水线 |
| DualPipe | “DeepSeek-V3 调度” | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | “Cut-in-half” | V 形改进，以略大气泡为代价去掉 2 倍参数复制 |
| 块（Chunk） | “流水线工作单元” | 一个微批次通过一个流水线阶段的前向或反向 |
| 全对全分发（All-to-all dispatch） | “将 token 发送到专家” | 跨节点通信，将 token 路由到其分配的 MoE 专家 |
| 全对全合并（All-to-all combine） | “将专家输出带回” | 跨节点通信，在 MLP 后收集专家输出 |
| 专家并行（Expert Parallelism，简称 EP） | “跨 GPU 的专家” | 将 MoE 专家跨 rank 分片，使得不同 GPU 持有不同专家 |
| 流水线并行（Pipeline Parallelism，简称 PP） | “跨 GPU 的层” | 将模型层跨 rank 分片；DualPipe 调度的维度 |
| 气泡分数（Bubble fraction） | “浪费的 GPU 时间” | (气泡时间 / 总时间)；DualPipe 将其推向零 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考文献
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — DualPipeV 分析，为 DeepSeek 的 EP 关闭模式提供信息
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 与之对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文及气泡问题
