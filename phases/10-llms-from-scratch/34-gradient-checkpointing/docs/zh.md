# 梯度检查点与激活重计算

> 反向传播会保留每一个中间激活值。在 70B 参数、128K 上下文的规模下，每个 rank 的激活值可达 3 TB。检查点用 FLOPs 换内存：重新计算而不是存储。问题在于应该丢弃哪些片段，答案并非“全部丢弃”。

**类型：** 构建  
**语言：** Python（使用 numpy，可选 torch）  
**前置知识：** 阶段 10 第 04 课（预训练 Mini-GPT），阶段 10 第 05 课（扩展与分布式）  
**预计时间：** ~70 分钟

## 问题

训练一个 transformer 会为每一层存储在反向传播中需要求导的每个操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出以及残差流。对于隐藏大小 `d`、序列长度 `L`、批次 `B` 的层，每层大约需要 `12 * B * L * d` 个浮点数。

以 `d=8192, L=8192, B=1` 为例，BF16 下每层为 800 MB。一个 64 层的模型需要 51 GB 的激活值——这还没乘以微批次大小，还没加上注意力-softmax 中间结果（每个头 `L^2`），还没考虑张量并行产生的部分副本。

双重账单：BF16 权重加上优化器状态可能能塞进 80GB，但激活值会把你推过线。梯度检查点（又称激活重计算）是标准的修复方法。丢弃大部分激活值；在反向传播时重新执行前向计算来获取它们。成本：额外的 FLOPs。收益：内存下降的比例等于检查点段数除以总层数。

朴素地做检查点，每一步大约增加 33% 的前向 FLOPs。做得好——根据 Korthikanti 等人的“智能选择”进行选择性检查点——可以节省 5 倍内存，而 FLOP 开销低于 5%。配合 FP8 矩阵乘法、FSDP 卸载以及专家并行的 MoE，这一点变得至关重要：你不能同时承受内存和计算浪费。

## 概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了计算它们，需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 某些激活导数的中间结果（ReLU/GELU/softmax 的导数依赖于激活值）

前向传播会自动将这些存储在 autograd 图中。每个 `tensor.retain_grad()` 以及每个需要其输入的操作都会保留一个引用。

### 朴素的完整检查点

将网络拆分成 `N` 个段。在前向过程中，只存储每个段的*输入*。当反向传播需要中间结果时，重新运行该段的前向过程来物化它们，然后再求导。

示例：32 层 transformer 拆分成 32 个段，每段 1 层。

- 内存：32 个层输入（小） vs 32 * （每层激活体积）（大）。
- 额外计算：每段额外一次前向传播，即总共约多出 33% 的前向 FLOPs（因为反向是前向的 2 倍，完整步骤变成 1 + 1 + 2 = 4 个单位，而不是 1 + 2 = 3）。

这是 Chen 等人 2016 年的原始方案：每 `sqrt(L)` 层设置一个检查点，以平衡内存与计算。对于 L=64，即 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活值的成本相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留廉价存储的激活值（线性投影、残差），仅重新计算昂贵的部分（注意力）。你付出极少的 FLOPs 进行重计算，但节省了 O(L^2) 的内存。

Megatron-Core 将其实现为“选择性”激活重计算。2024 年及之后的许多前沿训练运行都使用了它。

### 卸载

另一种替代重计算的方法：在前向和反向之间将激活值传输到 CPU 内存。需要 PCIe 带宽；当空闲带宽超过重新物化的成本时，这种方法更有益。混合策略很常见：对某些层做检查点，对另一些层做卸载。

FSDP2 将卸载作为一等选项提供。当 GPU 受限于内存但 CPU-GPU 传输尚有余量时，卸载表现出色。

### 重计算成本模型

使用朴素检查点每 `k` 层（共 `L` 层）的每步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，只重新计算注意力核，而不是整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

完整检查点（段大小为 1）：只存储 `L * input_volume`（对于标准 transformer，约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层一个检查点：存储 `L/k * A` 加上活跃段内的 `k-1` 层。

当 `k = sqrt(L)` 时，内存和重计算成本都与 `sqrt(L)` 成比例——这是均匀成本层之间的最优权衡。

### 何时不检查点

- 流水线阶段中已经在处理的内部层。它们无论如何都必须完成。
- 第一层和最后一层，如果它们主导了阶段的计算（在 transformer 中罕见）。
- 已经使用了 FlashAttention 的注意力核——Flash 已经快速重计算了 softmax，因此额外的层级检查点几乎没有增加什么。

### 实现模式

1. **函数包装器：** 使用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一个段。PyTorch 只存储 `input`，在反向传播时重新计算所有其他内容。

2. **装饰器方式：** 将层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己编写反向传播，调用自定义的 `recompute_forward`，该函数使用存储的输入复现前向传播。

这三种方式在功能上结果相同。包装器是标准用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入必须在重计算时进行收集或重新分散；需要处理通信成本。
- **流水线并行：** 典型的模式是对每个流水线阶段的前向进行检查点，以便反向顺序的微批次可以重用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向的一致，否则 FP8 缩放会发生漂移。大多数框架会快照缩放值。

## 构建它

### 步骤 1：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的规范包装器。包装一个函数；只存储输入，在反向传播时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024 年及之后的前沿训练的标准配置。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy`，将激活值卸载到 CPU 而不是重计算。
- **DeepSpeed ZeRO-Offload**：用于优化器状态和激活值的 CPU 卸载，与检查点相辅相成。

## 交付

本课程产出 `outputs/prompt-activation-recompute-policy.md`——一个提示，它接受你的模型配置（层数、隐藏大小、序列长度、批次）和可用的 GPU 内存，并输出每层的重计算策略（无/选择性/完整/卸载）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须与机器精度一致。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOP 开销和内存图。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块的输入但不存储其中间结果。对于 32 层模型，seq=8192，测量与完整层检查点相比的 FLOP 开销。

4. 添加卸载。将段输入保存到模拟的“CPU 缓冲区”（一个独立的列表）中。将“PCIe 带宽”测量为字节/时间，并找到卸载与重计算之间的盈亏平衡点。

5. 在一个真实的 PyTorch transformer 上，使用和不使用 `torch.utils.checkpoint` 进行基准测试。测量内存（通过 `torch.cuda.max_memory_allocated`）和单步时间。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 梯度检查点 | “通过重做前向来节省内存” | 仅存储段输入；在反向传播时重计算中间结果以获取支持梯度的张量 |
| 激活重计算 | “与检查点相同” | 同名技术的 HPC 风格称呼 |
| 段大小 (k) | “每检查点包含几层” | 中间结果被丢弃并一起重新物化的层数 |
| 选择性检查点 | “Korthikanti 的技巧” | 只重计算存储昂贵的激活值（注意力 softmax）；保留廉价的部分 |
| 完整检查点 | “朴素版本” | 在每个段中重计算每一层的中间结果 |
| 块检查点 | “粗粒度” | 检查点整个 transformer 块；最大的粒度 |
| FLOP 开销 | “计算税” | 每步额外 FLOPs = (重计算 FLOPs) / (前向 + 反向 FLOPs)；朴素 33%，选择性 5% |
| 激活卸载 | “传输到 CPU” | 在前向到反向之间将激活值移动到 CPU 内存；重计算的替代方案 |
| sqrt-L 规则 | “经典最优解” | 对于均匀成本的层，最优检查点间距为 sqrt(L) 层 |
| 注意力-softmax 体积 | “O(L^2) 问题” | L^2 * 头数 * 批次浮点数；在长上下文时主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 梯度检查点的原始论文，形式化了该技术
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算及其正式成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 另一种通过反向模式重新物化实现恒定内存的方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 大规模激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、完整和块模式
