# KV缓存、Flash注意力与推理优化

> 训练是并行的且受FLOP限制。推理是串行的且受内存限制。不同的瓶颈，不同的技巧。

**类型：** 构建
**语言：** Python
**前提条件：** 第7阶段·02（自注意力），第7阶段·05（完整Transformer），第7阶段·07（GPT）
**时间：** ~75分钟

## 问题

朴素的自回归解码器在生成长度为 `N` 的token时需要执行 `O(N²)` 的工作量：每一步它都会对完整前缀重新计算注意力。对于4K token的响应，这相当于1600万次注意力操作，其中大部分是冗余的。前缀token的每个隐藏状态一旦被计算就是确定性的——你只需要将新token的查询与之前所有token的缓存键和值进行注意力计算即可。

除此之外，注意力本身移动了大量数据。标准注意力会实例化一个N×N的分数矩阵、N×d的softmax输出以及N×d的最终输出——对HBM的读写次数过多。当N≥2K时，注意力在变为FLOP受限之前就已经是内存受限的了。经典的注意力内核在当代GPU上的利用率低4–10倍。

两项来自Dao等人的优化将前沿推理从“慢”推向了“快”：

1. **KV缓存。** 存储每个前缀token的K向量和V向量。每个新token的注意力只需一次查询与缓存的键进行。推理从每步 `O(N²)` 降为 `O(N)`。
2. **Flash注意力。** 对注意力计算进行分块，使得完整的N×N矩阵永远不会落至HBM。所有的softmax + matmul操作都在SRAM中完成。在A100上获得2–4倍的加速；在H100上使用FP8可获得5–10倍加速。

到2026年，两者都已普遍应用。每一个生产级推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们的存在。每一款前沿模型都默认启用了Flash注意力。

## 概念

![KV缓存增长与Flash注意力分块](../assets/kv-cache-flash-attn.svg)

### KV缓存计算

每解码器层、每token、每头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于7B模型，32层、32头、d_head=128、fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于Llama 3 70B（80层、d_head=128、GQA 8个KV头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这10 GB的KV缓存正是Llama 3 70B在128K上下文下、批大小为1时，几乎占满一块40 GB A100的原因。

**GQA是KV缓存的胜利。** 采用64头的MHA将需要32 GB。MLA则进一步压缩。

### Flash注意力——分块技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次HBM往返。在H100上，HBM带宽为3 TB/s，SRAM为30 TB/s。每次HBM往返相比在芯片上保持所有数据都会带来10倍的减速。

Flash注意力：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每块一次HBM往返。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播时通过前向传播的值重新计算一些值，而不是存储它们——这又是一个内存上的胜利。

**数值技巧。** 逐块执行softmax时，维护一个“（最大值，总和）”对，使得最终归一化是精确的。这不是近似——Flash注意力计算出的输出与标准注意力按位相同（忽略fp16的非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 在参考硬件上的加速比 |
|------|------|---------|-----------------------|
| Flash 1 | 2022 | 分块SRAM内核 | A100上2× |
| Flash 2 | 2023 | 更好的并行性、因果优先排序 | A100上3× |
| Flash 3 | 2024 | Hopper异步、FP8 | H100上1.5–2× (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5级流水线、软件exp2 | 以推理为主（起初仅前向） |

Flash 4 最初仅支持前向传播。训练仍使用Flash 3。Flash 4的GQA和varlen支持待定（2026年中）。

### 投机解码——另一种延迟优势

廉价模型提出N个token。大模型并行验证全部N个token。如果验证接受k个token，你只需一次大模型前向传播就生成了k个token。在代码和散文上，典型k=3–5。

2026年的默认方案：
- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。质量无损下实现2–3倍加速。
- **使用草稿模型的投机解码。** 在消费级硬件上实现2–4倍加速。
- **Lookahead解码。** 基于Jacobi迭代；无需草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等待最慢的序列完成后，再开始一个新批次。当短响应提前完成时，GPU资源被浪费。

连续批处理（首次在Orca中推出，现在已内置于vLLM、TensorRT-LLM、SGLang）：一旦旧序列完成，立即将新请求交换进批次。对于典型对话工作负载，吞吐量提升5–10倍。

### PagedAttention——KV缓存作为虚拟内存

vLLM的头号特性。KV缓存按16-token的块分配；页表将逻辑位置映射到物理块。这使得可以在并行采样（束搜索、并行采样）之间共享KV，为提示缓存热切换前缀，并消除内存碎片。相比朴素的连续分配，吞吐量提升4倍。

## 构建

见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV 缓存解码器。
3. 一个分块 softmax，模拟 Flash 注意力的运行最大值算法。

### 第1步：KV缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中不断为每个token的K、V向量增长。

### 第2步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

计算结果与一次性执行 `softmax(qK) V` 按位相同，但任何时刻的工作集只是一个 `tile × d_head` 块，而非完整的 `N × d_head`。

### 第3步：在100个token生成上比较朴素 vs 缓存解码

统计注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 使用

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产级配置：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是2026年的重大胜利——相同的系统提示、少样本示例或长上下文文档可以在多次调用间重用KV。对于重复工具提示的agent工作负载，前缀缓存通常能获得5倍的吞吐量提升。

## 投产

见 `outputs/skill-inference-optimizer.md`。该技能为一套新的推理部署选择注意力实现、KV缓存策略、量化方法和投机解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器产生相同的输出；注意操作计数的差异。
2. **中等。** 实现前缀缓存：给定一个提示 P 和多个补全，先对 P 执行一次前向传播填充 KV 缓存，然后为每个补全分支。测量与为每个补全重新编码 P 相比的速度提升。
3. **困难。** 实现一个玩具PagedAttention：KV缓存分配为固定16-token的块，并带有一个空闲列表。当一个序列完成时，将其块归还到池中。模拟1000个不同长度的聊天补全。比较与连续分配方式的内存碎片情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| KV缓存 | “让解码变快的技巧” | 存储每个前缀token的K和V；新的查询与之进行注意力计算，而不是重新计算 |
| HBM | “GPU主存” | 高带宽内存；H100上80 GB，B200上192 GB。带宽约3 TB/s |
| SRAM | “片上内存” | 每SM的快速内存，H100上每SM约256 KB。带宽约30 TB/s |
| Flash注意力 | “分块注意力内核” | 计算注意力而不在HBM中实例化N×N矩阵 |
| 连续批处理 | “无等待批处理” | 完成序列换出，新序列换入，无需清空整个批次 |
| PagedAttention | “vLLM的头号特性” | KV缓存按固定块分配并带有页表；消除碎片 |
| 前缀缓存 | “重用长提示” | 跨请求缓存共享前缀的KV；对agent来说极大降低成本 |
| 投机解码 | “草稿+验证” | 廉价草稿模型提出token；大模型一次前向验证k个 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5级流水线和软件exp2技巧；阅读仓库README了解本课程提到的仅前向发布注意事项
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM论文
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2论文，关于课程引用的集成草稿方法
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与EAGLE一同提及的Medusa方法
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于16-token块和页表设计的权威深度解读
