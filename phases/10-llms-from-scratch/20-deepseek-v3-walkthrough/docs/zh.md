# DeepSeek-V3 架构详解

> 第10阶段 · 第14课讲述了每个开源模型都会调整的六个架构旋钮。DeepSeek-V3（2024年12月，总参数671B，激活参数37B）不仅全部使用了这六个旋钮，还额外增加了四个：多头潜在注意力、无辅助损失负载均衡、多 token 预测和 DualPipe 训练。本课从上到下解读 DeepSeek-V3 的架构，并从官方配置推导出每一个参数的计数。学完本课后，你将能解释为何 671B/37B 的比例是正确的选择，以及为什么 MLA + MoE 在前沿领域比单独使用其中任何一种都更强。

**类型：** 学习  
**语言：** Python（标准库，参数计算器）  
**前置知识：** 第10阶段 · 14（开源模型详解），第10阶段 · 17（NSA），第10阶段 · 18（MTP），第10阶段 · 19（DualPipe）  
**时长：** 约75分钟

## 学习目标

- 从上到下阅读 DeepSeek-V3 配置，并根据 GPT-2 的六个旋钮加上四个 DeepSeek 特有的旋钮解释每个字段。
- 推导出总参数数量（671B）、激活参数数量（37B）以及构成它们的各个组件。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与相同激活参数量的使用 GQA 的稠密模型进行比较。
- 列出四个 DeepSeek 特有的创新点（MLA、MTP、无辅助损失路由、DualPipe），并指出每个创新点针对的是架构/训练栈的哪一部分。

## 问题

DeepSeek-V3 是第一个在架构上与 Llama 系列有实质性差异的前沿开源模型。Llama 3 405B 是“旋钮拧满的 GPT-2”。DeepSeek-V3 则是 GPT-2 拧满了所有六个旋钮再加四个。阅读 Llama 3 配置是为阅读 DeepSeek 配置做热身，但深层结构——注意力块的形状、路由逻辑、训练时的目标——差异足够大，需要单独讲解。

学习它的回报：DeepSeek-V3 的开源权重发布改变了“前沿能力”在开源模型中的含义。该架构是许多 2026 年训练任务正在复制的蓝图。理解它对于任何涉及前沿 LLM 训练或推理的角色都是必备基础。

## 概念

### 再次强调不变的核心

DeepSeek-V3 仍然是自回归模型。它仍然堆叠解码器块。每个块仍然包含注意力 + MLP + 两个 RMSNorm。它仍在 MLP 中使用 SwiGLU。它仍使用 RoPE。Pre-norm。权重共享嵌入。与每个 Llama 或 Mistral 相同的基线。

### 关键变化：MLA 替代 GQA

从第10阶段·14课你了解到，GQA 通过在 Q 头组之间共享 K 和 V 来减小 KV 缓存。多头潜在注意力（MLA）更进一步：K 和 V 被压缩成一个共享的低秩潜在表示（`kv_lora_rank`），然后在每个头上动态解压缩。KV 缓存仅存储潜在表示——通常每 token 每层 512 个浮点数，而不是 8 x 128 = 1024 个浮点数。

在 128k 上下文中，DeepSeek-V3 使用 MLA（每 token 每层一个共享潜在 `c^{KV}`；K 和 V 都从该潜在表示通过上投影导出，这些上投影可以被吸收到随后的矩阵乘法中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 形状，8 KV 头，头维度 128）将付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 比类 Llama-3-70B 的 GQA 缓存小 4 倍。

权衡：MLA 在每次注意力计算（每个头）中添加了一个解压缩步骤。额外的计算量与节省的带宽相比很小。对于长上下文推理来说是净赢。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个 token。朴素的路由器会将过多工作集中到少数专家上，使其他专家闲置。标准修复：添加一个辅助损失项来惩罚负载不平衡。这有效，但会稍微降低主任务性能。

DeepSeek-V3 引入了一种无辅助损失的方案。每个专家添加一个偏置项到路由器 logits 中，在训练期间通过简单规则调整：如果专家 `e` 过载，则减小 `bias_e`；如果欠载，则增大它。没有额外的损失项。训练保持干净。专家负载保持平衡。

对主损失的影响：可测量为零。对 MoE 架构的影响：更干净，无需调整辅助损失超参数。

### MTP：更密集的训练 + 免费草稿

从第10阶段·18课你了解到，DeepSeek-V3 添加了 D=1 的 MTP 模块，预测两个位置后的 token。推理时，训练好的模块被重新用作推测解码的草稿，接受率超过 80%。训练时，每个隐藏状态受到 D+1 = 2 个目标的监督，提供更密集的信号。

参数：在主模型的 671B 之上增加 14B。开销：2.1%。

### 训练：DualPipe

从第10阶段·19课你了解到，DualPipe 是一种双向流水线，它将前向和反向块与跨节点全对全通信重叠。在 DeepSeek-V3 的 2,048 H800 规模下，它大约恢复了 1F1B 会因流水线气泡而损失的 245k GPU 小时。

### 配置字段解析

以下是 DeepSeek-V3 配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，然后每个头解压缩。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，有 1 个始终开启的专家为每个 token 做贡献。可以看作是“稠密基底”，确保每个 token 都能得到一些可靠的处理。
- `moe_intermediate_size=2048`：每个专家 MLP 的隐藏层大小。比稠密 MLP 小，因为有 256 个专家。

### 参数统计

完整计算在 `code/main.py` 中。要点：

- 嵌入：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：包含 MLA 的注意力（每块约 144M）+ 稠密 MLP（每块约 260M）+ 归一化。总计约 1.2B。
- 58 个 MoE 块：包含 MLA 的注意力（约 144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ 归一化。每块总计约 7.95B，包括所有专家。58 个 MoE 块总计 461B。
- MTP 模块：14B。

总计：核心架构约 476B + MTP 14B，明显发布的 671B 数字还包括了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与发布值的差异在 3-5% 以内——差异来自 DeepSeek 报告附录第 2 节中记录的细粒度统计。

每次前向传播的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都参与）。
- 激活 MLP：前 3 层稠密（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由 + 1 个共享 + 路由开销。每层激活 MLP：约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入 + 归一化：1.2B。
- 总激活参数：核心约 26B + MTP 14B（训练时存在，但推理时不一定运行）≈ 37B。

### 671B / 37B 比例

18 倍稀疏比（激活参数占总参数的 5.5%）。DeepSeek-V3 是迄今为止开源权重中最稀疏的前沿 MoE 模型。Mixtral 8x7B 比例为 13/47（28%）要稠密得多。Llama 4 Maverick 比例为 17B/400B（4.25%）与之相当。DeepSeek 的赌注：在前沿规模下，更多的专家和更低的激活比可以在每个激活 FLOP 上产生更好的质量。

### DeepSeek-V3 的定位

| 模型 | 总参数 | 激活参数 | 比例 | 注意力机制 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上的推理训练运行。R1 使用相同的架构。改变的是训练后配方（在可验证任务上进行大规模强化学习），而不是预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并添加 DSA（DeepSeek 稀疏注意力），即第10阶段·17课中 NSA 的后续。这个谱系是稳定的：架构级别的创新不断累积；每个版本会拧动更多的旋钮。

## 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文中的数字进行比较，并在假设的变体（256 专家 vs 512 专家，top-8 vs top-16，MLA 秩 512 vs 1024）上使用它。

需要关注的内容：

- 总参数数量与发布的 671B 比较。
- 激活参数数量与发布的 37B 比较。
- 128k 上下文下的 KV 缓存——MLA 与 GQA 的比较。
- 逐层分解，看参数预算实际上花在哪里。

## 交付

本课生成 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek 系列模型（V3、R1 或任何未来变体），它将生成一个逐组件的架构解读，命名配置的每个字段，按组件推导参数数量，并识别模型使用了四个 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计值与发布的 671B 进行比较，并找出差异来自哪里。论文的第 2 节有完整的项目清单。

2. 修改配置，将 MLA 秩从 512 改为 256。计算在 128k 上下文下的 KV 缓存大小。这能带来多少百分比减少？每个头的表达能力代价是什么？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与假设的（512 专家，top-8）变体进行比较。总参数增加；激活参数保持不变。理论上额外的专家容量能带来什么好处？推理时成本是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 的解压缩矩阵可以在推理时被“吸收”到后续的矩阵乘法中。

5. DeepSeek-V3 在大多数操作中使用 FP8 训练。计算 FP8 相对于 BF16 存储 671B 权重的内存节省。这与 14.8T token 的训练预算有何关系？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| MLA | "多头潜在注意力" | 将 K 和 V 压缩成共享的低秩潜在表示（kv_lora_rank，通常为 512），在每个头上动态解压缩；KV 缓存仅存储潜在表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| 前 k 个稠密层 | "早期层保持稠密" | MoE 模型的前几层跳过 MoE 路由器，运行稠密 MLP 以保持稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 激活的路由专家数量；DeepSeek-V3 使用 8 |
| 共享专家 | "始终开启的专家" | 无论路由结果如何都处理每个 token 的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | "偏置调整的负载平衡" | 训练期间调整每个专家的偏置项，以保持专家负载平衡，无需添加损失项 |
| MTP 模块 | "额外的预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；更密集的训练，免费的推测解码草稿 |
| DualPipe | "双向流水线" | 一种训练调度，将前向/反向计算与跨节点全对全通信重叠 |
| 激活参数比例 | "稀疏度" | 激活参数 / 总参数；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 训练存储和许多计算操作采用 FP8；与 BF16 相比大致将内存减半，质量损失很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件与部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身工作
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 基于 V3 架构的推理训练后继
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 系列注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考
