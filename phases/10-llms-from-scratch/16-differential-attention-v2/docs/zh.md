# 差分注意力（V2）

> Softmax注意力会在每个不匹配的token上分配一小部分概率。在超过10万个token上，这种噪声累积起来会淹没信号。Differential Transformer（Ye等人，ICLR 2025）通过将注意力计算为两个softmax的差值，减去共享噪声基底来解决这个问题。DIFF V2（Microsoft，2026年1月）是生产栈的重写版本：匹配基线Transformer的解码延迟，无需自定义内核，兼容FlashAttention。本课程完整覆盖V1到V2，并提供一个在stdlib Python中可运行的差分操作的玩具实现。

**类型：** 构建  
**语言：** Python (stdlib)  
**前置知识：** 阶段7·02（自注意力），阶段7·15（注意力变体），阶段10·14（架构讲解）  
**时长：** 约60分钟

## 学习目标

- 精确阐述softmax注意力为何存在噪声基底以及为何其随上下文长度增长。
- 推导差分注意力公式，并解释为何减法能消除共享噪声分量同时保留信号。
- 走查V1到V2的差异：哪些部分变得更快、更简单、更稳定，以及每个变化对生产预训练的必要性。
- 使用纯Python从头实现差分注意力，并在合成信号加噪声的查询上通过实验验证噪声消除特性。

## 问题

标准softmax注意力具有一个数学特性，在规模化时会变成操作上的难题。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax永远无法产生精确的零——每个不匹配的token都会获得一些正的质量。这些残余质量即为噪声，并且它随上下文长度而增长。在128k个token时，即使每个不匹配的token只获得0.001%的概率，127,999个这样的token合计贡献约12%的总和。模型必须学会绕过一个随上下文增长的噪声基底。

经验上，这表现为注意力头干扰：长上下文RAG中的幻觉引用，100k token检索任务中的“中部丢失”失败，以及超过32k的“大海捞针”基准测试中微妙的准确度下降。Differential Transformer论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer相比相同大小的基线模型，实现了更低的困惑度、更高的长上下文准确度和更少的幻觉。

DIFF V1存在三个问题，使其无法进入前沿预训练流程。其值缓存每次解码步骤需要加载两次，需要自定义CUDA内核从而破坏FlashAttention兼容性，并且其每头RMSNorm在70B+规模的长程训练中变得不稳定。DIFF V2（Microsoft unilm博客，2026年1月20日）修复了这三个问题。本课程走查两个版本，构建差分算子，并在一个玩具查询上基准测试噪声消除。

## 概念

### Softmax的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 并非0——它围绕零波动，方差为 `||q||^2 / d`。经过softmax归一化后，每个无关token依然贡献 `O(1/N)` 给加权和。无关token的总贡献为 `O((N-1)/N) = O(1)` —— 不是一个小量。

模型想要的是一种类似硬top-k的效果：匹配token上权重高，其他地方权重接近零。Softmax过于平滑，无法直接做到这一点。

### 差分思想

将每个头的Q和K投影拆分为两部分：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法能抵消两个注意力图共享的任何噪声分布。如果两个注意力图在127k个无关token上都具有大致均匀的权重（在随机初始化时会是如此），这些噪声就会相互抵消。信号——集中在少数真正相关token上的尖峰权重——只有当它以相同幅度出现在两个注意力图中时才会被抵消，而一旦模型经过训练，这种情况就不会发生。

`lambda` 是每头可学习的标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认是一个小的正数，如0.8。

### 为什么这类似于带头部的噪声消除

想象两个麦克风同时录制同一个声音。两者都拾取说话者的声音以及相关的背景噪声。将一个减去另一个，共享噪声就会消失。声音之所以保留，是因为两个信号在相位或幅度上存在足够差异，防止完全抵消。每头的 `lambda` 正是学习这种平衡。

### V1与V2：差异对比

V1保持参数数量与基线Transformer相同。为了每头获得两个查询，它将头维度减半。这牺牲了头的表达能力，并且更痛苦的是，每头的值缓存减半。解码每一步必须加载值缓存两次（每个softmax分支一次）。结果：尽管参数数量匹配，解码速度仍慢于基线。

V2将查询头的数量加倍，同时保持KV头不变（从up-projection借用参数）。头维度与基线相同。减法后，额外的维度被投影回原始大小，以匹配基线Transformer的O_W投影。同时发生三件事：

1. 解码速度匹配基线（KV缓存只加载一次）。
2. FlashAttention无需修改即可运行（无需自定义内核）。
3. 解码时的算术强度增加（每从HBM加载一个字节执行更多计算）。

V2还移除了V1用于稳定减法的每头RMSNorm。在70B级别的预训练规模下，该RMSNorm使后期训练不稳定。V2用一个更简单的初始化方案替代它，无需额外模块即可保持训练稳定。

### 何时使用

| 工作负载 | 收益 |
|----------|------|
| 长上下文RAG（64k+） | 更清晰的注意力图，更少的幻觉引用 |
| 大海捞针基准测试 | 超过32k后显著准确度提升 |
| 多文档问答 | 更少的跨文档干扰 |
| 8k上下文代码补全 | 边际收益，不值得改变架构 |
| 短聊天（< 4k） | 与基线基本无区别 |

该价值随上下文长度增长。在4k token时，噪声基底足够小，标准注意力就够用。在128k时，它正在损害性能。

### 与其他2026年特性的兼容性

| 特性 | 与DIFF V2兼容？ |
|------|-----------------|
| GQA | 是（V2增加Q头，而非KV头） |
| MLA（DeepSeek） | 原则上兼容，但尚无公开论文结合两者 |
| MoE | 是（注意力与MLP模块独立） |
| RoPE | 是（保持不变） |
| YaRN / 长上下文缩放 | 是（这正是DIFF最有用的地方） |
| FlashAttention | V2中兼容（V1中不兼容） |
| 推测解码 | 是（注意力变化对推测解码循环不可见） |

## 构建

`code/main.py` 使用纯Python实现差分注意力。一个具有已知信号加噪声结构的玩具查询能让您直接测量噪声消除比。

### 步骤1：标准softmax注意力

标准库矩阵操作：列表的列表、手动矩阵乘法、带有数值稳定性最大值减法的softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤2：将Q、K拆分为两半

V1风格：将头维度减半。V2风格：保持头维度不变，将头数量加倍。玩具实现采用V1以便教学清晰——数学上相同，只是记账方式不同。

### 步骤3：两个softmax分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可能为负。这没问题——值缓存仍然处理有符号贡献。后续的V投影会吸收符号。

### 步骤4：噪声消除测量

构建一个长度为1024的合成序列。将信号token放在已知位置，其余部分填充噪声。计算(a)信号位置上的标准softmax注意力权重和(b)差分注意力权重。测量每个中的信噪比。DIFF注意力可靠地产生更高的信噪比，根据两个分支训练后的差异程度，因子在3倍到10倍之间。

### 步骤5：V1与V2参数统计

给定一个配置（hidden=4096, heads=32, d_head=128），输出：

- 基线Transformer：Q、K、V各大小 `hidden * hidden`，MLP为4 * hidden。
- DIFF V1：Q、K各大小 `hidden * hidden`，V大小 `hidden * hidden`（不变），内部头维度减半。增加每头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q大小 `2 * hidden * hidden`，K大小 `hidden * hidden`，V大小 `hidden * hidden`。额外维度在O_W之前投影回原尺寸。增加相同 `lambda` 参数。

玩具实现测量V2的额外参数成本（每个注意力块大约 `hidden * hidden` 额外参数）并输出。

## 使用它

截至2026年4月，DIFF V2尚未在每一个生产推理服务器中部署，但在vLLM和SGLang中正在进行集成。同时，该模式出现在以下场景中：

- Microsoft内部的长上下文生产模型。
- 针对256k以上上下文的一些开放模型训练运行中的研究复现。
- 将DIFF注意力与滑动窗口注意力交替层结合的混合架构。

在2026年何时使用它：

- 从零开始训练一个目标有效上下文64k以上的新模型。从一开始就添加差分注意力；事后重新训练代价高昂。
- 微调一个长上下文模型，其中“中部丢失”失败主导了您的评估。对Q投影进行LoRA可以近似DIFF结构。

何时不使用：

- 您正在服务于一个预训练的密集模型，且长上下文性能稳定。重新训练的成本很少能从现有权重得到回报。
- 您的上下文始终低于16k。噪声基底可以忽略不计。

## 部署

本课程生成 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉概况和训练预算，它会生成一个将差分注意力添加到新预训练运行或LoRA微调中的集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力在合成查询上的信噪比高于标准softmax注意力。改变噪声幅度，并展示标准注意力变得不可用的交叉点。

2. 对于一个7B级模型（hidden=4096, heads=32, d_head=128, 32层），计算从基线到DIFF V1以及从基线到DIFF V2的参数数量增量。展示哪些组件增加了参数，哪些保持不变。

3. 阅读DIFF V1论文（arXiv:2410.05258）第3节和DIFF V2 Hugging Face博客第2节。用两句话解释为什么V1的每头RMSNorm是必要的，以及V2为何能将其移除而不造成训练发散。

4. 实施一个消融实验：分别计算 `lambda = 0`（纯第一个softmax）和 `lambda = 1`（完全减法）时的差分注意力。在合成查询上，测量信噪比随lambda扫过的变化。找出最大化信噪比的 `lambda`。

5. 将玩具扩展到GQA + DIFF V2。选择8个KV头和32个Q头。证明KV缓存大小与具有相同(8, 32)配置的基线GQA模型匹配。

## 关键术语

| 术语 | 人们说它是什么 | 它实际意味着什么 |
|------|----------------|------------------|
| Differential attention | “两个softmax互相减” | 将Q、K拆分为两半，计算两个softmax图，从第一个减去第二个（乘以lambda缩放），然后乘以V |
| Noise floor | “softmax的非零尾部” | Softmax赋予每个无关token的O(1/N)权重，在长上下文中总和为O(1) |
| lambda | “减法缩放因子” | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可为负 |
| DIFF V1 | “ICLR 2025版本” | 原始Differential Transformer；将头维度减半以保持参数数量，需要自定义内核，解码较慢 |
| DIFF V2 | “2026年1月修复” | Q头数量加倍，保持KV头不变；匹配基线解码速度，兼容FlashAttention |
| Per-head RMSNorm | “V1稳定器” | V1在差分后应用的额外归一化；V2移除以防止训练后期不稳定 |
| Signal-to-noise ratio | “注意力有多少被浪费” | 真正信号位置上的权重与无关位置平均权重之比 |
| Lost in the middle | “长上下文失败模式” | 一个经验现象：长上下文中位于中间部分的文档检索准确率下降——DIFF注意力能减轻此问题 |
| Arithmetic intensity | “每加载字节的FLOPs” | V2通过每KV加载加倍查询数量而在解码时增加的比率；对受内存限制的解码很重要 |

## 延伸阅读

- [Ye等人 — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声消除理论和长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face博客，2026年1月)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线解码速度，兼容FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 关于减法如何恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani等人 — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF从中减去的基线Transformer
- [Liu等人 — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF注意力所针对的长上下文基准
