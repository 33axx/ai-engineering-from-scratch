# 注意力机制变体——滑动窗口、稀疏与差分注意力

> 完全注意力是一个圆。每个词元都能看到每个词元，而内存则为此付出代价。四种变体改变了圆的形状，并回收了一半成本。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 7 · 02（自注意力），阶段 7 · 03（多头注意力），阶段 7 · 12（KV 缓存 / Flash Attention）  
**预计时间：** ~60 分钟

## 问题

完全注意力在序列长度上消耗 `O(N²)` 的内存和 `O(N²)` 的计算量。对于一个 128K 上下文的 Llama 3 70B 模型，每层有 160 亿个注意力条目，共 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但并未改变算术成本——每个词元仍然关注所有其他词元。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA）。** 每个词元只关注一个固定的相邻窗口，而不是整个前缀。内存和计算量降至 `O(N·W)`，其中 `W` 为窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 块注意力。** 只对选定的 `(i, j)` 对进行评分；其余对强制为零权重。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 使用独立的 Q/K 投影计算两个注意力图，将两者相减。消除了将权重泄漏到前几个词元的“注意力下沉”。微软的 DIFF Transformer（2024）。

这些变体可以共存。一个 2026 年的前沿模型通常会混合使用：多数层是 SWA-1024，每五层有一个全局完全注意力层，少数层是用于清理检索的差分注意力头。Gemma 3 的 5:1 SWA 与全局层比例是目前教科书的默认设置。

## 概念

### 滑动窗口注意力（SWA）

每个位置 `i` 的查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）范围内的位置。窗口外的词元在得分矩阵中设置为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，得分矩阵中非零行数期望为 1024 × 8192——减少了 8 倍。

**KV 缓存随 SWA 缩小。** 每层只需保留最后 `W` 个词元的 K 和 V。对于类似 Gemma-3 的配置（1024 窗口，128K 上下文），KV 缓存减少 128 倍。

**质量代价。** 纯 SWA Transformer 在长距离检索上表现不佳。解决方案：在 SWA 层间穿插完全注意力层。Gemma 3 使用 5:1 的 SWA 与全局层比例。Mistral 7B 使用因果 SWA 堆叠，信息通过重叠窗口“向前流动”——每层将有效感受野扩展 `W`，经过 `L` 层后模型能关注 `L × W` 个词元之后。

### 稀疏 / 块注意力

预先选择一个 `N × N` 的稀疏模式。三种经典形状：

- **局部 + 跨步（OpenAI 稀疏 Transformer）。** 关注最后 `W` 个词元加上该词元之前的每隔 `stride` 步的词元。以 `O(N·√N)` 的计算量同时捕获局部和长距离信息。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局词元（例如 `[CLS]`），这些全局词元关注所有人且被所有人关注，再加上随机稀疏连接。在相同质量下，经验上可支持 2 倍的上下文长度。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块是重要的；在内核级别跳过零块。兼容 FlashAttention。

稀疏注意力是一个内核工程问题。其数学原理很简单（对得分矩阵进行掩码）；优势在于永远不需要将零条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式成为 PyTorch 中的一等公民。

### 差分注意力（DIFF Transformer，2024）

常规注意力存在“注意力下沉”问题：softmax 强迫每行求和为 1，因此那些不想关注任何特定内容的词元会将权重倾泻到第一个词元（或前几个词元）上。这窃取了本应用于真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来解决此问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习的标量（通常为 0.5–0.8）。A1 捕获真实内容权重；A2 捕获下沉。相减可消除下沉，将权重重新分配给相关词元。

报告的结果（微软 2024）：困惑度降低 5–10%，相同训练长度下有效上下文长度增加 1.5–2 倍，大海捞针检索更加精确。

### 变体对比

| 变体 | 计算量 | KV 缓存 | 与完全注意力相比的质量 | 生产使用情况 |
|------|--------|---------|------------------------|--------------|
| 完全注意力 | O(N²) | 每层 O(N) | 基准 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，与全局层配合良好 | Gemma 2/3, Phi-3-Long |
| 局部 + 跨步稀疏 | O(N·√N) | 混合 | 与 SWA 类似 | OpenAI sparse transformer, Longformer |
| BigBird（局部 + 全局 + 随机） | 近似 O(N) | 混合 | 在 2 倍上下文长度下匹配完全注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 在 0.05 ppl 以内 | DeepSeek-V3.2, 2025 |
| 差分注意力 | O(2·N²) | O(2N) | 困惑度降低 5–10% | DIFF Transformer, 早期 2026 模型 |

## 构建它

参见 `code/main.py`。我们实现了一个因果掩码比较器，在玩具序列上并排展示完全注意力、SWA、局部+跨步稀疏和差分注意力。

### 步骤 1：完全因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角矩阵；对角线以上权重为零。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复完全因果注意力。当 `window = 1` 时，每个词元只关注自身。

### 步骤 3：局部 + 跨步稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口加上每隔 `stride` 步回溯到序列开始的词元。感受野随着层数增加以对数步长增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，使用可学习的混合系数相减。在代码中，我们比较单注意力与差分注意力的注意力下沉热图，观察下沉的消失。

### 步骤 5：KV 缓存大小

在 `N = 131072` 时打印每种变体每层的缓存大小。SWA 和稀疏变体下降了 10–100 倍。差分注意力翻倍。有意识地支付你的内存账单。

## 使用它

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会被编译为自定义 Triton 内核。对于常见模式，速度达到 FlashAttention-3 的 10% 以内，并且掩码函数是一个 Python 可调用对象。

**何时选择每种变体：**

- **纯完全注意力**——上下文长度不超过 16K 的所有层，或当检索质量至关重要时。
- **SWA + 全局混合**——长上下文（>32K），训练和推理受内存限制。2026 年超过 32K 时的默认设置。
- **稀疏块注意力**——自定义内核，自定义模式。保留给专门工作负载（检索、音频）。
- **差分注意力**——任何注意力下沉污染会损害性能的工作负载（长上下文 RAG、大海捞针检索）。

## 交付它

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求以及训练/推理计算概况，为新模型选择注意力拓扑结构。

## 练习

1. **简单。** 运行 `code/main.py`。验证当 `window=4` 时，SWA 将每行最后 4 个词元之外的部分全部置零。验证 `window=n` 时，逐比特完全复现完全因果注意力。
2. **中等。** 在第 07 课旗舰模型的基础上，实现窗口大小为 1024 的因果 SWA。在 tinyshakespeare 上训练 1000 步。验证损失相比完全注意力退化了多少？峰值内存下降了多少？
3. **困难。** 在旗舰模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在相同参数下，与纯 SWA 和纯全局基线对比损失、内存和生成质量。
4. **困难。** 实现每头有可学习 `λ` 的差分注意力。在合成检索任务（一个目标，2000 个干扰项）上训练。与相同参数的单注意力基线比较检索准确率。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| 滑动窗口注意力（SWA） | “局部注意力” | 每个查询关注其最后 `W` 个词元；KV 缓存缩小至 `O(W)`。 |
| 有效感受野 | “模型能看多远” | 在 `L` 层 SWA 堆叠中，窗口为 `W` 时，最多可达 `L × W` 个词元。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 稀疏模式，包含少数始终参与注意力的全局词元；早期长上下文方法。 |
| 原生稀疏注意力 | “DeepSeek 的内核技巧” | 学习块级稀疏性；在内核级别跳过零块，同时保持质量。 |
| 差分注意力 | “两个图，一个相减” | DIFF Transformer：将第一个注意力图减去可学习的 `λ` 乘以第二个注意力图，以消除注意力下沉。 |
| 注意力下沉 | “权重泄漏到词元 0” | Softmax 归一化强制行和为 1；无信息查询将权重倾泻到位置 0。 |
| FlexAttention | “以 Python 函数作为掩码” | PyTorch 2.5+ API，可将任意掩码函数编译为 FlashAttention 形状的内核。 |
| 层类型混合 | “5:1 SWA 与全局层比例” | 在堆叠中交替稀疏层和完全注意力层，以在较低内存下保持质量。 |

## 进一步阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 经典的滑动窗口 + 全局词元论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部 + 跨步模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 SWA 与全局层混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 窗口为 1024 的 5:1 混合，现已成为教科书默认设置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的学习型稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— 使用它中所述掩码可调用模式的 API 参考。
