# 从零实现自注意力机制

> 注意力机制就像一张查找表，每个词都会问：“谁对我重要？”——然后学习答案。

**类型：** 动手构建  
**语言：** Python  
**前置知识：** 阶段3（深度学习核心）、阶段5第10课（序列到序列）  
**预计时长：** ~90分钟  

## 学习目标

- 仅使用 NumPy 从零实现缩放点积自注意力机制，包括查询/键/值投影及 softmax 加权求和
- 构建一个多头注意力层，能够分割头、并行计算注意力并拼接结果
- 追踪注意力矩阵如何捕获词元之间的关系，并解释为何使用 sqrt(d_k) 缩放能防止 softmax 饱和
- 应用因果掩码，将双向注意力转换为自回归（解码器风格）注意力

## 问题

RNN 一次处理一个词元。当到达第 50 个词元时，第 1 个词元的信息已经经历了 50 次压缩传递。长距离依赖被挤压进一个固定大小的隐藏状态——这个瓶颈即使是 LSTM 的门控也无法完全解决。

2014 年 Bahdanau 的注意力论文展示了修复方法：让解码器回顾编码器的每个位置，并决定哪些位置对当前步骤重要。但它仍然依附在 RNN 之上。2017 年的《Attention Is All You Need》提出了一个更尖锐的问题：如果注意力是*唯一*的机制呢？没有循环，没有卷积，只有注意力。

自注意力让序列中的每个位置都能在一步并行操作中关注到所有其他位置。这正是 Transformer 快速、可扩展且占据主导地位的原因。

## 概念

### 数据库查找的类比

将注意力想象成一种软性的数据库查找：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个词元生成三个向量：
- **查询 (Query, Q)**：“我在找什么？”
- **键 (Key, K)**：“我包含什么？”
- **值 (Value, V)**：“如果被选中，我提供什么信息？”

查询与所有键的点积产生注意力分数。高分意味着“这个键匹配我的查询”。这些分数对值进行加权。输出是值的加权和。

### Q、K、V 的计算

每个词元嵌入通过三个学习得到的权重矩阵进行投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

对于一个词元，可视化展示如下：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### 注意力矩阵

一旦你获得了所有词元的 Q、K、V，注意力分数就形成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### 为什么要缩放？

点积随维度 dk 增长。如果 dk = 64，点积可能落在数十的范围内，将 softmax 推入梯度消失的区域。解决方法：除以 sqrt(dk)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这将值保持在一个能使 softmax 产生有用梯度的范围内。

### Softmax 将分数转换为权重

Softmax 将原始分数转换为每一行上的概率分布：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

现在每个词元都有一组权重，表示它关注其他每个词元的程度。

### 值的加权和

每个词元的最终输出是所有值向量的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手构建

### 步骤 1：从零实现 Softmax

Softmax 将原始 logits 转换为概率。减去最大值以获得数值稳定性。

```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    stable_x = x - x_max
    exp_x = np.exp(stable_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 步骤 2：缩放点积注意力

核心函数。接受 Q、K、V 矩阵，返回注意力输出及权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 步骤 3：带学习投影的自注意力类

一个完整的自注意力模块，包含使用 Xavier 类缩放初始化的 Wq、Wk、Wv 权重矩阵。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 步骤 4：在句子上运行

为句子创建假嵌入，并观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 步骤 5：用 ASCII 热力图可视化注意力

将注意力权重映射为字符，以便快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用它

PyTorch 的 `nn.MultiheadAttention` 正好实现了我们构建的功能，外加多头分割和输出投影：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
attn_output, attn_weights = mha(query, key, value)
```

关键区别：多头注意力并行运行多个注意力函数，每个函数有自己大小为 dk = d_model / n_heads 的 Q、K、V 投影，然后拼接结果。这使模型能够同时关注不同关系类型。

## 交付物

本课程产生：
- `outputs/prompt-attention-explainer.md` —— 一个用于通过数据库查找类比解释注意力的提示词

## 练习

1. 修改 `scaled_dot_product_attention`，使其接受一个可选的掩码矩阵，在 softmax 之前将某些位置设置为负无穷（这就是因果/解码器掩码的工作方式）
2. 从零实现多头注意力：将 Q、K、V 分成 `n_heads` 个块，分别运行注意力，拼接后通过最终权重矩阵 Wo 进行投影
3. 取两个长度相同的不同句子，通过同一个 SelfAttention 实例，比较它们的注意力模式。哪些变了？哪些保持不变？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Query (Q) | “问题向量” | 输入的一种学习投影，表示该词元在寻找什么信息 |
| Key (K) | “标签向量” | 一种学习投影，表示该词元包含什么信息，与查询进行匹配 |
| Value (V) | “内容向量” | 一种学习投影，携带实际信息，根据注意力分数进行聚合 |
| Scaled dot-product attention | “注意力公式” | softmax(QK^T / sqrt(dk)) @ V —— 缩放防止了高维度下的 softmax 饱和 |
| Self-attention | “词元看自己和他人” | Q、K、V 都来自同一序列的注意力，让每个位置可以关注所有其他位置 |
| Attention weights | “关注的程度” | 基于缩放点积经 softmax 产生的概率分布，作用于各个位置 |
| Multi-head attention | “并行注意力” | 运行多个注意力函数（使用不同的投影），然后拼接结果以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) —— 原始 Transformer 论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) —— 完整架构的最佳可视化讲解
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 带有逐行解释的 PyTorch 实现
