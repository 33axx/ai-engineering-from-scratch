# 张量操作

> 张量是数据与深度学习之间的通用语言。每一张图像、每一个句子、每一个梯度都通过它们流动。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 1，课程 01（线性代数直觉），02（向量、矩阵与操作）
**时长：** ~90 分钟

## 学习目标

- 从零实现一个具有形状、步幅、重塑、转置和逐元素操作的张量类
- 应用广播规则在不复制数据的情况下对不同形状的张量进行操作
- 编写 einsum 表达式实现点积、矩阵乘法、外积和批量操作
- 精确追踪多头注意力中每一步的张量形状

## 问题

你构建了一个 transformer。前向传播看起来很简洁。运行后却得到：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着形状，尝试转置，现在它说 `Expected 4D input (got 3D input)`。你添加了一个 unsqueeze，然后别的又出错了。

形状错误是深度学习代码中最常见的 bug。它们在概念上并不困难——每个操作都有形状契约——但它们迅速增多。一个 transformer 会串联数十个 reshape、transpose 和 broadcast。一个轴出错就会导致错误级联。更糟糕的是，有些形状错误根本不会抛出错误，而是沿着错误的维度广播或在错误的轴上求和，悄无声息地产生垃圾结果。

矩阵处理两组事物之间的两两关系。但真实数据并不局限于二维。一批 32 张 224x224 的 RGB 图像是一个 4D 张量：`(32, 3, 224, 224)`。具有 12 个头的自注意力也是 4D：`(batch, heads, seq_len, head_dim)`。你需要一种能泛化到任意维度数的数据结构，并且其操作能在所有维度上干净地组合。这种结构就是张量。掌握其操作后，形状错误将变得微不足道且易于调试。

## 概念

### 张量是什么

张量是一个具有统一数据类型的多维数字数组。维度数称为**秩**（或**阶**）。每个维度是一个**轴**。**形状**是一个元组，列出了每个轴的大小。

```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```

总元素数 = 所有大小的乘积。形状 `(2, 3, 4)` 包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

不同的数据类型按惯例映射到特定的张量形状。

```mermaid
graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```

PyTorch 使用 NCHW（通道优先）。TensorFlow 默认使用 NHWC（通道最后）。不匹配的布局会导致静默性能下降或错误。

### 内存布局如何工作

内存中的二维数组是一个一维字节序列。**步幅**告诉你沿每个轴移动一步需要跳过多少个元素。

```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```

转置并不会移动数据。它交换步幅，使张量变为**非连续**——行的元素在内存中不再相邻。

### 广播规则

广播允许你在不复制数据的情况下对不同形状的张量进行操作。从右侧对齐形状。当两个维度相等或其中一个为 1 时，它们是兼容的。维度较少的会在左侧填充 1。

```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```

### Einsum：通用张量操作

爱因斯坦求和约定用字母标记每个轴。出现在输入但不出现在输出中的轴会被求和。同时出现的轴则保留。

```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```

关键模式：`i,i->`（点积），`i,j->ij`（外积），`ii->`（迹），`ij->ji`（转置），`bij,bjk->bik`（批量矩阵乘法），`bhtd,bhsd->bhts`（注意力分数）。

## 动手构建

代码位于 `code/tensors.py`。每一步都会引用该实现。

### 步骤 1：张量存储与步幅

张量存储一个扁平的数值列表以及形状元数据。步幅告诉索引逻辑如何将多维索引映射到扁平位置。

```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```

对于形状 `(3, 4)`，步幅为 `(4, 1)`——前进一行跳过 4 个元素，前进一列跳过 1 个元素。

### 步骤 2：Reshape、Squeeze、Unsqueeze

Reshape 改变形状而不改变元素顺序。元素总数必须保持不变。对某个维度使用 `-1` 可以自动推断其大小。

```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```

Squeeze 移除大小为 1 的轴。Unsqueeze 插入一个大小为 1 的轴。Unsqueeze 对于广播至关重要——一个偏置向量 `(D,)` 要添加到批处理 `(B, T, D)` 中，需要 unsqueeze 为 `(1, 1, D)`。

```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```

### 步骤 3：Transpose 与 Permute

Transpose 交换两个轴。Permute 重新排列所有轴。这就是在 NCHW 和 NHWC 之间转换的方法。

```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```

经过 transpose 或 permute 后，张量在内存中变为非连续。在 PyTorch 中，`view` 对非连续张量会失败——应使用 `reshape` 或先调用 `.contiguous()`。

### 步骤 4：逐元素操作与规约

逐元素操作（加法、乘法、减法）对每个元素独立应用并保持形状。规约操作（求和、均值、最大值）会折叠一个或多个轴。

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```

CNN 中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 产生 `(B, C)`。NLP 中的序列平均池化：`(B, T, D).mean(axis=1)` 产生 `(B, D)`。

### 步骤 5：使用 NumPy 进行广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。

```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```

通过广播计算成对距离：将 `(M, 2)` 重塑为 `(M, 1, 2)`，`(N, 2)` 重塑为 `(1, N, 2)`，相减，平方，沿最后一个轴求和，再开方。结果形状：`(M, N)`。

### 步骤 6：Einsum 操作

`demo_einsum()` 和 `demo_einsum_gallery()` 函数逐一展示了所有常见模式。

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

缩并的计算代价是所有索引大小（保留和求和）的乘积。对于 `bij,bjk->bik`，假设 B=32，I=128，J=64，K=128：`32 * 128 * 64 * 128 = 33,554,432` 次乘加操作。

### 步骤 7：通过 Einsum 实现注意力机制

`demo_attention_einsum()` 函数端到端地实现了多头注意力。

```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```

每一步都是张量操作：投影（通过 einsum 的矩阵乘法）、头分割（reshape + transpose）、注意力分数（通过 einsum 的批量矩阵乘法）、加权求和（通过 einsum 的批量矩阵乘法）、头合并（transpose + reshape）、输出投影（通过 einsum 的矩阵乘法）。

## 使用它

### 手写实现 vs NumPy

| 操作 | 手写 (Tensor 类) | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| Reshape | `t.reshape((3,4))` | `a.reshape(3,4)` |
| Transpose | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| Squeeze | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | 不支持 | `np.einsum("ij,jk->ik", a, b)` |

### 手写实现 vs PyTorch

```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```

PyTorch 增加了自动微分、GPU 支持和优化的 BLAS 内核。形状语义完全相同。如果你理解了手写版本，PyTorch 的形状错误就会变得可读。

### 每个神经网络层作为张量操作

| 操作 | 张量形式 | Einsum |
|---|---|---|
| 线性层 | `Y = X @ W.T + b` | `"bd,od->bo"` + 偏置 |
| 注意力 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力分数 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批归一化 | `(X - mu) / sigma * gamma` | 逐元素 + 广播 |
| Softmax | `exp(x) / sum(exp(x))` | 逐元素 + 规约 |

## 输出产物

本课程产生两个可复用的提示：

1. **`outputs/prompt-tensor-shapes.md`** —— 一个用于调试张量形状不匹配的系统性提示。包含每个常见操作（matmul、broadcast、cat、Linear、Conv2d、BatchNorm、softmax）的决策表以及修复查找表。

2. **`outputs/prompt-tensor-debugger.md`** —— 一个逐步调试提示，当形状错误阻碍你时，将其粘贴到任何 AI 助手中。输入错误消息和张量形状，即可获得精确修复。

## 练习

1. **简单 —— Reshape 往返。** 取一个形状为 `(2, 3, 4)` 的张量。将其重塑为 `(6, 4)`，然后为 `(24,)`，再回到 `(2, 3, 4)`。通过打印扁平数据验证每一步元素顺序保持不变。

2. **中等 —— 实现广播。** 为 `Tensor` 类扩展一个 `broadcast_to(shape)` 方法，将大小为 1 的维度扩展到目标形状。然后修改 `_elementwise_op` 使其在操作前自动广播。用形状 `(3, 1)` 和 `(1, 4)` 测试，预期结果 `(3, 4)`。

3. **困难 —— 从零构建 Einsum。** 实现一个基本的 `einsum(subscripts, *tensors)` 函数，至少处理：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）和转置（`ij->ji`）。解析下标字符串，识别缩并的索引，并遍历所有索引组合。将结果与 `np.einsum` 对比。

4. **困难 —— 注意力形状追踪器。** 编写一个函数，接收 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads` 作为输入，输出多头注意力每一步的精确形状：输入、Q/K/V 投影、头分割、注意力分数、softmax 权重、加权求和、头合并、输出投影。并与 `demo_attention_einsum()` 的输出进行验证。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| Tensor | “就是矩阵但更多维度” | 具有统一类型、定义好的形状、步幅和操作的多维数组 |
| Rank | “维度数目” | 轴的数量。矩阵的秩为 2，并非矩阵论中的秩 |
| Shape | “张量的大小” | 列出每个轴大小的元组。`(2, 3)` 表示 2 行 3 列 |
| Stride | “内存如何布局” | 沿每个轴前进一步需要跳过的元素数 |
| Broadcasting | “形状不同时它自动就能工作” | 一组严格的规则：从右对齐，维度必须相等或其中一个为 1 |
| Contiguous | “张量是正常的” | 元素在内存中连续存储，没有空隙或逻辑布局的重排 |
| Einsum | “一种花哨的写矩阵乘法的方式” | 一种通用记号，一行表达式即可表示任何张量缩并、外积、迹或转置 |
| View | “和 reshape 一样” | 共享同一内存缓冲区但具有不同形状/步幅元数据的张量。在非连续数据上会失败 |
| Contraction | “对某个索引求和” | 张量之间共享索引被相乘并求和的一般操作，产生低秩结果 |
| NCHW / NHWC | “PyTorch vs TensorFlow 格式” | 图像张量的内存布局约定。NCHW 将通道放在空间维度之前，NHWC 将通道放在之后 |

## 扩展阅读

- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html) —— 经典规则及可视化示例
- [PyTorch 张量视图](https://pytorch.org/docs/stable/tensor_view.html) —— 视图何时工作、何时复制
- [einops](https://github.com/arogozhnikov/einops) —— 使张量重塑可读且安全的库
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) —— 可视化注意力中张量形状的流动
- [NumPy 中的爱因斯坦求和](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) —— 完整的 einsum 文档及示例
