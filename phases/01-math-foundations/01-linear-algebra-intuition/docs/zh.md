# 线性代数直觉

> 每一个AI模型不过是穿着华丽外衣的矩阵运算。

**类型：** 学习  
**语言：** Python, Julia  
**前置要求：** 阶段0  
**时间：** 约60分钟  

## 学习目标

- 在Python中从头实现向量和矩阵运算（加法、点积、矩阵乘法）
- 从几何角度解释点积、投影和Gram-Schmidt过程的作用
- 使用行约简确定向量组的线性无关性、秩和基
- 将线性代数概念与AI应用联系起来：嵌入、注意力分数和LoRA

## 问题

翻开任何一篇ML论文。在第一页内，你就会看到向量、矩阵、点积和变换。没有线性代数直觉，它们只是符号。有了它，你就能看清神经网络实际上在做什么——在空间中移动点。

你不需要成为数学家。你需要看到这些运算在几何上的含义，然后自己动手编码。

## 概念

### 向量是点（也是方向）

一个向量只是一串数字。但这些数字是有含义的——它们是空间中的坐标。

**2D向量 [3, 2]：**

| x | y | 点 |
|---|---|-------|
| 3 | 2 | 向量从原点(0,0)指向平面上的点(3, 2) |

该向量的模为 sqrt(3^2 + 2^2) = sqrt(13)，方向指向右上方。

在AI中，向量代表一切：
- 一个词 → 一个包含768个数字的向量（它在嵌入空间中的“含义”）
- 一张图像 → 一个包含数百万像素值的向量
- 一个用户 → 一个表示偏好的向量

### 矩阵是变换

矩阵将一个向量变换为另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在AI中，矩阵就是模型：
- 神经网络权重 → 将输入变换为输出的矩阵
- 注意力分数 → 决定关注什么的矩阵
- 嵌入 → 将词语映射为向量的矩阵

### 点积衡量相似性

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

这正是搜索引擎、推荐系统和RAG的工作方式——找到具有高点积的向量。

### 线性无关性

如果集合中没有向量可以表示为其他向量的组合，则这些向量线性无关。如果 v1, v2, v3 无关，它们张成一个3D空间。如果某个向量是其他向量的组合，它们只张成一个平面。

为什么对AI重要：你的特征矩阵应该具有线性无关的列。如果两个特征完全相关（线性相关），模型无法区分它们的效果。这会导致回归中的多重共线性——权重矩阵变得不稳定，微小的输入变化会产生巨大的输出波动。

**具体例子：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1和v2无关——它们彼此不是标量倍数或组合。但 v3 = 2*v1 + v2，所以 {v1, v2, v3} 是相关集。这三个向量都位于xy平面内。无论你如何组合它们，都无法到达 [0, 0, 1]。你有三个向量，但只有两个自由维度。

在数据集中：如果特征3 = 2*特征1 + 特征2，添加特征3不会给模型带来任何新信息。更糟糕的是，它会使正规方程奇异——权重没有唯一解。

### 基与秩

基是张成整个空间的一组最小的线性无关向量。基向量的数量就是空间的维度。

3D空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}。但3D中任意三个无关向量都可以构成一个有效的基。基的选择就是坐标系的选择。

矩阵的秩 = 线性无关的列数 = 线性无关的行数。如果秩 < min(行数, 列数)，则矩阵是秩亏的。这意味着：
- 方程组有无穷多解（或无解）
- 变换中信息丢失
- 矩阵不可逆

| 情况 | 秩 | 对ML的意义 |
|-----------|------|---------------------|
| 满秩（秩 = min(m, n)） | 可能的最大值 | 存在唯一的最小二乘解。模型是良态的。 |
| 秩亏（秩 < min(m, n)） | 低于最大值 | 特征冗余。存在无穷多个权重解。需要正则化。 |
| 秩为1 | 1 | 每一列都是一个向量的缩放副本。所有数据位于一条直线上。 |
| 接近秩亏（奇异值很小） | 数值上低 | 矩阵是病态的。微小的输入噪声会导致输出巨大变化。使用SVD截断或岭回归。 |

### 投影

将向量**a**投影到向量**b**上，得到**a**在**b**方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差 (a - proj_b(a)) 与b垂直。这种正交分解是最小二乘拟合的基础。

投影在机器学习中无处不在：
- 线性回归最小化观测值与列空间的距离——解就是投影
- PCA将数据投影到最大方差方向
- Transformer中的注意力计算查询在键上的投影

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**例子：** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影丢弃了y分量。这是降维的最简单形式——丢弃你不关心的方向。

### Gram-Schmidt过程

将任意一组无关向量转换为标准正交基。标准正交意味着每个向量长度为1，且每对向量互相垂直。

算法步骤：
1. 取第一个向量，将其归一化
2. 取第二个向量，减去它在第一个上的投影，然后归一化
3. 取第三个向量，减去它在所有之前向量上的投影，然后归一化
4. 对剩余向量重复上述步骤

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这就是QR分解的内部工作原理。Q是标准正交基，R记录投影系数。QR分解用于：
- 求解线性方程组（比高斯消元更稳定）
- 计算特征值（QR算法）
- 最小二乘回归（标准数值方法）

## 动手构建

### 第一步：从头实现向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 第二步：从头实现矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 第三步：为什么这对AI重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第四步：Julia版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第五步：从头实现线性无关性和投影（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 使用它

现在用NumPy实现同样的功能——你在实际中会使用的工具：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 使用NumPy的秩、投影和QR分解

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch——张量是带自动微分的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积相对于x的梯度就是y。PyTorch自动计算了这个值。神经网络中的每个运算都由这样的运算组成——矩阵乘法、点积、投影——而自动微分会跟踪所有运算的梯度。

你刚刚从头实现了NumPy一行代码就能完成的事情。现在你知道了底层的工作原理。

## 交付

本课程产出：
- `outputs/prompt-linear-algebra-tutor.md` —— 用于AI助手通过几何直觉教授线性代数的提示词

## 关联

本课程中的所有内容都与现代AI的具体部分相关联：

| 概念 | 出现位置 |
|---------|------------------|
| 点积 | Transformer中的注意力分数，RAG中的余弦相似度 |
| 矩阵乘法 | 每个神经网络层，每个线性变换 |
| 线性无关性 | 特征选择，避免多重共线性 |
| 秩 | 判断系统是否可解，LoRA（低秩适配） |
| 投影 | 线性回归（投影到列空间），PCA |
| Gram-Schmidt / QR | 数值求解器，特征值计算 |
| 标准正交基 | 稳定的数值计算，白化变换 |

LoRA值得特别关注。它通过将权重更新分解为低秩矩阵来微调大型语言模型。LoRA不是更新一个4096x4096的权重矩阵（16M参数），而是更新两个尺寸为4096x16和16x4096的矩阵（131K参数）。秩为16的约束意味着LoRA假设权重更新存在于完整4096维空间中的一个16维子空间中。这就是线性代数在发挥实际作用。

## 练习

1. 实现 `Vector.angle_between(other)` 方法，返回两个向量之间的角度（度数）
2. 创建一个将x坐标加倍、y坐标三倍的2D缩放矩阵，然后将其应用于向量 [1, 1]
3. 给定5个随机词向量（维度50），使用余弦相似度找出最相似的两个
4. 验证Gram-Schmidt的输出是否确实是标准正交的：检查每一对向量点积为0，每个向量模长为1
5. 创建一个秩为2的3x3矩阵。使用 `rank()` 方法验证。然后解释列向量张成的几何对象。
6. 将向量 [1, 2, 3] 投影到 [1, 1, 1] 上。结果在几何上代表什么？

## 关键术语

| 术语 | 人们这样说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | “一个箭头” | 表示n维空间中一个点或方向的一个数字列表 |
| 矩阵 | “一个数字表格” | 将向量从一个空间映射到另一个空间的变换 |
| 点积 | “相乘并求和” | 衡量两个向量对齐程度的量——相似性搜索的核心 |
| 嵌入 | “某种AI魔法” | 表示某物（词、图像、用户）含义的向量 |
| 线性无关性 | “它们不重叠” | 集合中没有向量可以表示为其他向量的组合 |
| 秩 | “有多少维” | 矩阵中线性无关的列（或行）的数量 |
| 投影 | “影子” | 一个向量在另一个方向上的分量 |
| 基 | “坐标轴” | 张成该空间的一组最小的无关向量 |
| 标准正交 | “垂直的单位向量” | 相互垂直且每个模长为1的向量 |
