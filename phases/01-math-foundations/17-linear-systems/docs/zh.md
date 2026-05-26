# 线性方程组

> 解 Ax = b 是数学中最古老的问题，却至今仍在驱动你的神经网络。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，第01课（线性代数直觉），第02课（向量与矩阵），第03课（矩阵变换）
**时间：** 约120分钟

## 学习目标

- 使用带部分主元的高斯消去法和回代求解 Ax = b
- 用 LU、QR 和 Cholesky 分解分解矩阵，并解释每种方法适用的场景
- 推导最小二乘法的正规方程，并将其与线性回归和岭回归联系起来
- 使用条件数诊断病态系统，并利用正则化使其稳定

## 问题

每次训练线性回归，你都会解一个线性方程组。每次计算最小二乘拟合，你都会解一个线性方程组。每次神经网络层计算 `y = Wx + b`，它都在评估一个线性方程组的一侧。当你加入正则化时，你修改了这个方程组。当你使用高斯过程时，你分解了一个矩阵。当你为马氏距离求协方差矩阵的逆时，你解了一个线性方程组。

方程 Ax = b 无处不在。A 是已知系数的矩阵。b 是已知输出的向量。x 是你想找的未知数向量。在线性回归中，A 是你的数据矩阵，b 是你的目标向量，x 是权重向量。整个模型归结为：找到 x 使得 Ax 尽可能接近 b。

本课程从零开始构建求解该方程的每一种主要方法。你将理解为什么有些方法快而另一些稳定，为什么有些只适用于方阵系统而另一些能处理超定系统，以及为什么矩阵的条件数决定了你的答案是否有任何意义。

## 概念

### Ax = b 的几何含义

线性方程组有几何解释。每个方程定义一个超平面。解是所有超平面相交的点（或点集）。

```
2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["Solution: (2, 1)"]
    B["x - y = 1"] --- S
```

可能发生三种情况：

```mermaid
graph TD
    subgraph "One Solution"
        A1["Lines intersect at a single point"]
    end
    subgraph "No Solution"
        A2["Lines are parallel — no intersection"]
    end
    subgraph "Infinite Solutions"
        A3["Lines are identical — every point is a solution"]
    end
```

用矩阵形式表示，“一个解”意味着 A 可逆。“无解”意味着方程组矛盾。“无穷多解”意味着 A 有零空间。大多数机器学习问题属于“无精确解”的类别，因为你的方程（数据点）比未知数（参数）多。这就是最小二乘法发挥作用的地方。

### 列视角与行视角

有两种方式理解 Ax = b。

**行视角**。A 的每一行定义一个方程。每个方程是一个超平面。解是所有超平面相交的地方。

**列视角**。A 的每一列是一个向量。问题变成：A 的列如何线性组合才能产生 b？

```
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.
```

列视角更为基础。如果 b 位于 A 的列空间中，方程组有解。如果 b 不在，你找到列空间中距离最近的点。这个最近的点就是最小二乘解。

### 高斯消去法

高斯消去法将 Ax = b 转化为上三角系统 Ux = c，然后通过回代求解。这是最直接的方法。

算法：

```
1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.
```

示例：

```
Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```

高斯消去法需要 O(n³) 次运算。对于一个 1000×1000 的系统，大约需要十亿次浮点运算。很快，但如果你需要解多个具有相同 A 的系统，你可以做得更好。

### 部分主元：为什么重要

没有主元，高斯消去法可能失败或产生垃圾结果。如果主元元素为零，你会除以零。如果主元很小，你会放大舍入误差。

```
Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.
```

在有限精度的浮点运算中，没有主元的版本可能丢失有效数字。部分主元总是选择当前列中绝对值最大的元素作为主元，以最小化误差放大。

### LU 分解

LU 分解将 A 分解为下三角矩阵 L 和上三角矩阵 U：A = LU。L 矩阵存储高斯消去法中的乘数。U 矩阵是消去的结果。

```
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```

为什么分解而不是直接消去？因为一旦你有了 L 和 U，对于任意新的 b 求解 Ax = b 只需 O(n²)：

```
Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))
```

O(n³) 的代价在分解时只支付一次。之后的每次求解都是 O(n²)。如果你需要为同一个 A 但不同的 b 向量求解 1000 个系统，LU 将总工作量节省约 1000/3 倍。

使用部分主元时，你得到 PA = LU，其中 P 是记录行交换的置换矩阵。

### QR 分解

QR 分解将 A 分解为正交矩阵 Q 和上三角矩阵 R：A = QR。

正交矩阵具有性质 QᵀQ = I。其列是标准正交向量。乘以 Q 保持长度和角度不变。

```
A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.
```

在求解最小二乘问题时，QR 在数值上比 LU 更稳定。Gram-Schmidt 过程逐列构建 Q：

```
Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j
```

每一步去除沿着所有先前 q 向量方向的分量，只留下新的正交方向。

### Cholesky 分解

当 A 对称（A = Aᵀ）且正定（所有特征值为正）时，你可以将其分解为 A = LLᵀ，其中 L 是下三角矩阵。这就是 Cholesky 分解。

```
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j
```

Cholesky 比 LU 快两倍，且只需一半存储空间。它只适用于对称正定矩阵，但这些矩阵经常出现：

- 协方差矩阵是对称半正定的（加上正则化后正定）。
- 高斯过程中的核矩阵是对称正定的。
- 凸函数在最小值处的 Hessian 矩阵是对称正定的。
- AᵀA 始终是对称半正定的。

在高斯过程中，你用 Cholesky 分解核矩阵 K，然后解 Kα = y 得到预测均值。Cholesky 因子还给出边际似然的 log 行列式：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘法：当 Ax = b 没有精确解时

如果 A 是 m×n 且 m > n（方程多于未知数），则系统是超定的。没有精确解。相反，你最小化平方误差：

```
minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```

最小值满足正规方程：

```
A^T A x = A^T b
```

推导：展开 ||Ax - b||² = (Ax - b)ᵀ(Ax - b) = xᵀAᵀAx - 2xᵀAᵀb + bᵀb。对 x 求梯度，设为零：2AᵀAx - 2Aᵀb = 0。

```
Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.
```

### 正规方程 = 线性回归

这个联系是直接的。在线性回归中，你的数据矩阵 X 每行是一个样本，每列是一个特征。你的目标向量 y 每个样本有一个条目。权重向量 w 满足：

```
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```

这是线性回归的闭式解。每次调用 `sklearn.linear_model.LinearRegression.fit()` 都会计算这个（或通过 QR 或 SVD 等效方法）。

在矩阵中加入正则化项 λI，你就得到了岭回归：

```
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```

正则化使矩阵的条件更好（更容易精确求逆），并通过将权重向零收缩来防止过拟合。当 λ > 0 时，矩阵 XᵀX + λI 总是对称正定的，因此你可以使用 Cholesky 求解。

### 伪逆（Moore-Penrose）

伪逆 A⁺ 将矩阵求逆推广到非方阵和奇异矩阵。对于任意矩阵 A：

```
x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)
```

Σ⁺ 是通过取每个非零奇异值的倒数并转置结果形成的。如果 A = UΣVᵀ，则 A⁺ = VΣ⁺Uᵀ。

```
A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```

伪逆给出最小范数最小二乘解。如果系统：
- 有唯一解：A⁺b 给出它。
- 无解：A⁺b 给出最小二乘解。
- 有无穷多解：A⁺b 给出 ||x|| 最小的解。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 内部都使用 SVD。

### 条件数

条件数衡量解对输入微小变化的敏感程度。对于矩阵 A，条件数为：

```
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```

其中 σ_max 和 σ_min 是最大和最小奇异值。

```
Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage
```

经验法则：
- κ < 100：安全，解准确。
- κ ≈ 10ᵏ：你会损失大约 k 位浮点运算精度。
- κ ≈ 10¹⁶（对于 float64）：解毫无意义。矩阵实际上是奇异的。

在机器学习中，病态发生在特征几乎共线时。正则化（加上 λI）将条件数从 σ_max / σ_min 改善为 (σ_max + λ) / (σ_min + λ)。

### 迭代方法：共轭梯度法

对于非常大的稀疏系统（数百万个未知数），像 LU 或 Cholesky 这样的直接方法代价过高。迭代方法通过多次迭代改进猜测来近似解。

共轭梯度法（CG）在 A 对称正定时求解 Ax = b。在精确算术中，它最多在 n 次迭代内找到精确解，但如果 A 的特征值聚集，它通常收敛得快得多。

```
Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop
```

CG 用于：
- 大规模优化（Newton-CG 方法）
- 求解 PDE 离散化
- 核方法，当核矩阵太大无法分解时
- 作为其他迭代求解器的预条件器

收敛速度取决于条件数。条件数更好的系统收敛更快，这也是正则化有帮助的另一个原因。

### 全貌：何时使用哪种方法

| 方法 | 要求 | 代价 | 使用场景 |
|------|-------|------|----------|
| 高斯消去法 | 方阵、非奇异 A | O(n³) | 一次性求解方阵系统 |
| LU 分解 | 方阵、非奇异 A | O(n³) 分解 + O(n²) 求解 | 对同一 A 多次求解 |
| QR 分解 | 任意 A (m ≥ n) | O(mn²) | 最小二乘，数值稳定 |
| Cholesky | 对称正定 A | O(n³/3) | 协方差矩阵、高斯过程、岭回归 |
| 正规方程 | 超定 (m > n) | O(mn² + n³) | 线性回归（特征数 n 较小） |
| SVD / 伪逆 | 任意 A | O(mn²) | 秩亏系统、最小范数解 |
| 共轭梯度法 | 对称正定、稀疏 A | O(n * k * nnz) | 大型稀疏系统，k 为迭代次数 |

### 与机器学习的联系

本课程中的每种方法都出现在生产级机器学习中：

**线性回归。** 闭式解求解正规方程 XᵀX w = Xᵀy。这通过 Cholesky（如果 n 小）、QR（如果数值稳定性重要）或 SVD（如果矩阵可能秩亏）完成。

**岭回归。** 在 XᵀX 上加 λI。正则化系统 (XᵀX + λI) w = Xᵀy 总是可以通过 Cholesky 求解，因为当 λ > 0 时 XᵀX + λI 对称正定。

**高斯过程。** 预测均值需要求解 Kα = y，其中 K 是核矩阵。对 K 进行 Cholesky 分解是标准方法。对数边际似然使用 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化使用 QR 分解创建列标准正交的权重矩阵。这可以防止深度网络中的信号塌缩。

**预条件。** 大规模优化器使用不完全 Cholesky 或不完全 LU 作为共轭梯度求解器的预条件器。

**特征工程。** XᵀX 的条件数告诉你特征是否共线。如果 κ 很大，则丢弃特征或添加正则化。

## 动手构建

### 步骤 1：带部分主元的高斯消去法

```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x
```

### 步骤 2：LU 分解

```python
def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x
```

### 步骤 3：Cholesky 分解

```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```

### 步骤 4：通过正规方程的最小二乘

```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x
```

### 步骤 5：条件数

```python
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```

## 实际应用

将各部分组合起来，对真实数据进行线性回归和岭回归：

```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")
```

## 交付成果

本课程产出：
- `code/linear_systems.py`，包含从零实现的高斯消去法、LU 分解、Cholesky 分解、最小二乘法和岭回归
- 一个工作演示，展示正规方程与 sklearn 的 LinearRegression 产生相同的权重

## 练习

1. 使用你的高斯消去法、你的 LU 求解器和 `np.linalg.solve` 解系统 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证三种方法在浮点容差内给出相同答案。

2. 生成一个 50×5 的随机矩阵 X 和目标 y = X @ w_true + 噪声。使用正规方程、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 求解 w。比较所有四个解。测量 XᵀX 的条件数，并解释它如何影响你信任哪种方法。

3. 通过使两列几乎相同（例如，列 2 = 列 1 + 1e-10 * 噪声）创建一个几乎奇异的矩阵。计算其条件数。在不加正则化和加正则化（加 0.01 * I）的情况下解 Ax = b。比较解和残差。解释为什么正则化有帮助。

4. 实现共轭梯度算法，用于一个 100×100 的随机对称正定矩阵。统计收敛到容差 1e-8 需要多少次迭代。与理论最大值 n 次迭代进行比较。

5. 测试你的 Cholesky 求解器与你的 LU 求解器以及 `np.linalg.solve` 在大小为 10、50、200、500 的对称正定矩阵上的运行时间。绘制结果。验证 Cholesky 大约比 LU 快 2 倍。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Linear system | "解出 x" | 一组线性方程 Ax = b。求解 x 意味着找到在变换 A 下产生输出 b 的输入。 |
| Gaussian elimination | "行化简" | 通过行操作系统地消去对角线以下的元素，得到上三角系统，再通过回代求解。O(n³)。 |
| Partial pivoting | "交换行以获得稳定性" | 在消去第 k 列之前，将该列绝对值最大的行交换到主元位置。防止除以小数字。 |
| LU decomposition | "分解成三角矩阵" | 将 A 写为 A = LU，其中 L 是下三角（存储乘数），U 是上三角（消去后的矩阵）。将 O(n³) 代价分摊到多次求解中。 |
| QR decomposition | "正交分解" | 将 A 写为 A = QR，其中 Q 的列标准正交，R 是上三角。在最小二乘中比 LU 更稳定。 |
| Cholesky decomposition | "矩阵的平方根" | 对于对称正定 A，写为 A = LLᵀ。代价是 LU 的一半。用于协方差矩阵、核矩阵和岭回归。 |
| Least squares | "无法精确时取最佳拟合" | 当系统超定（方程多于未知数）时，最小化残差平方和 ||Ax - b||²。 |
| Normal equations | "微积分的捷径" | AᵀAx = Aᵀb。将 ||Ax - b||² 的梯度设为零得到。这就是线性回归的闭式解。 |
| Pseudoinverse | "非方阵的求逆" | 通过 SVD 得到 A⁺ = VΣ⁺Uᵀ。对于任意矩阵（方阵或矩形，奇异或非奇异）给出最小范数最小二乘解。 |
| Condition number | "这个答案有多可信" | κ = σ_max / σ_min。衡量对输入扰动的敏感性。大约损失 log10(κ) 位精度。 |
| Ridge regression | "正则化的最小二乘" | 解 (XᵀX + λI) w = Xᵀy。添加 λI 改善条件性并将权重向零收缩。防止过拟合。 |
| Conjugate gradient | "针对大型矩阵的迭代 Ax=b 求解" | 用于对称正定系统的迭代求解器。最多在 n 步内收敛。适用于分解代价过高的大型稀疏系统。 |
| Overdetermined system | "数据多于参数" | m×n 系统中 m > n。无精确解。最小二乘法找到最佳近似。这是每个回归问题。 |
| Back substitution | "从下往上求解" | 给定上三角系统，先解最后一个方程，然后向后代入。O(n²)。 |
| Forward substitution | "从上往下求解" | 给定下三角系统，先解第一个方程，然后向前代入。O(n²)。用于 LU 求解中的 L 步骤。 |

## 延伸阅读

- [MIT 18.06: Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang) —— 关于线性系统和矩阵分解的权威课程
- [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html) (Trefethen & Bau) —— 理解数值稳定性、条件性以及算法为何失败的经典参考书
- [Matrix Computations](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm) (Golub & Van Loan) —— 关于每种矩阵算法的百科全书式参考书
- [3Blue1Brown: Inverse Matrices](https://www.3blue1brown.com/lessons/inverse-matrices) —— 对求解 Ax = b 几何含义的直观理解
