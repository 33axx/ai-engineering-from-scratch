# 数值稳定性

> 浮点数是一个泄漏的抽象。它会在训练过程中给你带来麻烦，而你却无法预见到它的到来。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第一阶段，课程 01-04  
**时间：** 约 120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点运算中的上溢、下溢和灾难性抵消
- 使用中心有限差分法将解析梯度与数值梯度进行对比验证
- 解释为什么 bfloat16 在训练中优于 float16，以及损失缩放如何防止梯度下溢

## 问题所在

你的模型训练了三个小时，然后损失变成 NaN。你添加了一条打印语句。在第 9,000 步时，logits 还是正常的。到第 9,001 步，它们变成了 `inf`。到第 9,002 步，每个梯度都变成了 `nan`，训练死掉了。

或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了所有东西。架构匹配。超参数匹配。数据匹配。问题是论文用了 float32，而你用了 float16 却没有进行正确的缩放。三十二位累积的舍入误差悄无声息地吞噬了你的准确率。

或者：你从头实现了交叉熵损失。它在小 logits 上没问题。当 logits 超过 100 时，它返回 `inf`。Softmax 溢出是因为 `exp(100)` 大于 float32 所能表示的范围。每个机器学习框架都用一个两行的小技巧处理这个问题。而你并不知道这个技巧的存在。

数值稳定性不是一个理论问题。它区分了一次成功的训练运行和一次悄无声息的失败运行。你将来会调试的每一个严重的机器学习 bug，最终都会归结到浮点数上。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数有三部分：符号位、指数和尾数（有效数字）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少位有效数字）。指数决定范围（数字可以多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 提供大约 7 位十进制数字的精度。这意味着它可以区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。7 位之后，一切就都是舍入噪声了。

float16 提供大约 3 位。它能表示的最大数字是 65,504。这对于机器学习来说小得令人不安，因为 logits、梯度和激活值常常超过这个值。

bfloat16 是 Google 为解决 float16 的范围问题而提出的方案。它具有与 float32 相同的 8 位指数（相同范围，最大到 3.4e38），但只有 7 位尾数位（精度低于 float16）。对于训练神经网络，范围比精度更重要，因此 bfloat16 通常是赢家。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点数中精确表示。在基数为 2 的系统中，它是一个循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对机器学习很重要，因为：

1. 类似 `if loss < threshold` 的损失比较可能给出错误答案
2. 累加许多小值（数千步的梯度更新）会偏离真实和
3. 如果你用 `==` 比较浮点数，校验和和可重现性测试会失败

解决方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字会抵消，你得到的是被提升为首位的舍入噪声。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

这来自一次简单的减法，就有 19% 的相对误差。在机器学习中，这发生在当你：

- 计算具有大均值的数据的方差：当 E[x] 很大时，`E[x^2] - E[x]^2`
- 减去几乎相等的对数概率
- 使用太小的 epsilon 计算有限差分梯度

解决方法：重新排列公式，避免减去大的、几乎相等的数字。对于方差，使用 Welford 算法或先对数据中心化。对于对数概率，全程在对数空间中运算。

### 上溢和下溢

上溢发生在结果太大以至于无法表示时。下溢发生在结果太小（比最小的可表示正数更接近零）时。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

在机器学习中，`exp()` 函数是上溢的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则走向另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在机器学习中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。没有适当技巧的情况下，`log(exp(x))` 的组合是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任意 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会下溢为 0，然后 `log(0)` 是 `-inf`。

技巧：取指数之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么这样有效：减去 `max(x)` 之后，最大的指数项是 `exp(0) = 1`。不可能溢出。和中至少有一项是 1，所以和至少是 1，`log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，溢出被消除了。

这个技巧在机器学习中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有这个技巧，logits [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率相同，计算安全。这不是一个优化，它是正确性的要求。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中的一个 `nan` 会使权重变成 `nan`，进而使每个后续输出都变成 `nan`。训练在一个步骤之内就死掉了。

`inf` 如何出现：
- `exp()` 对一个大的正数求值
- 除以零：`1.0 / 0.0`
- `float32` 在累加中溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` 对一个负数求值
- `log()` 对一个负数求值
- 任何涉及已有 `nan` 的算术运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 对 `exp()` 的输入进行裁剪：`exp(clamp(x, -80, 80))`
2. 在分母中添加微小量：`x / (y + 1e-8)`
3. 在 `log()` 内部添加微小量：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp，稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试时在每个前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能存在 bug。数值梯度检查通过使用有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这个公式是 O(h^2) 精度的，比前向差分 `(f(x+h) - f(x)) / h`（仅为 O(h)）好得多。

选择 h：太大则近似不准确。太小则灾难性抵消会破坏答案。通常取 `h = 1e-5` 到 `1e-7`。

检查：计算解析梯度与数值梯度的相对差值。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时，务必检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 用于此目的。

### 混合精度训练

现代 GPU 拥有专门的硬件（张量核心），可以比 float32 快 2-8 倍地计算 float16 矩阵乘法。混合精度训练利用这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 会下溢任何低于约 6e-8 的值到零。你的模型停止学习，因为所有梯度更新都是零。

解决方法是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放自动调整缩放因子。从一个较大的值开始（65536）。如果梯度溢出为 `inf`，则将其减半。如果经过 N 步没有溢出，则加倍。

### bfloat16 与 float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 有更高的精度（10 位尾数位 vs 7），但范围有限（最大约 65,504）。bfloat16 精度较低，但范围与 float32 相同（最大约 3.4e38）。

对于训练神经网络：

- 激活值和 logits 在训练尖峰期间经常会超过 65,504。float16 会溢出；bfloat16 能处理。
- float16 需要损失缩放，而 bfloat16 通常不需要，因为它的范围覆盖了梯度幅度谱。
- bfloat16 是 float32 的简单截断：丢掉尾数的低 16 位。转换简单且指数无损。

float16 更适合推理，因为值有界且精度更重要。bfloat16 更适合训练，因为范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）内置了 bfloat16 支持。

### 梯度裁剪

梯度爆炸发生在梯度在许多层中指数级增长时（常见于 RNN、深度网络和 Transformer）。一个大的梯度可以在一步之内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立裁剪每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的工作。这是标准选择。

典型值：Transformer 使用 `max_norm=1.0`，RL 使用 `max_norm=0.5`，简单网络使用 `max_norm=5.0`。

梯度裁剪不是一个 hack，它是一个安全机制。没有它，单个离群批次可能产生大到足以毁掉数周训练的梯度。

### 归一化层作为数值稳定器

批归一化、层归一化和 RMS 归一化通常被描述为帮助训练收敛的正则化器。它们也是数值稳定器。

没有归一化，激活值可能通过各层指数级增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每一层重新居中并重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）防止当所有激活值都相同时除以零。可学习参数 `gamma` 和 `beta` 让网络在需要时恢复任何尺度。

这使值在整个网络中保持在数值安全的范围内，既防止了前向传播中的溢出，也防止了反向传播中的梯度爆炸。

### 常见的机器学习数值 Bug

**Bug：** 几个 epoch 后损失变成 NaN。
**原因：** logits 变得太大，softmax 溢出。或者学习率太高导致权重发散。
**修复：** 使用稳定 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：** 损失卡在 log(num_classes)。
**原因：** 模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
**修复：** 检查数据标签是否正确，验证损失函数，检查是否有死 ReLU。

**Bug：** 验证准确率低于预期 1-3%。
**原因：** 混合精度没有适当的损失缩放。梯度下溢悄无声息地将小更新置零。
**修复：** 启用动态损失缩放，或切换为 bfloat16。

**Bug：** 某些层的梯度范数为 0.0。
**原因：** 死 ReLU 神经元（所有输入为负），或 float16 下溢。
**修复：** 使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：** 模型在一个 GPU 上工作正常，但在另一个 GPU 上结果不同。
**原因：** 浮点累加顺序不确定。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不可结合。
**修复：** 接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug：** 损失计算中 `exp()` 返回 `inf`。
**原因：** 原始 logits 未使用最大值减法技巧就传入了 `exp()`。
**修复：** 使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：** 从 float32 切换到 float16 后训练发散。
**原因：** float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的激活值。
**修复：** 使用带有损失缩放的混合精度（AMP），或改用 bfloat16。

## 构建

### 步骤 1：展示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 softmax 与稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤 3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤 4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

请参见 `code/numerical.py` 了解包含所有边界情况的完整实现。

## 交付

本课程产出：
- `code/numerical.py`：包含稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在第三阶段构建训练循环时和第四阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用 float32 的朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。将误差与真实方差（0.6667）进行比较。

2. **精度探索。** 在 Python 中找到最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是机器 epsilon。验证它等于 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 测试你的 `logsumexp_stable` 函数，使用以下情况：（a）所有值相等，（b）一个值远大于其他值，（c）所有值都非常负（-1000）。验证它在朴素版本失败时给出正确结果。

4. **梯度检查一个神经网络层。** 实现一个简单的线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证一个 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围 [1e-9, 1e-3] 的随机梯度，转换为 float16，并测量多少比例变为零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回来，再次测量零比例。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。每个现代 CPU 和 GPU 都实现它。 |
| 机器 epsilon | "精度极限" | 最小值 e 使得在给定浮点格式中 1.0 + e != 1.0。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 减去两个几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 上溢 | "数字太大" | 结果超过最大可表示值，变为 inf。exp(89) 会使 float32 上溢。 |
| 下溢 | "数字太小" | 结果比最小可表示正数更接近零，变为 0.0。exp(-104) 会使 float32 下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取 exp(max(x)) 因式来计算 log(sum(exp(x))) 以防止上溢和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在取指数之前减去 max(logits)。数值结果相同，不可能上溢。 |
| 梯度检查 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度进行比较，以捕获实现中的 bug。 |
| 混合精度 | "前向 float16，后向 float32" | 对速度关键的操作使用低精度浮点数，对数值敏感的操作使用高精度浮点数。典型加速比为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以相同的常数。 |
| bfloat16 | "大脑浮点数" | Google 的 16 位格式，具有 8 位指数（与 float32 相同范围）和 7 位尾数（精度低于 float16）。训练首选。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 来自未定义操作（0/0, inf-inf, sqrt(-1)）的特殊浮点值。会传播到所有后续算术运算。 |
| Inf | "无穷大" | 来自上溢或除以零的特殊浮点值。可能组合产生 NaN（inf - inf, inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过评估 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。速度慢但可靠，用于验证。 |

## 进一步阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) —— 权威参考，内容密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) —— NVIDIA 论文，提出了 float16 训练的损失缩放方法
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) —— PyTorch 中混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) —— 为什么 Google 选择这种格式用于 TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) —— 减少浮点求和舍入误差的算法
