# 感知机

> 感知机是神经网络的原子。将其拆开，你会发现权重、偏置和一个决策。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 1（线性代数直觉）  
**时间：** 约 60 分钟

## 学习目标

- 用 Python 从零实现一个感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并展示 XOR 失败的案例
- 通过组合 OR、NAND 和 AND 门构建一个多层感知机来解决 XOR 问题
- 使用 Sigmoid 激活函数和反向传播训练一个两层网络，自动学习 XOR 问题

## 问题

你已经了解了向量和点积，也知道矩阵将输入转换为输出。但机器如何*学习*使用哪种变换？

感知机回答了这个问题。它是最简单的学习机器：接受一些输入，乘以权重，加上偏置，然后做出一个二元决策。接着进行调整。仅此而已。所有神经网络都是由这个想法层层堆叠而成。

理解感知机意味着理解“学习”在代码中真正的含义：调整数值直到输出与真实值匹配。

## 概念

### 一个神经元，一个决策

感知机接受 n 个输入，每个乘以一个权重，求和后加上偏置，再将结果通过激活函数。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

阶跃函数是残酷的：如果加权和加上偏置 >= 0，输出 1；否则输出 0。

```
step(z) = 1  if z >= 0
           0  if z < 0
```

这是一个线性分类器。权重和偏置定义了一条直线（或高维空间中的超平面），将输入空间分割成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条直线：

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

直线一侧的所有点输出 0，另一侧的所有点输出 1。训练过程就是移动这条直线直到它正确分离类别。

### 学习规则

感知机学习规则非常简单：

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，误差 = 0，什么都不变。如果预测为 0 但实际应为 1，权重增加。如果预测为 1 但实际应为 0，权重减少。学习率控制每次调整的幅度。

### XOR 问题

这就是它失效的地方。看看这些逻辑门：

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

与门和或门是线性可分的：你可以画一条直线将 0 和 1 分开。异或门则不是。没有任何一条直线能将 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本性的限制。单个感知机只能解决线性可分问题。Minsky 和 Papert 在 1969 年证明了这一点，并且这几乎扼杀了神经网络研究长达十年之久。

解决方法是：将感知机堆叠成层。多层感知机可以通过组合两个线性决策形成一个非线性决策来解决 XOR 问题。

## 构建

### 步骤 1：感知机类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### 步骤 2：在逻辑门上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 步骤 3：观察 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它将永远无法收敛。这是单个感知机无法学习 XOR 的硬性证明。

### 步骤 4：用两层网络解决 XOR

技巧：XOR = (x1 OR x2) AND NOT (x1 AND x2)。组合三个感知机：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

所有四种情况都正确。将感知机堆叠成层，可以创建单个感知机无法产生的决策边界。

### 步骤 5：训练一个两层网络

步骤 4 是手动设定权重的。这虽然对 XOR 有效，但在实际问题上你不可能事先知道正确的权重。解决方案：用 Sigmoid 替代阶跃函数，并通过反向传播自动学习权重。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

与步骤 4 有两个关键区别。首先，Sigmoid 替代了阶跃函数——它是平滑的，因此梯度存在。其次，`train` 方法将误差从输出层反向传播到隐藏层，每个权重都根据其对误差的贡献按比例调整。这就是 20 行代码的反向传播。

这是通往第三课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学原理是将链式法则应用于网络图。我们将在那里详细推导。

## 使用

你刚刚从零构建的一切都可以通过一个 import 实现：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你 30 行的 `Perceptron` 类做的是一样的工作。sklearn 版本增加了收敛检查、多种损失函数和稀疏输入支持——但核心循环是一致的：加权求和、阶跃函数、根据误差更新权重。

真正的差距在于规模。生产网络中有什么变化：

- 阶跃函数变为 Sigmoid、ReLU 或其他平滑激活函数
- 权重通过反向传播自动学习（第三课）
- 层数更深：3 层、10 层、100 层以上
- 相同的基本原理：每一层从前一层的输出中创建新的特征

单个感知机只能画直线。将它们堆叠起来，你就可以画出任意形状。

## 输出

本课产生：
- `outputs/skill-perceptron.md` – 涵盖单层与多层架构何时适用的技能点

## 练习

1. 在与非门（通用门——任何逻辑电路都可以由与非门构建）上训练一个感知机。验证其权重和偏置构成了一个有效的决策边界。
2. 修改 Perceptron 类，在每个 epoch 记录决策边界（w1*x1 + w2*x2 + b = 0）。打印在与门训练过程中直线的移动情况。
3. 构建一个 3 输入感知机，仅当至少 2 个输入为 1 时才输出 1（多数投票函数）。这是线性可分的吗？为什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Perceptron | “一个人造神经元” | 一个线性分类器：输入与权重点积，加偏置，通过阶跃函数 |
| Weight | “输入有多重要” | 一个乘数，缩放每个输入对决策的贡献 |
| Bias | “阈值” | 一个常数，平移决策边界，使感知机在零输入时也能激活 |
| Activation function | “那个压缩数值的东西” | 在加权和之后应用的函数——感知机用阶跃函数，现代网络用 Sigmoid/ReLU |
| Linearly separable | “你可以在它们之间画一条线” | 一个数据集，其中单个超平面可以完美分离类别 |
| XOR problem | “感知机做不到的事” | 证明单层网络无法学习非线性可分函数 |
| Decision boundary | “分类器切换的地方” | 超平面 w*x + b = 0，将输入空间划分为两个类别 |
| Multi-layer perceptron | “一个真正的神经网络” | 感知机按层堆叠，每层输出作为下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) —— 开创一切的原论文
- Minsky & Papert, "Perceptrons" (1969) —— 证明单层网络无法解决 XOR 并扼杀了感知机研究十年之久的著作
- Michael Nielsen, "Neural Networks and Deep Learning", 第 1 章 (http://neuralnetworksanddeeplearning.com/) —— 免费在线资源，关于感知机如何组成网络的最佳可视化解释
