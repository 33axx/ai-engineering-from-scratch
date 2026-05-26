# 正则化

> 你的模型在训练数据上达到 99%，在测试数据上只有 60%。它记住了数据而非学习了规律。正则化是你对复杂性施加的“税”，迫使模型学会泛化。

**类型：** 实战
**语言：** Python
**前置要求：** 第03.06课（优化器）
**预计时间：** 约75分钟

## 学习目标

- 从头实现带有反向缩放（inverted scaling）的丢弃法（dropout）、L2 权重衰减、批归一化（batch normalization）、层归一化（layer normalization）以及 RMSNorm
- 衡量训练集与测试集准确率差距，通过正则化实验诊断过拟合
- 解释为什么 Transformer 使用 LayerNorm 而非 BatchNorm，以及为什么现代大语言模型更偏爱 RMSNorm
- 根据过拟合的严重程度，应用正确的正则化技术组合

## 问题

一个具有足够参数量的神经网络可以记住任何数据集。这并非假设——Zhang 等人在 2017 年通过使用随机标签在 ImageNet 上训练标准网络证明了这一点。网络在完全随机的标签分配上达到了接近零的训练损失。它记住了百万个没有任何规律可学的随机输入-输出对。训练损失完美，测试准确率为零。

这就是过拟合问题，而且随着模型变大而变得更糟。GPT-3 有 1750 亿个参数。训练集大约有 5000 亿个 token。有了这么多参数，模型有足够的能力逐字记住训练数据的很大一部分。如果没有正则化，模型只会机械地背诵训练样本，而不是学习可泛化的模式。

训练性能与测试性能之间的差距就是过拟合差距。本课中的每一项技术都从不同角度攻击这个差距。丢弃法迫使网络不依赖于任何一个神经元。权重衰减阻止任何一个权重变得过大。批归一化平滑了损失曲面，使优化器找到更平坦、更可泛化的极小值。层归一化做同样的事，但在批归一化失效的地方（小批量、变长序列）依然有效。RMSNorm 通过去掉均值计算，快 10%。每项技术都很简单。但它们合在一起，就是区分记住还是泛化的关键所在。

## 概念

### 过拟合光谱

每个模型都处于从欠拟合（过于简单，无法捕捉模式）到过拟合（过于复杂，甚至捕捉噪声）的光谱上。最佳点在中间，而正则化将模型从过拟合端推向这个最佳点。

```mermaid
graph LR
    Under["Underfitting<br/>Train: 60%<br/>Test: 58%<br/>Model too simple"] --> Good["Good Fit<br/>Train: 95%<br/>Test: 92%<br/>Generalizes well"]
    Good --> Over["Overfitting<br/>Train: 99.9%<br/>Test: 65%<br/>Memorized noise"]

    Dropout["Dropout"] -->|"Pushes left"| Over
    WD["Weight Decay"] -->|"Pushes left"| Over
    BN["BatchNorm"] -->|"Pushes left"| Over
    Aug["Data Augmentation"] -->|"Pushes left"| Over
```

### 丢弃法（Dropout）

最简单的正则化技术，有着最优雅的解释。在训练过程中，以概率 p 随机将每个神经元的输出置为零。

```
output = activation(z) * mask    where mask[i] ~ Bernoulli(1 - p)
```

当 p = 0.5 时，每次前向传播有一半的神经元被置零。网络必须学习冗余的表示，因为它无法预测哪些神经元可用。这防止了“共适应”——神经元学会依赖于特定其他神经元的存在。

集成解释：一个有 N 个神经元的网络使用丢弃法，会产生 2^N 个可能的子网络（每个神经元打开或关闭的所有组合）。使用丢弃法训练，相当于同时训练所有 2^N 个子网络，每个子网络在不同的 mini-batch 上。在测试时，你使用所有神经元（不丢弃），并将输出乘以 (1 - p) 以匹配训练时的期望值。这相当于对 2^N 个子网络的预测进行平均——一个来自单一模型的巨大集成。

在实践中，缩放是在训练时而不是测试时应用的（反向丢弃法）：

```
During training:  output = activation(z) * mask / (1 - p)
During testing:   output = activation(z)   (no change needed)
```

这样做更简洁，因为测试代码完全不需要知道丢弃法。

默认比率：Transformer 用 p = 0.1，MLP 用 p = 0.5，CNN 用 p = 0.2-0.3。丢弃率越高 = 正则化越强 = 越容易欠拟合。

### 权重衰减（L2 正则化）

将所有权重的平方和加到损失上：

```
total_loss = task_loss + (lambda / 2) * sum(w_i^2)
```

正则化项的梯度是 lambda * w。这意味着每一步，每个权重都会按与其大小成比例的量向零收缩。大的权重受到更大的惩罚。模型被推向没有单个权重占主导地位的解。

为什么这有助于泛化：过拟合的模型往往有大的权重，这些权重放大了训练数据中的噪声。权重衰减使权重很小，从而限制了模型的有效容量，迫使模型依赖于鲁棒的、可泛化的特征，而不是记忆下来的特殊之处。

lambda 超参数控制强度。典型值：

- 对于 Transformer 上的 AdamW：0.01
- 对于 CNN 上的 SGD：1e-4
- 对于严重过拟合的模型：0.1

如第 06 课所述：权重衰减和 L2 正则化在 SGD 中是等价的，但在 Adam 中不相等。使用 Adam 训练时，务必使用 AdamW（解耦权重衰减）。

### 批归一化（Batch Normalization）

在将每一层的输出传递到下一层之前，在 mini-batch 上对其进行归一化。

对于某层的一个 mini-batch 的激活值：

```
mu = (1/B) * sum(x_i)           (batch mean)
sigma^2 = (1/B) * sum((x_i - mu)^2)   (batch variance)
x_hat = (x_i - mu) / sqrt(sigma^2 + eps)   (normalize)
y = gamma * x_hat + beta        (scale and shift)
```

Gamma 和 beta 是可学习的参数，如果归一化不是最优的，它们可以让网络撤销归一化。没有它们，你会强迫每一层的输出都是零均值、单位方差，这可能不是网络想要的。

**训练与推理的区分：** 训练时，mu 和 sigma 来自当前 mini-batch。推理时，你使用训练期间累积的滑动平均值（指数移动平均，动量 = 0.1，意味着 90% 旧值 + 10% 新值）。

为什么 BatchNorm 有效仍有争议。原始论文称它减少了“内部协变量偏移”（随着较早层的更新，层输入的分布发生变化）。Santurkar 等人（2018）表明这个解释是错误的。真正的原因是：BatchNorm 使损失曲面更平滑。梯度更具预测性，Lipschitz 常数更小，优化器可以安全地迈出更大的步长。这就是为什么 BatchNorm 允许你使用更高的学习率并更快收敛。

BatchNorm 有一个根本限制：它依赖于批统计量。当批大小为 1 时，均值和方差毫无意义。当批大小很小（< 32）时，统计量噪声很大，反而损害性能。这在目标检测（内存限制批大小）和语言建模（序列长度变化）等任务中很重要。

### 层归一化（Layer Normalization）

在特征维度上而不是在批维度上进行归一化。对于一个样本：

```
mu = (1/D) * sum(x_j)           (feature mean)
sigma^2 = (1/D) * sum((x_j - mu)^2)   (feature variance)
x_hat = (x_j - mu) / sqrt(sigma^2 + eps)
y = gamma * x_hat + beta
```

D 是特征维度。每个样本独立归一化——不依赖于批大小。这就是 Transformer 使用 LayerNorm 而不是 BatchNorm 的原因。序列变长，批大小通常很小（或生成时为 1），并且训练和推理时的计算是相同的。

Transformer 中的 LayerNorm 应用在每个自注意力块和每个前馈块之后（Post-LN），或者之前（Pre-LN，训练更稳定）。

### RMSNorm

不做均值减去的 LayerNorm。由 Zhang 和 Sennrich（2019）提出。

```
rms = sqrt((1/D) * sum(x_j^2))
y = gamma * x / rms
```

仅此而已。没有均值计算，没有 beta 参数。观察结果：LayerNorm 中的重定中心（减去均值）对模型性能贡献很小，但却耗费计算。移除它可以在约 10% 的开销下获得相同的准确率。

LLaMA、LLaMA 2、LLaMA 3、Mistral 以及大多数现代大语言模型都使用 RMSNorm 而不是 LayerNorm。在数十亿参数和数万亿 token 的规模下，那 10% 的节省是显著的。

### 归一化方法比较

```mermaid
graph TD
    subgraph "Batch Normalization"
        BN_D["Normalize across BATCH<br/>for each feature"]
        BN_S["Batch: [x1, x2, x3, x4]<br/>Feature 1: normalize [x1f1, x2f1, x3f1, x4f1]"]
        BN_P["Needs batch > 32<br/>Different train vs eval<br/>Used in CNNs"]
    end
    subgraph "Layer Normalization"
        LN_D["Normalize across FEATURES<br/>for each sample"]
        LN_S["Sample x1: normalize [f1, f2, f3, f4]"]
        LN_P["Batch-independent<br/>Same train vs eval<br/>Used in Transformers"]
    end
    subgraph "RMS Normalization"
        RN_D["Like LayerNorm<br/>but skip mean subtraction"]
        RN_S["Just divide by RMS<br/>No centering"]
        RN_P["10% faster than LayerNorm<br/>Same accuracy<br/>Used in LLaMA, Mistral"]
    end
```

### 数据增强：一种正则化手段

不是修改模型，而是修改数据。在保留标签的同时变换训练输入：

- 图像：随机裁剪、翻转、旋转、颜色抖动、cutout
- 文本：同义词替换、回译、随机删除
- 音频：时间拉伸、音高偏移、添加噪声

其效果与正则化完全相同：它增加了训练集的有效大小，使得模型更难记住特定样本。只见过每个图像原始版本的模型可以记住它。而见过每个图像 50 个增强版本的模型被迫学习不变的结构。

### 早停（Early Stopping）

最简单的正则化：当验证损失开始增加时停止训练。此时模型尚未过拟合。实践中，你每个 epoch 跟踪验证损失，保存最佳模型，并在一个“耐心”窗口（通常 5-20 个 epoch）内继续训练。如果验证损失在耐心窗口内没有改善，则停止并加载保存的最佳模型。

### 何时应用何种技术

```mermaid
flowchart TD
    Gap{"Train-test<br/>accuracy gap?"} -->|"> 10%"| Heavy["Heavy regularization"]
    Gap -->|"5-10%"| Medium["Moderate regularization"]
    Gap -->|"< 5%"| Light["Light regularization"]

    Heavy --> D5["Dropout p=0.3-0.5"]
    Heavy --> WD2["Weight decay 0.01-0.1"]
    Heavy --> Aug["Aggressive data augmentation"]
    Heavy --> ES["Early stopping"]

    Medium --> D3["Dropout p=0.1-0.2"]
    Medium --> WD1["Weight decay 0.001-0.01"]
    Medium --> Norm["BatchNorm or LayerNorm"]

    Light --> D1["Dropout p=0.05-0.1"]
    Light --> WD0["Weight decay 1e-4"]
```

## 动手构建

### 第 1 步：丢弃法（训练与评估模式）

```python
import random
import math


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads
```

### 第 2 步：L2 权重衰减

```python
def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty

def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]
```

### 第 3 步：批归一化

```python
class BatchNorm:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output
```

### 第 4 步：层归一化

```python
class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output
```

### 第 5 步：RMSNorm

```python
class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output
```

### 第 6 步：有正则化与无正则化的训练

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history
```

## 使用它

PyTorch 提供了所有归一化和正则化模块：

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),
)

model.train()
out_train = model(torch.randn(32, 784))

model.eval()
out_test = model(torch.randn(1, 784))
```

`model.train()` / `model.eval()` 的切换至关重要。它开关丢弃法，并告诉 BatchNorm 使用批统计量还是滑动统计量。在推理之前忘记调用 `model.eval()` 是深度学习中最常见的 bug 之一。你的测试准确率会随机波动，因为丢弃法仍然有效，而 BatchNorm 正在使用 mini-batch 统计量。

对于 Transformer，模式不同：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.ff(x))
        return x
```

使用 LayerNorm 而非 BatchNorm。丢弃率 p=0.1，而非 p=0.5。这些是 Transformer 的默认设置。

## 交付

本课输出：
- `outputs/prompt-regularization-advisor.md` —— 一个 prompt，用于诊断过拟合并推荐正确的正则化策略

## 练习

1. 实现用于二维数据的空间丢弃法：不是丢弃单个神经元，而是丢弃整个特征通道。通过将连续的特征组视为通道并丢弃整组来模拟。在 circle 数据集上，使用 hidden_size=32，比较空间丢弃法与标准丢弃法在训练-测试差距上的表现。

2. 将第 05 课的标签平滑与本课的丢弃法相结合。使用四种配置进行训练：两者都不用、仅丢弃法、仅标签平滑、两者都用。测量每种配置下最终的训练-测试准确率差距。哪种组合产生的差距最小？

3. 在 circle 数据集的网络中，在隐藏层和激活函数之间添加一个 BatchNorm 层。分别以学习率 0.01、0.05 和 0.1 进行有 BatchNorm 和无 BatchNorm 的训练。BatchNorm 应该允许在更高学习率下稳定训练，而普通网络会发散。

4. 实现早停：每个 epoch 跟踪测试损失，保存最佳权重，如果测试损失在 20 个 epoch 内没有改善则停止。运行正则化网络 1000 个 epoch。报告哪个 epoch 取得了最佳测试准确率，以及你节省了多少次 epoch 的计算量。

5. 在一个 4 层网络（不仅仅是 2 层）上比较 LayerNorm 与 RMSNorm。使用相同权重初始化两者。训练 200 个 epoch，比较最终准确率、训练速度（每 epoch 时间）以及第一层的梯度大小。验证 RMSNorm 是否更快且准确率相同。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Overfitting | “模型记住了数据” | 当模型的训练性能显著超过测试性能时，表明它学习了噪声而非信号 |
| Regularization | “防止过拟合” | 任何约束模型复杂度以提高泛化能力的技术：丢弃法、权重衰减、归一化、数据增强 |
| Dropout | “随机删除神经元” | 训练时以概率 p 随机将神经元置零，强制冗余表示；等价于训练一个集成 |
| Weight decay | “L2 惩罚” | 每一步通过减去 lambda * w 将所有权重向零收缩；通过权重大小惩罚复杂度 |
| Batch normalization | “按批次归一化” | 在批次维度上归一化层输出，训练时使用批统计量，推理时使用滑动平均统计量 |
| Layer normalization | “按样本归一化” | 在每个样本内沿特征维度归一化；不依赖批次，用于变批次大小的 Transformer |
| RMSNorm | “不带均值的 LayerNorm” | 均方根归一化；省去 LayerNorm 的均值减法，获得相同准确率并提速约 10% |
| Early stopping | “在过拟合前停止” | 当验证损失不再改善时停止训练；最简单的正则化器，常与其他方法一起使用 |
| Data augmentation | “从少量数据中获取更多” | 变换训练输入（翻转、裁剪、添加噪声）以增加有效数据集大小并迫使学习不变性 |
| Generalization gap | “训练-测试差距” | 训练性能与测试性能之间的差异；正则化旨在最小化这个差距 |

## 延伸阅读

- Srivastava 等人，“Dropout: A Simple Way to Prevent Neural Networks from Overfitting”（2014）——原始丢弃法论文，包含集成解释和大量实验
- Ioffe 和 Szegedy，“Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”（2015）——介绍了 BatchNorm 及其训练过程，是深度学习被引用最多的论文之一
- Zhang 和 Sennrich，“Root Mean Square Layer Normalization”（2019）——展示了 RMSNorm 在减少计算量的同时达到与 LayerNorm 相同的准确率；被 LLaMA 和 Mistral 采用
- Zhang 等人，“Understanding Deep Learning Requires Rethinking Generalization”（2017）——里程碑式论文，展示了神经网络可以记忆随机标签，挑战了传统的泛化观点
