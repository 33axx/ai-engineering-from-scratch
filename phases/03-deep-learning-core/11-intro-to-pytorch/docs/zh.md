# PyTorch 入门

> 你已经用活塞和曲轴亲手打造了引擎——现在来学学大家真正在开的这一款吧。

**类型：** 动手构建  
**语言：** Python  
**前置要求：** 课程 03.10（构建你自己的迷你框架）  
**时间：** 约 75 分钟

## 学习目标

- 使用 PyTorch 的 `nn.Module`、`nn.Sequential` 和 `autograd` 构建并训练神经网络
- 使用 PyTorch 张量、GPU 加速以及标准训练循环（`zero_grad`、`forward`、`loss`、`backward`、`step`）
- 将你从零构建的迷你框架组件转换为其 PyTorch 等价物
- 在相同任务上，对比你的纯 Python 框架与 PyTorch 的训练速度并进行分析

## 问题

你已经有了一个可以工作的迷你框架。线性层、ReLU、Dropout、批量归一化、Adam、DataLoader、训练循环——一应俱全。它能在纯 Python 中对一个圆型分类问题训练一个 4 层网络。

然而，在同一个问题上，它比 PyTorch 慢 500 倍以上。

你的迷你框架用嵌套的 Python 循环一次处理一个样本。而 PyTorch 将同样的操作分发给经过优化的 C++/CUDA 内核，这些内核在 GPU 上运行。在一块 NVIDIA A100 上，PyTorch 训练一个 ResNet-50（2560 万参数）处理 ImageNet（128 万张图像）大约需要 6 小时。你的框架在同样的任务上大概需要 3000 小时——前提是它还没耗尽内存。

速度差距不是唯一的鸿沟。你的框架不支持 GPU，没有自动微分——你必须为每个模块手动编写 `backward()`。它也没有序列化、分布式训练、混合精度，更没有除 `print` 语句之外的方法来调试梯度流。

PyTorch 填补了所有这些空白。而且，它保持了与你已构建的完全相同的思维模型：`Module`、`forward()`、`parameters()`、`backward()`、`optimizer.step()`。这些概念是一一对应的，语法也几乎一模一样。区别在于，PyTorch 在你从零设计的同一套接口背后，包裹了整整一个十年的系统工程积淀。

## 概念

### 为什么 PyTorch 能胜出

2015 年，TensorFlow 要求你在运行任何东西之前先定义静态计算图。你先构建图，然后编译它，再把数据送进去。调试意味着盯着图的可视化发呆。改变架构则意味着从头重建整个图。

PyTorch 在 2017 年带着不同的哲学面世：**即时执行**（eager execution）。你写 Python，它立即运行。`y = model(x)` 会**此刻**就计算出 y，而不是“向图中添加一个节点，该节点将在未来计算 y”。这意味着标准的 Python 调试工具都能用：`print()` 可以工作，`pdb` 可以工作，`forward` 中的 `if/else` 也可以工作。

到 2020 年，市场已经给出了答案。PyTorch 在机器学习研究论文中的占比从 2017 年的 7% 增长到 2022 年的 75% 以上。Meta、Google DeepMind、OpenAI、Anthropic 和 Hugging Face 都使用 PyTorch 作为主要框架。TensorFlow 2.x 也采用了即时执行——这等于默认了 PyTorch 的设计是正确的。

教训：开发者体验会不断累加。一个慢 10% 但调试快 50% 的框架，每一次都会赢。

### 张量

张量是一种多维数组，具有三个关键属性：形状（shape）、数据类型（dtype）和设备（device）。

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**形状** 是维度信息。标量是 `()`，向量是 `(n,)`，矩阵是 `(m, n)`，一批图像是 `(batch, channels, height, width)`。

**数据类型** 控制精度和内存占用。

| dtype | 位数 | 范围 | 使用场景 |
|-------|------|------|----------|
| float32 | 32 | ~7 位十进制数字 | 默认训练 |
| float16 | 16 | ~3.3 位十进制数字 | 混合精度 |
| bfloat16 | 16 | 与 float32 相同范围，精度更低 | 大模型训练 |
| int8 | 8 | -128 到 127 | 量化推理 |

**设备** 决定了计算发生的位置。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

每一个操作都要求所有张量位于同一个设备上。这是初学者遇到的最常见的 PyTorch 错误：`RuntimeError: Expected all tensors to be on the same device`。解决方法是：在计算之前将所有张量移动到同一个设备。

**重塑** 是常数时间操作——它只改变元数据，不改变数据本身。

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### Autograd

你的迷你框架要求你为每个模块实现 `backward()`。PyTorch 不需要。它会把每个张量上的操作记录到一个有向无环图（计算图）中，然后反向遍历该图来自动计算梯度。

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

与你框架的关键区别：PyTorch 使用基于磁带（tape）的自动微分。前向传递过程中，每个操作都被附加到一条“磁带”上。调用 `.backward()` 会反向重放磁带。

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

Autograd 的三条规则：

1. 只有 `requires_grad=True` 的叶子张量才会累积梯度
2. 梯度默认会累积——每次反向传播前要调用 `optimizer.zero_grad()`
3. `torch.no_grad()` 会禁用梯度跟踪（在评估时使用）

### nn.Module

`nn.Module` 是 PyTorch 中所有神经网络组件的基类。你在第 10 课已经构建过这个抽象。PyTorch 的版本增加了自动参数注册、递归模块发现、设备管理和状态字典序列化。

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

当你在 `__init__` 中将一个 `nn.Module` 或 `nn.Parameter` 赋值给属性时，PyTorch 会自动注册它。`model.parameters()` 会递归收集所有已注册的参数。这就是为什么你不再需要像在迷你框架中那样手动收集权重。

关键构建模块：

| 模块 | 功能 | 参数数量 |
|--------|-------------|------------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D 卷积 | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | 归一化激活值 | 2 * features |
| nn.Dropout(p) | 随机置零 | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | 高斯误差线性单元 | 0 |
| nn.Embedding(vocab, dim) | 查找表 | vocab * dim |
| nn.LayerNorm(dim) | 逐样本归一化 | 2 * dim |

### 损失函数与优化器

PyTorch 提供了你之前手撸的所有组件的生产级版本。

**损失函数**（来自 `torch.nn`）：

| 损失函数 | 任务 | 输入 |
|----------|------|------|
| nn.MSELoss() | 回归 | 任意形状 |
| nn.CrossEntropyLoss() | 多类分类 | logits（不是 softmax） |
| nn.BCEWithLogitsLoss() | 二分类 | logits（不是 sigmoid） |
| nn.L1Loss() | 回归（鲁棒） | 任意形状 |
| nn.CTCLoss() | 序列对齐 | 对数概率 |

注意：`CrossEntropyLoss` 在内部组合了 `LogSoftmax` + `NLLLoss`。传入原始 logits，而不是 softmax 输出。这是一个常见错误，会导致梯度被静默地计算错误。

**优化器**（来自 `torch.optim`）：

| 优化器 | 何时使用 | 典型学习率 |
|--------|----------|------------|
| SGD(params, lr, momentum) | CNN、调优好的流程 | 0.01--0.1 |
| Adam(params, lr) | 默认起点 | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformer、微调 | 1e-4--1e-3 |
| LBFGS(params) | 小规模、二阶优化 | 1.0 |

### 训练循环

每一个 PyTorch 训练循环都遵循相同的 5 步模式。你从第 10 课已经知道了这一点。

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

标准模式：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

批循环内的五行代码。这五行代码训练了 GPT-4、Stable Diffusion 和 LLaMA。架构在变，数据在变，但这五行代码不变。

### Dataset 与 DataLoader

PyTorch 的 `Dataset` 是一个抽象类，包含两个方法：`__len__` 和 `__getitem__`。`DataLoader` 将其包装，提供批处理、打乱和多进程数据加载功能。

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4` 会打开 4 个进程并行加载数据，而 GPU 则在当前批次上训练。在磁盘密集型工作负载（大图像、音频）上，仅此一项就可以让训练速度翻倍。

### GPU 训练

将模型移动到 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

这会递归地将每一个参数和缓冲区移到 GPU。然后在训练期间移动每个批次：

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**混合精度** 通过在前向/反向传播中使用 float16 计算，同时将主权重保留在 float32，从而将现代 GPU（A100、H100、RTX 4090）上的内存使用减半并使吞吐量翻倍：

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 对比：迷你框架 vs PyTorch vs JAX

| 特性 | 迷你框架（第 10 课） | PyTorch | JAX |
|------|---------------------|---------|-----|
| 自动微分 | 手动 backward() | 基于磁带的 autograd | 函数式变换 |
| 执行方式 | 即时（Python 循环） | 即时（C++ 内核） | 跟踪 + JIT 编译 |
| GPU 支持 | 无 | 有（CUDA、ROCm、MPS） | 有（CUDA、TPU） |
| 速度（MNIST MLP） | ~300 秒/epoch | ~0.5 秒/epoch | ~0.3 秒/epoch |
| 模块系统 | 自定义 Module 类 | nn.Module | 无状态函数（Flax/Equinox） |
| 调试 | print() | print()、pdb、breakpoint() | 更困难（JIT 跟踪破坏 print） |
| 生态系统 | 无 | Hugging Face、Lightning、timm | Flax、Optax、Orbax |
| 学习曲线 | 你亲手构建 | 中等 | 陡峭（函数式范式） |
| 生产使用 | 玩具问题 | Meta、OpenAI、Anthropic、HF | Google DeepMind、Midjourney |

## 动手构建

一个使用 PyTorch 基本组件在 MNIST 上训练的 3 层 MLP。不使用高层封装。不用 `torchvision.datasets`。我们自己下载并解析原始数据。

### 第 1 步：从原始文件加载 MNIST

MNIST 以 4 个 gzip 文件的形式分发：训练图像（60,000 x 28 x 28）、训练标签、测试图像（10,000 x 28 x 28）、测试标签。我们下载它们并解析二进制格式。

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### 第 2 步：定义模型

一个 3 层 MLP：784 -> 256 -> 128 -> 10。ReLU 激活函数。Dropout 用于正则化。不使用批量归一化以保持简单。

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

输出层产生 10 个原始 logits（每个数字一个）。没有 softmax——`CrossEntropyLoss` 会在内部处理。

参数量：784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146。按现代标准非常小。GPT-2 small 有 1.24 亿参数。这个模型几秒钟就能训练完。

### 第 3 步：训练循环

标准的前向-损失-反向-更新模式。

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

注意评估时使用 `torch.no_grad()`。这会禁用 autograd，减少内存使用并加速推理。如果没有它，PyTorch 会构建一个你永远用不到的计算图。

### 第 4 步：整合所有代码

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

10 个 epoch 后的预期输出：测试准确率约 97.8%。CPU 训练时间：约 30 秒。GPU 训练时间：约 5 秒。使用相同的架构，你的迷你框架训练大约需要 45 分钟。

## 使用它

### 快速对比：迷你框架 vs PyTorch

| 迷你框架（第 10 课） | PyTorch |
|----------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` then `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| 无 GPU | `model.to("cuda")` |
| 每个模块手动实现 backward | Autograd 处理一切 |

接口几乎相同。区别在于底层的所有实现。

### 保存和加载模型

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

始终保存 `state_dict()`（参数字典），而不是模型对象。保存模型对象会使用 pickle，当你重构代码时会出问题。状态字典是可移植的。

### 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

PyTorch 内置了 15 种以上的调度器：`StepLR`、`ExponentialLR`、`CosineAnnealingLR`、`OneCycleLR`、`ReduceLROnPlateau`。它们都接入同一个优化器接口。

## 交付

本课程产出两个产物：

- `outputs/prompt-pytorch-debugger.md` —— 一个用于诊断常见 PyTorch 训练失败的提示
- `outputs/skill-pytorch-patterns.md` —— 一个关于 PyTorch 训练模式的技能参考

## 练习

1. **添加批量归一化。** 在每个线性层之后（激活函数之前）插入 `nn.BatchNorm1d`。与仅使用 dropout 的版本比较测试准确率和训练速度。批量归一化应该能在更少的 epoch 内达到 98% 以上的准确率。

2. **实现学习率查找器。** 用一个 epoch 进行训练，学习率指数增长（从 1e-7 到 1.0）。绘制损失 vs 学习率曲线。最优学习率恰好位于损失开始上升之前的位置。使用此方法为 MNIST 模型选择一个更好的学习率。

3. **移植到 GPU 并使用混合精度。** 在训练循环中添加 `torch.amp.autocast` 和 `GradScaler`。测量带混合精度和不带混合精度时的吞吐量（样本/秒）。在 A100 上，预计约 2 倍加速。

4. **构建一个自定义 Dataset。** 下载 Fashion-MNIST（与 MNIST 格式相同，但包含服装物品）。实现一个 `FashionMNISTDataset(Dataset)` 类，包含 `__getitem__` 和 `__len__`。用同样的 MLP 训练并比较准确率。Fashion-MNIST 更难——预期准确率约 88%，而不是 98%。

5. **用 SGD + 动量替换 Adam。** 使用 `SGD(params, lr=0.01, momentum=0.9)` 训练。比较收敛曲线。然后添加一个 `CosineAnnealingLR` 调度器，看看 SGD 能否在第 10 个 epoch 追上 Adam。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|------------|----------|
| 张量 | “一个多维数组” | 一个具有类型、设备感知能力、且每个操作都内置自动微分支持的数组 |
| Autograd | “自动反向传播” | 一个基于磁带的系统，在前向传递时记录操作，然后在反向传递时重放它们以计算精确的梯度 |
| nn.Module | “一个层” | 任何可微分计算块的基类——注册参数、支持嵌套、处理训练/评估模式 |
| state_dict | “模型权重” | 一个 OrderedDict，将参数名映射到张量——训练模型的可移植、可序列化表示 |
| .backward() | “计算梯度” | 反向遍历计算图，为每个 `requires_grad=True` 的叶子张量计算并累积梯度 |
| .to(device) | “移到 GPU” | 递归地将所有参数和缓冲区转移到指定的设备（CPU、CUDA、MPS） |
| DataLoader | “数据管道” | 一个迭代器，对 Dataset 进行批处理、打乱，并可选择并行化数据加载 |
| 混合精度 | “使用 float16” | 用 float16 进行前向/反向传播以获得速度，同时保留 float32 主权重以保证数值稳定性 |
| 即时执行 | “现在就运行” | 操作在调用时立即执行，而不是延迟到后续编译步骤——这是 PyTorch 区别于 TF 1.x 的核心设计选择 |
| zero_grad | “重置梯度” | 将所有参数梯度置零，以便进行下一次反向传播——因为 PyTorch 默认会累积梯度 |

## 延伸阅读

- Paszke 等人，《PyTorch: An Imperative Style, High-Performance Deep Learning Library》（2019）—— 解释 PyTorch 设计权衡的原始论文
- PyTorch 教程："Learning PyTorch with Examples"（https://pytorch.org/tutorials/beginner/pytorch_with_examples.html）—— 从张量到 nn.Module 的官方路径
- PyTorch 性能调优指南（https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html）—— 混合精度、DataLoader 工作进程、固定内存及其他生产级优化
- Horace He，《Making Deep Learning Go Brrrr》（https://horace.io/brrr_intro.html）—— 为什么 GPU 训练快，以及针对 PyTorch 的优化策略
