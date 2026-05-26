# 图像分类

> 分类器是从像素到类别概率分布的函数。其他都是管道问题。

**类型：** 构建项目  
**语言：** Python  
**前置知识：** 阶段2 第09课（模型评估），阶段3 第10课（微型框架），阶段4 第03课（卷积神经网络）  
**时间：** ~75分钟  

## 学习目标

- 在CIFAR-10上构建端到端图像分类流水线：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（数据加载器、损失函数、优化器、调度器、数据增强）的作用，并预测破坏其中任何一个会在损失曲线上如何体现
- 从头实现 mixup、cutout 和标签平滑，并说明各自何时值得加入
- 读取混淆矩阵和每类精确率/召回率表格，从聚合准确率之外诊断数据集和模型的失败原因

## 问题描述

每个上线的视觉任务在某个层次上都归结为图像分类。检测是对区域分类，分割是对像素分类，检索是通过与类别质心的相似度进行排序。把分类这件事搞对——数据集循环、增强策略、损失函数、评估——是能够迁移到该阶段所有其他任务的技能。

大多数分类错误并不在模型里。它们藏在流水线中：一个损坏的归一化、一个未打乱的训练集、使标签扭曲的增强、被训练数据污染的验证集划分、一个在第30轮后悄然发散的学习率。一个正确配置下能在CIFAR-10上达到93%的CNN，在故障配置下通常只能达到70-75%，而且损失曲线全程看起来都很合理。

本课程手动连接整个流水线，使得每个部分都可被检查。你不会使用 `torchvision.datasets` 中任何可能隐藏bug的东西。

## 概念

### 分类流水线

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环中的每一行都可能藏着一个bug。交叉熵接收原始 logits，而不是 softmax 输出，因此在损失之前任何 `model(x).softmax()` 都会安静地计算出错误的梯度。增强只应用于输入，不应用于标签——除了 mixup，它同时混合两者。`optimizer.zero_grad()` 必须在每一步执行一次；跳过它会导致梯度累加，看起来像是学习率极度不稳定。上述每个错误都会压低学习曲线，但不会抛出错误。

### 交叉熵、logits 和 softmax

分类器为每张图像生成 `C` 个数字，称为 logits。应用 softmax 将其转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

交叉熵衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右边的形式是数值稳定的（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax + NLL 融合为一个操作，直接接受原始 logits。如果你先自行应用 softmax 几乎总是个错误——你计算的是 log(softmax(softmax(z)))，一个无意义的量。

### 为什么数据增强有效

CNN 具有平移的归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内建的不变性。教会它这些不变性的唯一方法是展示使用它们的像素。训练期间的每个随机变换都在说：“这两张图像有相同的标签；学习忽略差异的特征。”

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

规则：增强必须保持标签不变。对数字进行 cutout 和旋转可能把 "6" 变成 "9"；对于那个数据集，你应使用较小的旋转范围，并选择尊重数字特定不变性的增强。

### Mixup 和 CutMix

普通的数据增强变换像素但保持标签为独热编码。**Mixup** 和 **CutMix** 通过插值两者来打破这一点。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

为什么有帮助：模型停止记忆尖锐的独热目标，转而学习在类别之间插值。训练损失上升，测试准确率上升。对于任何分类器来说，这是最便宜且最有效的鲁棒性升级。

### 标签平滑

mixup 的变体。不再针对 `[0, 0, 1, 0, 0]` 训练，而是针对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]`，其中 `eps` 很小，如 0.1。阻止模型产生任意尖锐的 logits，几乎不花代价就能改善校准。自 PyTorch 1.10 起内置在 `nn.CrossEntropyLoss(label_smoothing=0.1)` 中。

### 超越准确率的评估

聚合准确率掩盖了不平衡。一个90-10的二元分类器如果始终预测多数类，分数为90%。真正告诉你发生了什么的有用工具：

- **每类准确率** —— 每个类别一个数字；立即暴露表现不佳的类别。
- **混淆矩阵** —— C x C 网格，第 i 行第 j 列 = 真实类别 i 被预测为 j 的计数；对角线正确，非对角线是模型犯错的地方。
- **Top-1 / Top-5** —— 正确类别是否在 top 1 或 top 5 预测中；Top-5 对 ImageNet 很重要，因为像 "Norwich terrier" 和 "Norfolk terrier" 这样的类别确实是模糊的。
- **校准（ECE）** —— 0.8 置信度的预测在 80% 的情况下正确吗？现代网络系统性地过度自信；通过温度缩放或标签平滑修复。

## 构建过程

### 步骤1：确定性合成数据集

CIFAR-10 存储在磁盘上。为了使本课程可重复且快速，我们构建一个看起来像 CIFAR 的合成数据集——32x32 RGB 图像，具有模型必须学习的类别特定结构。完全相同的流水线可以直接在真实 CIFAR-10 上运行，无需修改。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类有自己的调色板和频率模式，加上高斯噪声，迫使模型学习信号而不是记忆像素。十个类别，每类一千张图像，经过排列打乱。

### 步骤2：归一化与数据增强

每个视觉流水线都有的两种变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

先反射填充再裁剪，而不是零填充，因为黑色边框是一种信号，模型会学习以一种无用的方式忽略它。

### 步骤3：Mixup

在训练步骤内部混合两张图像和两个标签。实现为批处理变换，因此它位于前向传播旁边，而不是数据集内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是针对软标签分布的交叉熵。当目标是严格的独热编码时，它退化为通常的独热情况。

### 步骤4：训练循环

完整配方：在数据上遍历一次，每批一次梯度，每轮一次调度器步进。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次编写训练循环时检查的五个不变条件：

1. 训练前 `model.train()`，评估前 `model.eval()` —— 切换 dropout 和 batch norm 行为。
2. `.backward()` 之前执行 `.zero_grad()`。
3. 累加指标时使用 `.item()`，这样不会有任何东西保留计算图。
4. 评估期间使用 `@torch.no_grad()` —— 节省内存和时间，防止细微意外。
5. 对原始 logits 取 argmax，而不是 softmax —— 结果相同，少一次操作。

### 步骤5：整合

使用上一课中的 `TinyResNet`，训练几个周期，进行评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，该流程在五个周期内就能达到近乎完美的验证准确率，这就是重点：流水线是正确的，模型可以学习可学习的内容。将数据集换成真实 CIFAR-10，相同的循环无需修改即可训练到 ~90%。

### 步骤6：读取混淆矩阵

准确率永远不会告诉你模型在何处失败。混淆矩阵会。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类，列是预测。类别 3 和 5 之间出现非对角计数簇意味着模型混淆了这两个类别，这为定向数据收集或按类别增强提供了起点。

## 使用

`torchvision` 将上述所有内容包装成惯用组件。对于真实 CIFAR-10，完整流水线是四行代码加一个训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

需要注意两点：均值/标准差是**数据集特定的**——在 CIFAR-10 训练集上计算，而非 ImageNet——反射填充是社区默认的裁剪策略。在这里复制粘贴 ImageNet 的统计信息会造成约 1% 的准确率泄漏，直到有人分析模型才会被发现。

## 交付物

本课程产出：

- `outputs/prompt-classifier-pipeline-auditor.md` —— 一个提示词，用于审计训练脚本中上述五个不变条件，并报告第一个违反的地方。
- `outputs/skill-classification-diagnostics.md` —— 一项技能，给定混淆矩阵和类别名称列表后，总结每类失败并提出最具影响力的单一修复措施。

## 练习

1. **(简单)** 在合成数据集上使用 mixup 和不使用 mixup 分别训练模型五个周期。绘制两者的训练和验证损失曲线。解释为什么使用 mixup 时训练损失更高，但验证准确率相近或更好。
2. **(中等)** 实现 Cutout——在每个训练图像中随机遮盖一个 8x8 的方形区域——并进行消融实验：无增强、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每种配置的验证准确率。
3. **(困难)** 构建一个 CIFAR-100 流水线（100 个类别，相同输入大小），并复现 ResNet-34 的训练运行，使其与发表准确率相差在 1% 以内。额外任务：扫描三个学习率和两个权重衰减值，记录到本地 CSV，生成最终混淆矩阵的 top-confusions 表格。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| Logits | "原始输出" | 每张图像 C 个数的预 softmax 向量；交叉熵期望这些，而不是 softmax 后的值 |
| 交叉熵 | "损失" | 正确类别的负对数概率；将对数 softmax 和 NLL 合并为一个稳定操作 |
| DataLoader | "批处理器" | 将数据集与打乱、分批和（可选）多工作线程加载封装在一起；一半训练 bug 被归咎于它 |
| 数据增强 | "随机变换" | 训练时任何保持标签不变的像素级变换；教会 CNN 不天然具备的不变性 |
| Mixup / CutMix | "混合两张图片" | 融合输入和标签，使分类器学习平滑插值而非硬边界 |
| 标签平滑 | "更软的目标" | 用 (1-eps, eps/(C-1), ...) 替换独热编码；改善校准并略微提升准确率 |
| Top-k 准确率 | "Top-5" | 正确类别在 k 个最高概率预测中；用于具有真正模糊类别的数据集 |
| 混淆矩阵 | "错误所在之处" | C x C 表格，其中第 (i, j) 项计数真实类别 i 被预测为 j 的图像数；对角线是正确的，非对角线告诉你该修复什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) —— 仍然是单页上最清晰的训练流水线讲解
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) —— 每个小技巧加在一起可在 ImageNet 上为 ResNet 准确率增加 3-4%
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) —— 原始的 mixup 论文；三页理论加上令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) —— 证明现代网络校准不良并用一个标量参数修复的论文
