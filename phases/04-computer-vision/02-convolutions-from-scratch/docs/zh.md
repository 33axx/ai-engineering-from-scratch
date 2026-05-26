# 从头实现卷积

> 卷积就是一个微小的密集层，你将其滑过图像，在每个位置共享相同的权重。

**类型：** 动手构建
**语言：** Python
**前置知识：** 阶段三（深度学习核心）、阶段四第01课（图像基础）
**时长：** ~75分钟

## 学习目标

- 仅使用 NumPy 从头实现二维卷积，包括嵌套循环版本和向量化的 `im2col` 版本
- 针对任意输入尺寸、卷积核尺寸、填充和步幅组合，计算输出空间大小，并证明公式 `(H - K + 2P) / S + 1` 的正确性
- 手工设计卷积核（边缘检测、模糊、锐化、Sobel），并解释为什么每个卷积核会产生它所产生的激活模式
- 将多个卷积堆叠成一个特征提取器，并将堆叠深度与感受野大小联系起来

## 问题

一张 224x224 的 RGB 图像上，一个全连接层每个神经元需要 224 * 224 * 3 = 150,528 个输入权重。一个含有 1,000 个单元的隐藏层就已经是 1.5 亿个参数 —— 这还是在学到任何有用信息之前。更糟的是，该层没有意识到左上角的狗和右下角的狗是相同的模式。它将每个像素位置视为独立，这对图像而言完全错误：将一只猫平移三个像素不应该迫使网络重新学习这个概念。

图像模型所需的两个性质是 **平移等变性**（输入平移时输出也平移）和 **参数共享**（相同的特征检测器在所有位置运行）。密集层两者都不提供。卷积则免费为你提供了两者。

卷积并非为深度学习而发明。它是支撑 JPEG 压缩、Photoshop 中的高斯模糊、工业视觉中的边缘检测以及所有曾发布的音频滤波器的同一操作。CNN 在 2012 至 2020 年间统治 ImageNet 的原因在于，对于邻近值相关且相同模式可能出现在任何位置的数据，卷积是正确的先验。

## 概念

### 一个卷积核，滑动进行

二维卷积接受一个称为核（或滤波器）的小权重矩阵，将其滑过输入，在每个位置计算逐元素乘积之和。该和成为一个输出像素。

```mermaid
flowchart LR
    subgraph IN["Input (H x W)"]
        direction LR
        I1["5 x 5 image"]
    end
    subgraph K["Kernel (3 x 3)"]
        K1["learned<br/>weights"]
    end
    subgraph OUT["Output (H-2 x W-2)"]
        O1["3 x 3 map"]
    end
    I1 --> |"slide kernel<br/>compute dot product<br/>at each position"| O1
    K1 --> O1

    style IN fill:#dbeafe,stroke:#2563eb
    style K fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

一个具体的 3x3 示例，作用在 5x5 输入上（无填充，步幅 1）：

```
Input X (5 x 5):                Kernel W (3 x 3):

  1  2  0  1  2                   1  0 -1
  0  1  3  1  0                   2  0 -2
  2  1  0  2  1                   1  0 -1
  1  0  2  1  3
  2  1  1  0  1

The kernel slides across every valid 3 x 3 window. Output Y is 3 x 3:

 Y[0,0] = sum( W * X[0:3, 0:3] )
 Y[0,1] = sum( W * X[0:3, 1:4] )
 Y[0,2] = sum( W * X[0:3, 2:5] )
 Y[1,0] = sum( W * X[1:4, 0:3] )
 ... and so on
```

这一公式 —— **共享权重、局部性、滑动窗口** —— 就是整个思想。其余都是簿记工作。

### 输出尺寸公式

给定输入空间尺寸 `H`，卷积核尺寸 `K`，填充 `P`，步幅 `S`：

```
H_out = floor( (H - K + 2P) / S ) + 1
```

记住它。你在每个架构中都会计算几十次。

| 场景 | H | K | P | S | H_out |
|------|---|---|---|---|-------|
| 有效卷积，无填充 | 32 | 3 | 0 | 1 | 30 |
| 相同卷积（保持尺寸） | 32 | 3 | 1 | 1 | 32 |
| 下采样 2 倍 | 32 | 3 | 1 | 2 | 16 |
| 2x2 池化 | 32 | 2 | 0 | 2 | 16 |
| 大感受野 | 32 | 7 | 3 | 2 | 16 |

“相同填充”意味着当 S == 1 时选择 P 使得 H_out == H。对于偶数 K，那是 P = (K - 1) / 2。这就是 3x3 卷积核占主导的原因 —— 它们是最小的、仍有中心的奇数卷积核。

### 填充

没有填充时，每个卷积都会缩小特征图。堆叠 20 个，你的 224x224 图像就会变成 184x184，这既浪费边界上的计算，又使需要匹配形状的残差连接变得复杂。

```
Zero padding (P = 1) on a 5 x 5 input:

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0       Now the kernel can centre on pixel
  0  1  0  2  1  3  0       (0, 0) and still have three rows and
  0  2  1  1  0  1  0       three columns of values to multiply.
  0  0  0  0  0  0  0
```

实践中会遇到的模式：`zero`（最常见）、`reflect`（镜像边缘，避免生成模型中的硬边界）、`replicate`（复制边缘）、`circular`（环绕，用于环面问题）。

### 步幅

步幅是滑动的步长。`stride=1` 是默认值。`stride=2` 将空间维度减半，是在 CNN 内部下采样的经典方式，无需单独的池化层 —— 每个现代架构（ResNet、ConvNeXt、MobileNet）都在某处使用步进卷积代替最大池化。

```
Stride 1 on a 5 x 5 input, 3 x 3 kernel:

  starts: (0,0) (0,1) (0,2)        -> output row 0
          (1,0) (1,1) (1,2)        -> output row 1
          (2,0) (2,1) (2,2)        -> output row 2

  Output: 3 x 3

Stride 2 on the same input:

  starts: (0,0) (0,2)              -> output row 0
          (2,0) (2,2)              -> output row 1

  Output: 2 x 2
```

### 多输入通道

真实图像有三个通道。RGB 输入上的 3x3 卷积实际上是一个 3x3x3 的立方体：每个输入通道对应一个 3x3 切片。在每个空间位置，你将所有三个切片相乘求和，再加上一个偏置。

```
Input:   (C_in,  H,  W)        3 x 5 x 5
Kernel:  (C_in,  K,  K)        3 x 3 x 3 (one kernel)
Output:  (1,     H', W')       2D map

For a layer that produces C_out output channels, you stack C_out kernels:

Weight:  (C_out, C_in, K, K)   e.g. 64 x 3 x 3 x 3
Output:  (C_out, H', W')       64 x 3 x 3

Parameter count: C_out * C_in * K * K + C_out   (the + C_out is biases)
```

最后一行是你规划模型时会计算的内容。在 3 通道输入上一个 64 通道的 3x3 卷积具有 `64 * 3 * 3 * 3 + 64 = 1,792` 个参数。很便宜。

### im2col 技巧

嵌套循环易于阅读但速度慢。GPU 需要大型矩阵乘法。技巧：将输入的每个感受野窗口展平为大矩阵的一列，将卷积核展平为一行，整个卷积就变成了一次矩阵乘法。

```mermaid
flowchart LR
    X["Input<br/>(C_in, H, W)"] --> IM2COL["im2col<br/>(extract patches)"]
    IM2COL --> COLS["Cols matrix<br/>(C_in * K * K, H_out * W_out)"]
    W["Weight<br/>(C_out, C_in, K, K)"] --> FLAT["Flatten<br/>(C_out, C_in * K * K)"]
    FLAT --> MM["matmul"]
    COLS --> MM
    MM --> OUT["Output<br/>(C_out, H_out * W_out)<br/>reshape to (C_out, H_out, W_out)"]

    style X fill:#dbeafe,stroke:#2563eb
    style W fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

每个生产级卷积实现都是此技巧的变体，再加上缓存分块技巧（直接卷积、Winograd、大卷积核的 FFT 卷积）。理解了 im2col 就理解了核心。

### 感受野

单个 3x3 卷积查看 9 个输入像素。堆叠两个 3x3 卷积，第二层中的神经元查看 5x5 个输入像素。三个 3x3 卷积给出 7x7。通常：

```
RF after L stacked K x K convs (stride 1) = 1 + L * (K - 1)

With strides:   RF grows multiplicatively with stride along each layer.
```

“全部向下使用 3x3”能够工作（VGG、ResNet、ConvNeXt）的整个原因在于，两个 3x3 卷积与一个 5x5 卷积看到相同的输入区域，但参数更少，且中间有一个额外的非线性层。

## 动手构建

### 步骤 1：填充数组

从最小的原语开始：一个在 H x W 数组周围用零填充的函数。

```python
import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```

尾部轴技巧 `x.shape[:-2]` 意味着同一函数可以不加修改地作用于 `(H, W)`、`(C, H, W)` 或 `(N, C, H, W)`。

### 步骤 2：使用嵌套循环的二维卷积

参考实现 —— 速度慢，但无歧义。这就是 `torch.nn.functional.conv2d` 在原理上做的事情。

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```

四个嵌套循环（输出通道、行、列，再加上对 C_in、kh、kw 的隐式求和）。这就是你将用于验证所有更快实现正确性的基准真值。

### 步骤 3：用手工设计的卷积核验证

构建一个垂直 Sobel 卷积核，将其应用于一个合成阶梯图像，观察垂直边缘亮起。

```python
def synthetic_step_image():
    img = np.zeros((1, 16, 16), dtype=np.float32)
    img[:, :, 8:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```

期望在列 7（从左到右亮度增加）处出现大的正值，其他地方为零。这个打印结果就是你验证数学正确性的健康检查。

### 步骤 4：im2col

将输入中的每个卷积核窗口转换为一列矩阵。对于 `C_in=3, K=3`，每列有 27 个数字。

```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```

这仍然是一个 Python 循环，但现在繁重的工作将由一个单一向量化的矩阵乘法完成。

### 步骤 5：通过 im2col + matmul 实现快速卷积

用一次矩阵乘法替换四重循环。

```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```

正确性检查：运行两个实现并比较。

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)

print(f"max abs diff: {np.max(np.abs(y_naive - y_im2col)):.2e}")
```

`max abs diff` 应在 `1e-5` 左右 —— 差异来源于浮点累加顺序，而非错误。

### 步骤 6：一组手工设计的卷积核

五个滤波器，展示单个卷积层在训练之前就能表达的内容。

```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    x = img2d[None].astype(np.float32)
    w = kernel[None, None]
    return conv2d_im2col(x, w, padding=1)[0]
```

应用于任何灰度图像，模糊使图像柔和，锐化使边缘更清晰，Sobel-x 照亮垂直边缘，Sobel-y 照亮水平边缘。这些正是 AlexNet 和 VGG 中第一个训练的卷积层最终学到的模式 —— 因为无论后续任务是什么，一个好的图像模型都需要边缘和斑点检测器。

## 使用它

PyTorch 的 `nn.Conv2d` 封装了相同的操作，并带有自动梯度、CUDA 内核和 cuDNN 优化。形状语义完全相同。

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"weight shape: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"bias shape:   {tuple(conv.bias.shape)}")
print(f"param count:  {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)
y = conv(x)
print(f"\ninput  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}")
```

将 `padding=1` 替换为 `padding=0`，输出降至 222x222。将 `stride=1` 替换为 `stride=2`，输出降至 112x112。与你上面记住的公式相同。

## 交付物

本课程生成：

- `outputs/prompt-cnn-architect.md` —— 一个提示，给定输入尺寸、参数预算和目标感受野，设计一个带有每一层正确 K/S/P 的 `Conv2d` 层堆栈。
- `outputs/skill-conv-shape-calculator.md` —— 一个技能，逐层遍历网络规范，返回每个块的输出形状、感受野和参数计数。

## 练习

1. **（简单）** 给定一个 128x128 灰度输入和一个堆栈 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]`，手工计算每个层的输出空间尺寸和感受野。用包含虚拟卷积的 PyTorch `nn.Sequential` 验证。
2. **（中等）** 扩展 `conv2d_naive` 和 `conv2d_im2col` 以接受 `groups` 参数。证明 `groups=C_in=C_out` 重现了深度可分离卷积，并且其参数计数为 `C * K * K` 而非 `C * C * K * K`。
3. **（困难）** 手工实现 `conv2d_im2col` 的反向传播：给定输出的梯度，计算 `x` 和 `w` 的梯度。用相同输入和权重下的 `torch.autograd.grad` 验证。技巧：im2col 的梯度是 `col2im`，且必须累加重叠窗口。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Convolution | "Sliding a filter" | 一个可学习的点积，在每个空间位置应用且权重共享；数学上是互相关，但所有人都称之为卷积 |
| Kernel / filter | "The feature detector" | 一个小权重张量，形状为 (C_in, K, K)，其与输入窗口的点积产生一个输出像素 |
| Stride | "How far you jump" | 连续卷积核放置之间的步长；步幅 2 将每个空间维度减半 |
| Padding | "Zeros on the edges" | 在输入周围添加的额外值，以便卷积核可以覆盖边界像素；`same` 填充使输出尺寸等于输入尺寸 |
| Receptive field | "How much the neuron sees" | 给定输出激活所依赖的原始输入区域，随深度和步幅增加而增长 |
| im2col | "The GEMM trick" | 将每个感受野窗口重排为列，从而使卷积变为一次大型矩阵乘法 —— 每个快速卷积核的核心 |
| Depthwise conv | "One kernel per channel" | `groups == C_in` 的卷积，每个输出通道仅从其匹配的输入通道计算；MobileNet 和 ConvNeXt 的骨干 |
| Translation equivariance | "Shift in, shift out" | 输入平移 k 个像素时输出也平移 k 个像素的性质；由共享权重免费获得 |

## 延伸阅读

- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285) —— 每个课程默默抄袭的填充/步幅/膨胀的权威图表
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/) —— 规范的讲义笔记，包括原始的 im2col 解释
- [The Annotated ConvNet (fast.ai)](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb) —— 一个从手动卷积到训练好的数字分类器的笔记本
- [Receptive Field Arithmetic for CNNs (Dang Ha The Hien)](https://distill.pub/2019/computing-receptive-fields/) —— 感受野计算的论文级交互式解释器
