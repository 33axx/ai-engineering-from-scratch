# Vision Transformers (ViT)

> 一张图像是一个图块网格。一个句子是一个标记网格。同一个变换器能处理两者。

**类型：** 构建  
**语言：** Python  
**前置要求：** 阶段7 · 05（完整Transformer），阶段4 · 03（CNN），阶段4 · 14（Vision Transformers入门）  
**时间：** 约45分钟  

## 问题

2020年之前，计算机视觉意味着卷积。ImageNet、COCO以及各类检测基准上的所有SOTA模型都使用CNN骨架。Transformer是为语言设计的。

Dosovitskiy等人（2020）—— "An Image is Worth 16x16 Words"——表明你可以完全抛弃卷积。将图像切成固定大小的图块，将每个图块线性投影为嵌入，然后将其序列送入一个标准Transformer编码器。在足够的规模下（ImageNet-21k预训练或更大），ViT能够匹配甚至超越基于ResNet的模型。

ViT开启了2026年一种更广泛的模式：一种架构，多种模态。Whisper将音频标记化。ViT将图像标记化。机器人技术中的动作标记。视频中的像素标记。Transformer不在乎这些——输入一个序列，它就能学习。

到2026年，ViT及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了视觉领域的大部分。在边缘设备和延迟敏感任务上，CNN仍然胜出。其他所有任务都在其栈中某处使用了ViT。

## 概念

![图像 → 图块 → 标记 → 变换器](../assets/vit.svg)

### 步骤1——分块

将一张 `H × W × C` 的图像切分成一个 `N × (P·P·C)` 的扁平图块序列。典型设定：`224 × 224` 图像，`16 × 16` 图块 → 196个图块，每个图块768个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

图块大小是一个杠杆。图块越小 = 标记越多，分辨率更好，但注意力计算成本呈二次增长。图块越大 = 越粗糙，计算越便宜。

### 步骤2——线性嵌入

一个单一的可学习矩阵将每个扁平图块投影到 `d_model` 维。等价于一个卷积核大小为 `P`、步长为 `P` 的卷积。在PyTorch中，这实际上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 两行代码的实现。

### 步骤3——前置 `[CLS]` 标记，添加位置嵌入

- 前置一个可学习的 `[CLS]` 标记。它的最终隐藏状态就是用于分类的图像表示。
- 添加可学习的位置嵌入（原始ViT）或正弦二维位置嵌入（后来的变体）。
- 2024年之后，RoPE被扩展到二维用于定位，有时不显式使用位置嵌入。

### 步骤4——标准Transformer编码器

堆叠L个块，每个块为 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`。与BERT完全相同。没有视觉专用的层。这是该论文教学上的关键结论。

### 步骤5——输出头

对于分类任务：取 `[CLS]` 隐藏状态 → 线性层 → softmax。对于DINOv2或SAM，丢弃 `[CLS]`，直接使用图块嵌入。

### 重要的变体

| 模型 | 年份 | 变更 |
|------|------|------|
| ViT | 2020 | 原始版本。固定图块大小，全局注意力。 |
| DeiT | 2021 | 蒸馏；仅可在ImageNet-1k上训练。 |
| Swin | 2021 | 层次化结构，使用滑动窗口。固定的次二次成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳的通用视觉特征。 |
| ViT-22B | 2023 | 220亿参数；规模定律适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么花了较长时间

ViT需要*大量*数据才能匹配CNN，因为它不具备CNN的归纳偏置（平移不变性、局部性）。如果没有超过1亿张带标签的图像或强大的自监督预训练，在同等计算量下CNN仍然胜出。DeiT在2021年通过蒸馏技巧解决了这个问题；DINOv2在2023年通过自监督永久性地解决了这个问题。

## 构建它

参见 `code/main.py`。纯标准库实现分块 + 线性嵌入 + 合理性检查。不包含训练——任何实际规模的ViT都需要PyTorch和数小时的GPU时间。

### 步骤1：伪造图像

一张24×24的RGB图像，表示为 `(R, G, B)` 元组的行列表。我们使用6×6的图块 → 16个图块，每个图块嵌入向量为108维。

### 步骤2：分块

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：按行主序遍历网格。所有ViT都使用这种顺序。

### 步骤3：线性嵌入

将每个扁平图块乘以一个随机矩阵 `(patch_flat_size, d_model)`。验证前置 `[CLS]` 标记后的输出形状为 `(N_patches + 1, d_model)`。

### 步骤4：计算实际ViT的参数数量

打印ViT-Base的参数数量：12层，12个头，d=768，patch=16。与ResNet-50（约2500万）对比。ViT-Base约为8600万参数。ViT-Large约为3.07亿。ViT-Huge约为6.32亿。

## 使用它

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2嵌入是2026年图像特征的默认选择。** 冻结骨干网络，训练一个小的输出头。适用于分类、检索、检测、图像描述。Meta的DINOv2检查点在所有非文本视觉任务上均优于CLIP。

**图块大小的选择。** 小型模型使用16×16（ViT-B/16）。密集预测任务（分割）使用8×8或14×14（SAM、DINOv2）。非常大的模型使用14×14。

## 交付它

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新的视觉任务选择ViT变体和图块大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证图块数量等于 `(H/P) * (W/P)`，扁平图块维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入——为每个图块的 `行` 和 `列` 分别生成两个独立的正弦编码，然后拼接。将它们输入一个小型PyTorch ViT，并在CIFAR-10上与可学习位置嵌入的精度进行比较。
3. **困难。** 构建一个3层ViT（PyTorch），使用4×4图块在1000张MNIST图像上训练。测量测试精度。现在，在同一1000张图像上添加DINOv2预训练（简化版：仅训练编码器从掩码图块预测图块嵌入）。精度是否提升？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Patch | “视觉Transformer的标记” | 图像中一个 `P × P × C` 区域的扁平像素值向量。 |
| Patchify | “切割并展平” | 将图像切分为不重叠的图块，每个展平成一个向量。 |
| `[CLS]` token | “图像摘要” | 前置的可学习标记；其最终嵌入作为图像表示。 |
| Inductive bias | “模型假设了什么” | ViT的归纳偏置比CNN少，因此需要更多数据来弥补差距。 |
| DINOv2 | “自监督ViT” | 使用图像增强和动量教师无需标签训练。2026年最佳的通用图像特征。 |
| SigLIP | “CLIP的接班人” | ViT + 文本编码器，使用sigmoid对比损失训练；在同计算量下优于CLIP。 |
| Swin | “窗口化ViT” | 层次化ViT，局部注意力 + 滑动窗口；次二次复杂度。 |
| Register tokens | “2023年的技巧” | 几个额外可学习标记，用于吸收注意力汇聚点；改善DINOv2特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2的注册标记修复方法。
