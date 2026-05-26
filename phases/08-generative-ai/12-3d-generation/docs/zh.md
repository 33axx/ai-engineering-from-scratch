# 3D 生成

> 3D 是 2D 到 3D 迁移效果最强的模态。2023 年的突破是 3D 高斯泼溅（3D Gaussian Splatting）。2024–2026 年的生成式推进在此基础上叠加了多视图扩散 + 3D 重建，从而从单个提示或照片中生成物体和场景。

**类型：** 学习  
**语言：** Python  
**前置条件：** 阶段 4（视觉），阶段 8 · 07（潜在扩散）  
**时间：** ~45 分钟  

## 问题所在

3D 内容制作令人痛苦：

- **表示方式。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每种都有其权衡。
- **数据稀缺。** ImageNet 拥有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）约有 1000 万个物体，但大多数质量较低。
- **内存。** 一个 512³ 的体素网格包含 1.28 亿个体素；一个有实用价值的场景 NeRF 每射线需要 100 万个样本。生成比重建更难。
- **监督信号。** 对于 2D 图像，你有像素数据。对于 3D，你通常只有少量 2D 视图，需要将其提升到 3D。

2026 年的技术栈将这两个问题分开处理。首先，使用扩散模型生成 *2D 多视图图像*。其次，将 *3D 表示*（通常是高斯泼溅）拟合到这些图像上。

## 概念

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示方式：3D 高斯泼溅（Kerbl 等人，2023）

将场景表示为约 100 万个 3D 高斯的点云。每个高斯有 59 个参数：位置（3），协方差（6，或四元数 4 + 缩放 3），不透明度（1），球谐颜色（3 阶 48，0 阶 3）。

渲染 = 投影 + 阿尔法合成。速度快（4090 上 1080p 约 100 fps）。可微分。通过梯度下降拟合真实照片。一个场景在消费级 GPU 上 5-30 分钟即可完成拟合。

在此基础上 2023–2024 年的两项创新：
- **生成式高斯泼溅。** LGM、LRM、InstantMesh 等模型直接从一张或几张图像预测高斯云。
- **4D 高斯泼溅。** 高斯具有每帧偏移，用于动态场景。

### 多视图扩散

对预训练的图像扩散模型进行微调，使其能够从文本提示或单张图像生成同一物体的多个一致视图。Zero123（Liu 等人，2023）、MVDream（Shi 等人，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围的 4-16 个视图，然后通过高斯泼溅或 NeRF 提升到 3D。

### 文本到 3D 管线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | 文本 | 通过 SDS 的 NeRF | ~1 小时每个资产 |
| Magic3D | 文本 | 网格 + 纹理 | ~40 分钟 |
| Shap-E (OpenAI, 2023) | 文本 | 隐式 3D | ~1 分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | ~30 分钟 |
| LRM (Meta, 2023) | 图像 | 三平面 | ~5 秒 |
| InstantMesh (2024) | 图像 | 网格 | ~10 秒 |
| SV3D (Stability, 2024) | 图像 | 新视角 | ~2 分钟 |
| CAT3D (Google, 2024) | 1-64 张图像 | 3D NeRF | ~1 分钟 |
| TripoSR (2024) | 图像 | 网格 | ~1 秒 |
| Meshy 4 (2025) | 文本 + 图像 | PBR 网格 | ~30 秒 |
| Rodin Gen-1.5 (2025) | 文本 + 图像 | PBR 网格 | ~60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | 图像 | 网格 | ~30 秒 |

2025–2026 年的方向：直接文本到网格模型，并带有适用于游戏引擎的 PBR 材质。多视图扩散中间步骤仍然是处理通用物体的最佳方案。

### NeRF（背景介绍）

神经辐射场（NeRF，Mildenhall 等人，2020）。一个微小的 MLP 接收 `(x, y, z, 视角方向)` 并输出 `(颜色, 密度)`。通过沿射线积分进行渲染。在新视角合成质量上优于基于网格的方法，但渲染速度慢 100-1000 倍。对于大多数实时应用已被高斯泼溅取代，但在研究领域仍占主导地位。

## 动手实现

`code/main.py` 实现了一个玩具 2D “高斯泼溅”拟合：将合成目标图像（一个平滑渐变）表示为多个 2D 高斯泼溅的总和。通过梯度下降优化位置、颜色和协方差以匹配目标。你会看到两个核心操作：前向渲染（泼溅 + 阿尔法合成）和梯度下降拟合。

### 步骤 1：2D 高斯泼溅

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过求和泼溅进行渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D 高斯泼溅会按深度对高斯排序，并按顺序进行阿尔法合成。我们的 2D 玩具只是求和。

### 步骤 3：通过梯度下降进行拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视角不一致。** 如果你独立生成 4 个视图，且它们对物体结构不一致，则 3D 拟合结果会模糊。解决方法：使用带有共享注意力的多视图扩散。
- **背面幻觉。** 单张图像到 3D 必须臆造看不到的一面。质量差异极大。
- **高斯泼溅爆炸。** 无约束训练会导致高斯数量增长到 1000 万并过拟合。关键要使用致密化 + 修剪启发式规则（来自 3D-GS 原始论文）。
- **拓扑问题。** 从隐式场（SDF）生成的网格常带有孔洞或自交。在导出前应运行重网格化工具（例如 Blender 的体素重网格化）。
- **训练数据许可。** Objaverse 的许可混杂，商业使用因模型而异。

## 如何使用

| 任务 | 2026 推荐选择 |
|------|-----------|
| 从照片重建场景 | 高斯泼溅（3DGS、Gsplat、Scaniverse） |
| 文本到游戏用 3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图片到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 从少量图像合成新视角 | CAT3D、SV3D |
| 动态场景重建 | 4D 高斯泼溅 |
| 虚拟人像 / 穿衣人体 | Gaussian Avatar、HUGS |
| 研究 / 最新技术 | 上周发布的任何新方法 |

对于在游戏或电商管线中生产 3D 内容：Meshy 4 或 Rodin Gen-1.5 输出的 PBR 网格可直接导入 Unity / Unreal。

## 交付指南

保存 `outputs/skill-3d-pipeline.md`。技能需要一份 3D 需求文档（输入：文本 / 一张图像 / 少量图像；输出：网格 / 泼溅 / NeRF；用途：渲染 / 游戏 / VR），并输出：管线（多视图扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 运行 `code/main.py`，分别使用 4、16、64 个高斯。报告最终 MSE 与目标值的对比。
2. **中等。** 扩展至彩色高斯（RGB）。确认重建结果与目标颜色图案匹配。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获中重建一个真实物体。报告拟合时间以及留出视图上的最终 SSIM。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 3D 高斯泼溅 | “3DGS” | 场景表示为 3D 高斯云；可微阿尔法合成渲染。 |
| NeRF | “神经辐射场” | 在 3D 点输出颜色 + 密度的 MLP；通过射线积分渲染。 |
| 三平面 | “三个 2D 平面” | 将 3D 分解为三个沿坐标轴的 2D 特征网格；比体积表示更节省资源。 |
| SDS | “分数蒸馏采样” | 使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| 多视图扩散 | “一次生成多个视图” | 一次输出一批一致相机视角的扩散模型。 |
| PBR | “基于物理的渲染” | 包含反照率、粗糙度、金属度、法线通道的材质。 |
| 致密化 | “增长泼溅” | 3DGS 训练启发式：在高梯度区域分裂/克隆泼溅。 |

## 生产注意事项：3D 尚无通用运行时

与图像（潜在扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年还没有一个单一的主流运行时。生产决策树会根据表示方式产生分支：

- **NeRF / 三平面。** 推理过程是光线行进 + 每个样本一次 MLP 前向传播。512² 的渲染需要数百万次 MLP 前向传播。需要激进地批量处理射线样本；可应用 SDPA/xformers。
- **多视图扩散 + LRM 重建。** 两阶段管线。阶段 1（多视图 DiT）与第 07 课相似，是一个扩散服务器。阶段 2（LRM 转换器）是对视图的一次性前向传播。整体延迟特征是“扩散 + 一次推理”——据此选择每阶段的服务原语。
- **SDS / DreamFusion。** 是每资产的优化，而非推理。应该构建作业而非请求处理器。

对于大多数 2026 年的产品，正确答案是“按需运行多视图扩散模型，异步重建为 3DGS，然后提供 3DGS 供实时查看”。这样可以将工作负载清晰地分为 GPU 推理服务器（快速）和离线优化器（缓慢）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
