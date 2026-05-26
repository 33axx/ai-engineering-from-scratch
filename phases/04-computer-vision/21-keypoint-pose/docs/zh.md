# 关键点检测与姿态估计

> 姿态是一组有序的关键点。关键点检测器是一个热图回归器。其余的都是记账工作。

**类型：** 构建  
**语言：** Python  
**前置要求：** 第四阶段第6课（检测），第四阶段第7课（U-Net）  
**时长：** ~45分钟  

## 学习目标

- 区分自顶向下和自底向上的姿态估计，并说明各自的使用场景
- 使用每个关键点对应的高斯目标热图回归K个关键点的热图，并在推理时提取关键点坐标
- 解释部位亲和场（PAFs）以及自底向上流水线如何将关键点关联成实例
- 使用MediaPipe Pose或MMPose进行生产级关键点估计，并理解其输出格式

## 问题描述

关键点任务有许多名称：人体姿态（17个身体关节）、人脸关键点（68或478个点）、手部关键点（21个点）、动物姿态、机器人物体姿态、医学解剖标志。它们都共享相同的结构：检测物体上的K个离散点，并输出它们的（x, y）坐标。

姿态估计是动作捕捉、健身应用、体育分析、手势控制、动画、AR试穿和机器人抓取的基础。2D姿态已经成熟；3D姿态（从单摄像头估计世界坐标系中的关节位置）是当前的研究前沿。

工程问题在于规模。单图像、单人姿态是一个20毫秒的问题。在拥挤场景中30帧每秒的多人体姿态则是一个不同的问题，需要不同的架构。

## 概念

### 自顶向下 vs 自底向上

```python
# 两种范式
# 自顶向下: 检测 -> 裁剪 -> 姿态估计（多次前向传播，取决于人数）
# 自底向上: 一次前向传播预测所有关键点 + 分组
```

- **自顶向下** — 先检测人，然后在每个裁剪区域上运行单人关键点模型。精度最高；计算量与人数成线性关系。
- **自底向上** — 一次前向传播预测所有关键点以及一个关联场；然后进行分组。无论人群大小，时间恒定。

自顶向下（HRNet, ViTPose）是精度领先者；自底向上（OpenPose, HigherHRNet）是拥挤场景中的吞吐量领先者。

### 热图回归

不直接回归（x, y），而是为每个关键点预测一个H x W的热图，其中以真实位置为中心放置高斯斑点。

```python
# 目标热图生成
def make_gaussian_heatmap(im_h, im_w, cx, cy, sigma=1.5):
    x = np.arange(im_w)
    y = np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    heat = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return heat
```

在推理时，每个热图的argmax就是预测的关键点位置。

为什么热图比直接回归效果好：网络的空间结构（卷积特征图）与空间输出自然对齐。高斯目标也能起到正则化作用——微小的定位误差会产生较小的损失，而非零损失。

### 亚像素定位

Argmax给出整数坐标。为了实现亚像素精度，通过拟合抛物线到argmax及其邻居进行细化，或者使用著名的偏移方向 `(dx, dy) = 0.25 * (heatmap[y, x+1] - heatmap[y, x-1], ...)`。

### 部位亲和场（PAFs）

OpenPose用于自底向上关联的技巧。对于每对连接的关键点（例如左肩到左肘），预测一个2通道的场，编码从一个指向另一个的单位向量。要将肩部与其肘部关联，沿着候选对之间的连线对PAF进行积分；积分最高的对即为匹配。

```python
# 沿线段对PAF进行积分
def paf_score(paf_map, joint_a, joint_b, samples=10):
    xs = np.linspace(joint_a[0], joint_b[0], samples)
    ys = np.linspace(joint_a[1], joint_b[1], samples)
    paf_x = paf_map[0][ys.astype(int), xs.astype(int)]
    paf_y = paf_map[1][ys.astype(int), xs.astype(int)]
    vec = np.array([joint_b[0]-joint_a[0], joint_b[1]-joint_a[1]])
    vec = vec / (np.linalg.norm(vec)+1e-6)
    dots = paf_x*vec[0] + paf_y*vec[1]
    return np.mean(dots)
```

优雅且可扩展到任意人群规模，无需逐人裁剪。

### COCO关键点

标准人体姿态数据集：每人17个关键点，使用PCK（正确关键点百分比）和OKS（物体关键点相似度）作为度量。OKS是关键点版本的IoU，是COCO mAP@OKS所报告的指标。

### 2D vs 3D

- **2D姿态** — 图像坐标；已达到生产质量（MediaPipe, HRNet, ViTPose）。
- **3D姿态** — 世界/相机坐标；仍然活跃研究。常见方法：
  - 使用小型MLP将2D预测提升到3D（VideoPose3D）。
  - 直接从图像进行3D回归（PyMAF, MHFormer）。
  - 多视角设置（CMU Panoptic）用于获取真实值。

## 构建部分

### 步骤1：高斯热图目标

```python
def make_gaussian_heatmap(im_h, im_w, cx, cy, sigma=1.5):
    x = np.arange(im_w)
    y = np.arange(im_h)
    xx, yy = np.meshgrid(x, y)
    heat = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return heat
```

每个关键点的热图沿着通道轴堆叠，形成完整的目标张量。

### 步骤2：微型关键点头

一个U-Net风格的模型，输出K个热图通道。

```python
import torch
import torch.nn as nn

class TinyPoseHead(nn.Module):
    def __init__(self, n_keypoints=17):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, n_keypoints, 1)
        )
    def forward(self, x):
        return self.model(x)
```

输入 `(N, 3, H, W)`，输出 `(N, K, H, W)`。损失函数为逐像素MSE，对标高斯目标。

### 步骤3：推理——提取关键点坐标

```python
def heatmap_to_coords(heatmaps):
    # heatmaps: (K, H, W)
    coords = []
    for k in range(heatmaps.shape[0]):
        h = heatmaps[k]
        y, x = np.unravel_index(h.argmax(), h.shape)
        coords.append((x, y))
    return np.array(coords)
```

推理时一行代码。若要亚像素细化，可在argmax周围插值。

### 步骤4：合成关键点数据集

简单：在白色画布上绘制四个点，学习预测它们。

```python
import numpy as np

def make_synthetic_sample():
    img = np.ones((64, 64, 3), dtype=np.float32)
    # 四个随机点
    pts = np.random.uniform(10, 54, (4, 2))
    for (x, y) in pts:
        # 在图像中绘制圆
        xx, yy = np.meshgrid(np.arange(64), np.arange(64))
        mask = (xx - x)**2 + (yy - y)**2 < 16
        img[mask] = [0, 0, 1]  # 蓝色点
    targets = np.stack([make_gaussian_heatmap(64, 64, pt[0], pt[1])
                        for pt in pts], axis=0)
    return img, targets
```

足够简单，微型模型在一分钟内就能学会。

### 步骤5：训练

```python
model = TinyPoseHead(n_keypoints=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for step in range(200):
    img, targets = make_synthetic_sample()
    img_t = torch.tensor(img).permute(2,0,1).unsqueeze(0)
    tgt_t = torch.tensor(targets).unsqueeze(0)
    pred = model(img_t)
    loss = loss_fn(pred, tgt_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step}: loss {loss.item():.4f}")
```

## 使用部分

- **MediaPipe Pose** — Google的生产级姿态估计器；提供WebGL和移动端运行时，延迟低于10毫秒。
- **MMPose**（OpenMMLab）— 全面的研究代码库；包含所有SOTA架构及预训练权重。
- **YOLOv8-pose** — 最快的实时多人姿态估计，一次前向传播完成。
- **transformers HumanDPT / PoseAnything** — 更新的视觉-语言方法，用于开放词汇姿态（任意物体，任意关键点集）。

## 交付部分

本课程产出：

- `outputs/prompt-pose-stack-picker.md` — 一个提示，根据延迟、人群规模和2D/3D需求选择MediaPipe / YOLOv8-pose / HRNet / ViTPose。
- `outputs/skill-heatmap-to-coords.md` — 一个技能，编写每个生产级姿态模型都使用的亚像素热图转坐标例程。

## 练习

1. **(简单)** 在合成的4点数据集上训练微型关键点模型。报告200步后预测关键点与真实关键点之间的平均L2误差。
2. **(中等)** 添加亚像素细化：给定argmax位置，从相邻像素沿x和y拟合一维抛物线。报告相比整数argmax的精度提升。
3. **(困难)** 构建一个包含2人的合成数据集，每张图像显示两个4关键点模式实例。训练一个带有PAF的自底向上流水线，预测哪个关键点属于哪个实例，并评估OKS。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 关键点 | “一个标志点” | 物体上的一个特定有序点（关节、角点、特征） |
| 姿态 | “骨架” | 属于一个实例的一组有序关键点 |
| 自顶向下 | “先检测后姿态” | 两阶段流水线：人员检测器 + 逐裁剪关键点模型；精度最高 |
| 自底向上 | “先姿态后分组” | 单次全关键点预测 + 分组；人群规模下时间恒定 |
| 热图 | “高斯目标” | 每个关键点一个H x W张量，峰值在真实位置；优先选择的回归目标 |
| PAF | “部位亲和场” | 编码肢体方向的2通道单位向量场；用于将关键点分组为实例 |
| OKS | “关键点IoU” | 物体关键点相似度；COCO姿态指标 |
| HRNet | “高分辨率网络” | 主导的自顶向下关键点架构；全程保持高分辨率特征 |

## 延伸阅读

- [OpenPose (Cao et al., 2017)](https://arxiv.org/abs/1812.08008) — 使用PAF的自底向上方法；仍是该方法的最佳论述
- [HRNet (Sun et al., 2019)](https://arxiv.org/abs/1902.09212) — 自顶向下的参考架构
- [ViTPose (Xu et al., 2022)](https://arxiv.org/abs/2204.12484) — 纯ViT作为姿态骨干网络；许多基准上的当前SOTA
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — 生产级实时姿态；2026年最快部署方案
