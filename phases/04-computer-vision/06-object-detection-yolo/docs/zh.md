# 目标检测——从零实现 YOLO

> 检测 = 分类 + 回归，在特征图的每个位置执行，然后通过非极大值抑制进行清理。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段4 第03课（CNN），阶段4 第04课（图像分类），阶段4 第05课（迁移学习）  
**时长：** ~75 分钟

## 学习目标

- 解释网格与锚框设计如何将检测转化为密集预测问题，并说明输出张量中每个数字的含义
- 计算两个框之间的交并比，并从零实现非极大值抑制
- 在预训练骨干网络之上构建一个极简的 YOLO 风格检测头，包括分类损失、目标性损失和框回归损失
- 阅读一行检测指标（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95），并决定下一步应该调整哪个旋钮

## 问题

分类说的是“这张图像是一只狗”。检测说的是“在像素 (112, 40, 280, 210) 处有一只狗，在 (400, 180, 560, 310) 处有一只猫，画面中没有其他物体”。这一个结构上的变化——预测可变数量的带标签框，而不是每张图像一个标签——是每一个自动驾驶系统、每一个安防产品、每一个文档布局解析器和每一条工厂视觉流水线所依赖的基础。

检测也是视觉领域中所有工程权衡同时爆发的地方。你需要精确的框（回归头），需要每个框的正确类别（分类头），需要模型知道什么时候没有东西需要检测（目标性分数），并且需要每个真实物体只产生一个预测（非极大值抑制）。缺少任何一个环节，流水线要么漏检物体，要么报告幻觉框，要么将同一个物体以略微不同的位置预测十几次。

YOLO（You Only Look Once，Redmon 等人，2016）是一种通过单次卷积网络前向传播实现这一切并达到实时运行的设计，其结构决策至今仍是现代检测器（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨干。学习核心思想后，每一个变体都只是相同部件的重新排列。

## 概念

### 检测作为密集预测

分类器每张图像输出 C 个数字。YOLO 风格的检测器每张图像输出 `(S x S x (5 + C))` 个数字，其中 S 是空间网格的尺寸。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

每个 `S * S` 的网格单元预测 B 个框。对于每个框：

- 4 个数字描述几何形状：`tx, ty, tw, th`
- 1 个数字是目标性分数：“这个网格单元中心是否有物体？”
- C 个数字是类别概率

每个网格单元总计：`B * (5 + C)`。对于 VOC（S=13, B=2, C=20），每个单元就是 50 个数字。

### 为什么用网格和锚框

简单的回归会为每个物体预测 `(x, y, w, h)` 作为绝对坐标。这对卷积网络来说很困难，因为平移图像不应该使所有预测平移相同的量——每个物体在空间上是有锚点的。网格通过将每个真实框分配给它中心所在的网格单元来解决这个问题；只有那个单元负责该物体。

锚框解决了第二个问题。一个 3x3 的卷积很难从一个 16 像素感受野的特征单元回归出一个 500 像素宽的框。相反，我们为每个单元预定义 B 个先验框形状（锚框），并预测每个锚框的小量偏移。模型学会选择正确的锚框并微调，而不是从零回归。

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代检测器通常使用 FPN，每个分辨率对应不同的锚框集——浅层高分辨率图上用小锚框，深层低分辨率图上用大锚框。思路相同，只是多了尺度。

### 解码预测

原始的 `tx, ty, tw, th` 不是框坐标；它们是回归目标，需要在绘图前进行变换：

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid` 将中心偏移量限制在单元内。`exp` 允许宽度相对于锚框自由缩放而不会出现符号翻转。`stride` 将网格坐标缩放回像素值。自 v2 以来，这个解码步骤在每个 YOLO 版本中都是一样的。

### IoU

检测中两个框之间的通用相似性度量：

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1 表示完全相同；IoU = 0 表示无重叠。预测框与真实框之间的 IoU 决定了一个预测是否算作真正例（通常 IoU >= 0.5）。两个预测框之间的 IoU 是 NMS 用于去重的依据。

### 非极大值抑制

在相邻锚框上训练的卷积网络通常会对同一个物体预测多个重叠框。NMS 保留置信度最高的预测，并删除任何与它 IoU 超过阈值的其他预测。

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

典型阈值：目标检测中为 0.45。最近的检测器用 soft-NMS、DIoU-NMS 或直接学习抑制（RT-DETR）替代标准 NMS，但结构目的相同。

### 损失函数

YOLO 损失由三个带权重的损失相加：

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含物体的网格单元才对框回归和分类损失有贡献。不包含物体的单元只贡献目标性损失（教会模型保持安静）。`lambda_noobj` 通常很小（约 0.5），因为绝大多数单元是空的，否则会主导总损失。

现代变体将 MSE 框损失替换为 CIoU / DIoU（直接优化 IoU），对类别不平衡使用 Focal Loss，并用质量 Focal Loss 平衡目标性。三部分结构不变。

### 检测指标

准确率不能直接迁移到检测任务。以下四个数值更有效：

- **Precision@IoU=0.5** —— 在被视为正例的预测中，有多少实际上是正确的。
- **Recall@IoU=0.5** —— 在真实物体中，我们找到了多少。
- **AP@0.5** —— 在 IoU 阈值 0.5 下的精确率-召回率曲线面积；每个类别一个数字。
- **mAP@0.5:0.95** —— 在 IoU 阈值 0.5, 0.55, ..., 0.95 上 AP 的平均值。COCO 指标；最严格也最有信息量。

报告所有四个数字。如果一个检测器在 mAP@0.5 上很强但在 mAP@0.5:0.95 上较弱，说明定位粗略但不精确；修复方法：使用更好的框回归损失。如果检测器精确率高但召回率低，说明过于保守；降低置信度阈值或增加目标性权重。

## 构建它

### 步骤 1：IoU

这是整个课程的主力函数。适用于两组框，格式为 `(x1, y1, x2, y2)`。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回一个 `(N_a, N_b)` 的成对 IoU 矩阵。通过将其中一个形状设为 `(1, 4)` 来与单个真实框进行比较。

### 步骤 2：非极大值抑制

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性算法，排序带来 `O(N log N)` 复杂度，并且与 `torchvision.ops.nms` 在相同输入上的行为一致。

### 步骤 3：框编码与解码

在像素坐标与网络实际回归的 `(tx, ty, tw, th)` 目标之间进行转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：先编码一个框再解码——应该能恢复到接近原始的值（但由于 sigmoid 逆变换在 `tx` 不在 sigmoid 后范围内时不是完全可逆的，会有细微差异）。

### 步骤 4：极简 YOLO 检测头

在特征图上用一个 1x1 卷积，然后重塑为 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一个维度保存 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### 步骤 5：真实框分配

对于每个真实框，决定由哪个 `(网格单元, 锚框)` 负责。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

锚框选择依据是“与真实框的最佳形状 IoU”——一个廉价的近似，与 YOLOv2/v3 的分配方式一致。v5 及以后使用更复杂的策略（任务对齐匹配、动态 k），但对相同思想进行了改进。

### 步骤 6：三个损失

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

每个 YOLO 教程要么硬编码要么搜索五个超参数。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 源自原始 YOLOv1 论文，作为合理的默认值仍然有效。

### 步骤 7：推理流水线

解码检测头的原始输出，应用 sigmoid/exp，根据目标性得分进行阈值筛选，然后执行 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整的评估路径：检测头 -> 解码 -> 阈值筛选 -> NMS。

## 使用它

`torchvision.models.detection` 提供了具有相同概念结构的现成检测器。加载预训练模型只需三行代码。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推理流水线，`ultralytics`（YOLOv8/v9）是标准选择：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。该模型内部处理解码和 NMS，返回与你上面构建的相同 `boxes / scores / labels` 三元组。

## 交付产物

本课程产出：

- `outputs/prompt-detection-metric-reader.md` —— 一个提示词，能够将一行 `precision, recall, AP, mAP@0.5:0.95` 转化为一行诊断结果和最有用的下一个实验建议。
- `outputs/skill-anchor-designer.md` —— 一个技能，给定一个由真实框组成的数据集，在 `(w, h)` 上运行 k-means，返回每个 FPN 级别的锚框集，以及选择正确锚框数量所需的覆盖统计量。

## 练习

1. **（简单）** 实现 `box_iou` 并在 1,000 个随机框对上与 `torchvision.ops.box_iou` 进行比较。验证最大绝对差值低于 `1e-6`。
2. **（中等）** 将 `yolo_loss` 移植为使用 CIoU 框损失代替 MSE 的版本。在一个 100 张图像的合成数据集上展示，在相同 epochs 数下，CIoU 收敛到比 MSE 更好的最终 mAP@0.5:0.95。
3. **（困难）** 实现多尺度推理：将同一张图像以三种分辨率输入模型，合并所有框预测，最后执行一次 NMS。在留出集上测量与单尺度推理相比的 mAP 提升。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|------------|----------|
| Anchor | “框先验” | 每个网格单元中预定义的框形状，网络预测相对于它的偏移量而非绝对坐标 |
| IoU | “重叠度” | 两个框的交并比；检测中的通用相似性度量 |
| NMS | “去重” | 贪心算法，保留最高得分的预测并删除与之重叠超过阈值的其他预测 |
| Objectness | “这里是否有东西” | 每个锚框、每个网格单元的标量，预测该单元中心是否有物体 |
| Grid stride | “下采样因子” | 每个网格单元对应的像素数；416 像素输入配合 13 网格检测头，步幅为 32 |
| mAP | “平均精度均值” | 精确率-召回率曲线下面积的均值，按类别（以及 COCO 标准下按 IoU 阈值）平均 |
| AP@0.5 | “PASCAL VOC AP” | IoU 阈值为 0.5 时的平均精度；指标中较宽松的版本 |
| mAP@0.5:0.95 | “COCO AP” | 在 IoU 阈值 0.5 到 0.95（步长 0.05）上 AP 的平均值；严格版本，是目前社区的标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) —— 奠基论文；此后所有的 YOLO 都是对此结构的改进
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) —— 引入多尺度 FPN 风格检测头的论文；仍然是最清晰的图表
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com) —— 当前的通用参考；涵盖数据集格式、数据增强、训练配方
- [目标检测图解指南 (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) —— 最通俗易懂的全景图；对理解 DETR、RetinaNet、FCOS 和 YOLO 之间的关系非常宝贵
