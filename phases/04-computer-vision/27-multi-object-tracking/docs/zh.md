# 多目标跟踪与视频记忆

> 跟踪就是检测加关联。逐帧检测，并将当前帧的检测结果与上一帧的跟踪结果通过 ID 进行匹配。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 4 第 06 课（YOLO 检测），阶段 4 第 08 课（Mask R-CNN），阶段 4 第 24 课（SAM 3）
**时长：** ~60 分钟

## 学习目标

- 区分基于检测的跟踪与基于查询的跟踪，并指出相关算法家族（SORT、DeepSORT、ByteTrack、BoT-SORT、SAM 2 内存跟踪器、SAM 3.1 对象复用）
- 从零实现 IoU + 匈牙利分配，用于经典的基于检测的跟踪
- 解释 SAM 2 的内存库为何比基于 IoU 的关联更能处理遮挡
- 阅读三种跟踪指标（MOTA、IDF1、HOTA），并根据具体用例选择最合适的指标

## 问题

检测器告诉你单帧中目标的位置。跟踪器告诉你第 `t` 帧中的某个检测结果与第 `t-1` 帧中的哪个检测结果是同一对象。没有跟踪器，你就无法统计跨越某条线的对象数量、跟踪遮挡中的球，或知道“4 号车已在车道内行驶 8 秒”。

跟踪对于所有与视频相关的产品都至关重要：体育分析、监控、自动驾驶、医学视频分析、野生动物监测、文字计数。核心构建块是共享的：每帧检测器、运动模型（卡尔曼滤波器或更复杂的模型）、关联步骤（基于 IoU / 余弦 / 学习特征的匈牙利算法）以及跟踪生命周期（创建、更新、消亡）。

2026 年带来了两种新模式：**SAM 2 基于内存的跟踪**（用特征记忆替代运动模型关联）和 **SAM 3.1 对象复用**（为同一概念的多个实例共享内存）。本课首先介绍经典栈，然后介绍基于内存的方法。

## 概念

### 基于检测的跟踪

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

你在 2026 年遇到的所有跟踪器都是这个循环的变体。不同点在于：

- **SORT**（2016）：卡尔曼滤波器 + IoU 匈牙利算法。简单、快速、无外观模型。
- **DeepSORT**（2017）：SORT + 每个跟踪的 CNN 外观特征（ReID 嵌入）。能更好地处理交叉场景。
- **ByteTrack**（2021）：将低置信度检测作为第二阶段进行关联；无需外观特征但在 MOT17 上表现最佳。
- **BoT-SORT**（2022）：Byte + 摄像机运动补偿 + ReID。
- **StrongSORT / OC-SORT** — ByteTrack 的衍生版本，拥有更好的运动和外观模型。

### 卡尔曼滤波器（一句话概括）

卡尔曼滤波器为每个跟踪维护一个状态 `(x, y, w, h, dx, dy, dw, dh)` 及其协方差。在每一帧，使用恒定速度模型**预测**状态，然后用匹配到的检测结果进行**更新**。当预测不确定性高时，更新会更信任检测结果。这提供了平滑的轨迹，并能通过短时间遮挡（1-5 帧）继续跟踪。

所有经典跟踪器都在运动预测步骤中使用卡尔曼滤波器。

### 匈牙利算法

给定一个 `M x N` 的成本矩阵（跟踪 × 检测），找到最小化总成本的一对一分配。成本通常是 `1 - IoU(跟踪框, 检测框)` 或外观特征的负余弦相似度。运行时间为 O((M+N)^3)；对于 M、N 最多约 1000 的情况，通过 `scipy.optimize.linear_sum_assignment` 在 Python 中运行已足够快。

### ByteTrack 的关键思想

标准跟踪器会丢弃低置信度检测（< 0.5）。ByteTrack 将其保留为**第二阶段候选**：在将跟踪与高置信度检测匹配后，未匹配的跟踪尝试以稍宽松的 IoU 阈值与低置信度检测进行匹配。这能恢复短时遮挡、人群附近的 ID 切换。

### SAM 2 基于内存的跟踪

SAM 2 通过保留一个逐实例的时空特征**内存库**来处理视频。给定某一帧上的提示（点击、框、文本），它将实例编码到内存中。在后续帧上，内存与新帧的特征进行交叉注意力计算，解码器为新帧中的同一实例生成掩码。

没有卡尔曼滤波器，也没有匈牙利分配。关联隐含在内存-注意力操作中。

优点：
- 对大规模遮挡鲁棒（内存能在多帧中保持实例身份）。
- 结合 SAM 3 的文本提示可实现开放词汇跟踪。
- 无需独立的运动模型。

缺点：
- 对于多目标跟踪，速度慢于 ByteTrack。
- 内存库会增长，限制了上下文窗口。

### SAM 3.1 对象复用

之前的 SAM 2 / SAM 3 跟踪为每个实例维护独立的内存库。对于 50 个对象，就需要 50 个内存库。对象复用（2026 年 3 月）将它们压缩为一个共享内存，并带有**逐实例查询 token**。成本随实例数量呈次线性增长。

复用已成为 2026 年人群跟踪的新默认选择：演唱会人群、仓库工人、交通路口。

### 需要了解的三个指标

- **MOTA（多目标跟踪准确度）** — 1 - (FN + FP + ID 切换) / GT。按错误类型加权；一个混合了检测和关联失败的单一指标。
- **IDF1（ID F1）** — ID 精确率和召回率的调和平均值。专门关注每个真实跟踪在其生命周期内保持 ID 的好坏程度。对于对 ID 切换敏感的任务，优于 MOTA。
- **HOTA（高阶跟踪准确度）** — 分解为检测准确度（DetA）和关联准确度（AssA）。自 2020 年以来的社区标准；最为全面。

对于监控（谁是谁）：报告 IDF1。对于体育分析（传球计数）：HOTA。对于通用学术比较：HOTA。

## 动手构建

### 步骤 1：基于 IoU 的成本矩阵

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### 步骤 2：最小化 SORT 风格跟踪器

为简洁起见，省略了固定的恒定速度卡尔曼滤波器——我们在这里使用简单的 IoU 关联；实际生产环境中卡尔曼预测是必需的。`sort` Python 包提供了完整版本。

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 行代码。接收每帧检测结果，返回每帧跟踪 ID。实际系统会添加卡尔曼预测、ByteTrack 的第二阶段重新匹配以及外观特征。

### 步骤 3：合成轨迹测试

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

三个沿直线移动的对象应在所有 20 帧中保持其 ID。

### 步骤 4：ID 切换指标

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

这是一个简化的 IDF1 相邻指标：统计一个真实对象改变其分配到的预测跟踪 ID 的次数。真正的 MOTA / IDF1 / HOTA 工具位于 `py-motmetrics` 和 `TrackEval` 中。

## 实际使用

2026 年生产环境中的跟踪器：

- `ultralytics` — YOLOv8 + 内置 ByteTrack / BoT-SORT。`results = model.track(source, tracker="bytetrack.yaml")`。这是默认选项。
- `supervision`（Roboflow）— ByteTrack 封装器以及标注工具。
- SAM 2 / SAM 3.1 — 通过 `processor.track()` 实现基于内存的跟踪。
- 自定义栈：检测器（YOLOv8 / RT-DETR）+ `sort-tracker` / `OC-SORT` / `StrongSORT`。

选用建议：

- 行人 / 车辆 / 箱子，帧率 30+ fps：**ByteTrack 配合 ultralytics**。
- 人群中同一类别的多个实例：**SAM 3.1 对象复用**。
- 严重遮挡但外观可区分：**DeepSORT / StrongSORT**（ReID 特征）。
- 体育 / 复杂交互：**BoT-SORT** 或学习型跟踪器（MOTRv3）。

## 输出成果

本课将产生：

- `outputs/prompt-tracker-picker.md` — 根据场景类型、遮挡模式和延迟预算选择 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1。
- `outputs/skill-mot-evaluator.md` — 编写一个完整的评估工具，用于针对真实跟踪计算 MOTA / IDF1 / HOTA。

## 练习

1. **(简单)** 运行上述合成跟踪器，分别使用 3、10 和 30 个对象。在每种情况下报告 ID 切换次数。指出简单的纯 IoU 关联在何处开始失效。
2. **(中等)** 在关联之前添加一个恒定速度卡尔曼预测步骤。证明短时（2-3 帧）遮挡不再导致 ID 切换。
3. **(困难)** 集成 SAM 2 基于内存的跟踪器（通过 `transformers`）作为替代跟踪后端。在 30 秒的人群片段上同时运行 SimpleTracker 和 SAM 2，比较 ID 切换次数，并手动标记 5 个显著人物的真实 ID。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 基于检测的跟踪 | "先检测再关联" | 逐帧检测器 + 基于 IoU / 外观的匈牙利分配 |
| 卡尔曼滤波器 | "运动预测" | 线性动力学 + 协方差，实现平滑的轨迹预测和遮挡处理 |
| 匈牙利算法 | "最优分配" | 解决最小成本二分图匹配问题；`scipy.optimize.linear_sum_assignment` |
| ByteTrack | "低置信度第二遍" | 将未匹配的跟踪与低置信度检测重新匹配，恢复短时遮挡 |
| DeepSORT | "SORT + 外观" | 添加 ReID 特征用于跨帧匹配；更利于 ID 保持 |
| 内存库 | "SAM 2 技巧" | 跨帧存储的逐实例时空特征；交叉注意力替代显式关联 |
| 对象复用 | "SAM 3.1 共享内存" | 单个共享内存，带有逐实例查询，用于快速多目标跟踪 |
| HOTA | "现代跟踪指标" | 分解为检测准确度和关联准确度；社区标准 |

## 扩展阅读

- [SORT (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763) — 最简化的基于检测的跟踪论文
- [DeepSORT (Wojke et al., 2017)](https://arxiv.org/abs/1703.07402) — 添加外观特征
- [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864) — 低置信度第二遍
- [BoT-SORT (Aharon et al., 2022)](https://arxiv.org/abs/2206.14651) — 摄像机运动补偿
- [HOTA (Luiten et al., 2020)](https://arxiv.org/abs/2009.07736) — 分解式跟踪指标
- [SAM 2 视频分割 (Meta, 2024)](https://ai.meta.com/sam2/) — 基于内存的跟踪器
- [SAM 3.1 对象复用 (Meta, March 2026)](https://ai.meta.com/blog/segment-anything-model-3/)
