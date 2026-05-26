# 音频分类——从MFCC上的k-NN到AST和BEATs

> 从“狗叫vs警笛”到“这是哪种语言”都属于音频分类。特征是梅尔频谱。架构每十年更新一次。评估指标始终是AUC、F1和每类召回率。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段6·02（频谱图和梅尔），阶段3·06（CNN），阶段5·08（文本的CNN和RNN）  
**时间：** ~75分钟  

## 问题

你拿到一段10秒的音频片段，想回答：“这是什么？”城市声音（警笛、电钻、狗叫）、语音指令（是/否/停）、语言识别（英语/西班牙语/阿拉伯语）、说话人情绪（愤怒/中性）或环境声音（室内/室外、嘈杂声）。所有这些都属于*音频分类*，而到了2026年，基线架构已经成熟：对数梅尔频谱→CNN或Transformer→softmax。

核心难点不在于网络，而在于数据。音频数据集存在严重的类别不平衡、强烈的领域偏移（干净 vs 嘈杂）以及标签噪声（谁能定义“城市嘈杂声”与“餐厅噪音”的区别？）。80%的问题出在数据筛选、增强和评估上，而不是把CNN换成Transformer。

## 概念

![音频分类阶梯：从MFCC上的k-NN到AST再到BEATs](../assets/audio-classification.svg)

**MFCC上的k-NN（1990年代的基线）。** 将每个片段的MFCC展平，与标记好的样本库计算余弦相似度，返回前K个的多数投票。在干净的小数据集（Speech Commands，ESC-50）上表现惊人，无需GPU。

**对数梅尔上的2D CNN（2015–2019）。** 将`(T, n_mels)`对数梅尔频谱视为图像，应用ResNet-18或VGG风格，在时间轴上做全局均值池化，然后softmax分类。截至2026年，大多数Kaggle比赛仍以此作为基线。

**音频频谱图Transformer，AST（2021–2024）。** 将对数梅尔频谱切块（例如16×16的块），加入位置嵌入，送入ViT。在AudioSet上达到监督学习的SOTA（mAP 0.485）。

**BEATs和WavLM-base（2024–2026）。** 基于数百万小时的自监督预训练，在你的任务上微调仅需原来监督数据的1%–10%。到2026年，这已成为非语音音频的默认起点。BEATs-iter3在AudioSet上比AST高出1–2个mAP，而计算量仅为后者的1/4。

**Whisper编码器作为冻结主干网络（2024）。** 取出Whisper的编码器，丢弃解码器，接上一个线性分类器。在语言识别和简单事件分类上接近SOTA，无需任何音频增强。这是“免费午餐”基线。

### 类别不平衡才是真正的挑战

ESC-50：50个类别，每类40个片段——平衡，容易。UrbanSound8K：10个类别，不平衡比例10:1。AudioSet：632个类别，长尾比例高达100000:1。有效技术包括：

- 训练时使用平衡采样（评估时不使用）。
- Mixup：将两个片段（及其标签）进行线性插值作为增强。
- SpecAugment：随机遮蔽时间和频率带。很简单，但至关重要。

### 评估

- 互斥多分类（Speech Commands）：top-1准确率，top-5准确率。
- 多标签多分类（AudioSet、UrbanSound风格）：平均精度均值（mAP）。
- 高度不平衡：每类召回率 + 宏平均F1。

2026年你应该知道的数字：

| 基准测试 | 基线 | 2026年SOTA | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR排行榜2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2结果 |

## 构建它

### 步骤1：特征化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤2：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：时间上的均值+方差为13维MFCC系数提供了一个26维的固定嵌入。瞬间运行。直到2017年，它在ESC-50上仍能击败当时的SOTA NN基线。

### 步骤3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤4：升级到对数梅尔上的CNN

在PyTorch中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300万参数。在单块RTX 4090上训练ESC-50约需10分钟，准确率80%以上。

### 步骤5：2026年默认方案——微调BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于BEATs，使用`microsoft/BEATs-base`通过`beats`库；transformers API的形状相同。

## 使用它

2026年的技术栈：

| 场景 | 起点 |
|-----------|-----------|
| 小数据集（<1000个片段） | MFCC均值的k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | 微调BEATs或AST |
| 大数据集（>100K） | 从头训练或微调Whisper编码器 |
| 实时、边缘设备 | 40-MFCC CNN，量化到int8（KWS风格） |
| 多标签（AudioSet） | BEATs-iter3配合BCE损失 + mixup + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107基线 |

决策规则：**从冻结的主干网络开始，而不是全新模型。** 微调BEATs头部在几小时内就能达到SOTA的95%，而不是几周。

## 发布它

保存为 `outputs/skill-classifier-designer.md`。针对给定的音频分类任务，选择架构、增强方法、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在4类合成数据集（不同音高的纯音）上训练k-NN MFCC基线。报告混淆矩阵。
2. **中等。** 将`summarize`替换为[均值、方差、偏度、峰度]。在相同合成数据集上，四矩池化能否胜过均值+方差？
3. **困难。** 使用 `torchaudio`，在ESC-50的折1上训练2D CNN。报告5折交叉验证准确率。加入SpecAugment（时间掩码=20，频率掩码=10）并报告差值。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频领域的ImageNet | Google的200万片段、632类、弱标签的YouTube数据集。 |
| ESC-50 | 小型分类基准 | 50类×40个环境音片段。 |
| AST | 音频频谱图Transformer | 基于对数梅尔切块的ViT；2021年SOTA。 |
| BEATs | 自监督音频 | 微软模型，iter3截至2026年领先AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将频谱中的随机时间和频率带置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024年间的标准架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年之后的默认模型。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 经久不衰的50类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632类YouTube分类体系，至今仍是黄金标准。
