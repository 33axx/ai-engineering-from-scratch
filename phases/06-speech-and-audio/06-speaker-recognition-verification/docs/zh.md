# 说话人识别与验证

> ASR 询问“他们说了什么？”说话人识别则问“谁说的？”两者的数学原理相似——嵌入加余弦——但每个生产决策都取决于单一的 EER 数值。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段6·02（声谱图与梅尔），阶段5·22（嵌入模型）  
**时长：** 约45分钟

## 问题

用户说出一段口令。你需要知道：此人是否与声称的身份相符（*验证*，1:1），还是他是注册库中的第一个人（*识别*，1:N）？或者两者都不是——这是一个未知说话人（*开放集*）？

2018年前：GMM-UBM + i-vector。EER尚可，但对信道变化（手机 vs 笔记本电脑）和情感敏感。2018–2022年：x-vector（基于角度间隔训练的TDNN骨干）。2022年以后：ECAPA-TDNN和WavLM-large嵌入。到2026年，该领域主要由三个模型和一个指标主导。

该指标就是**EER**——等错误率。设定决策阈值，使得误接受率等于误拒绝率。交叉点即为EER。每篇论文、每个排行榜、每次采购招标中都用到它。

## 概念

![注册+验证流程：嵌入+余弦+EER](../assets/speaker-verification.svg)

**流程。** 注册：录制目标说话人5–30秒音频；计算固定维度的嵌入（ECAPA-TDNN为192维，WavLM-large为256维）。验证：获取测试语句的嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020年，2026年仍占主导）。** 强调信道注意力、传播与聚合的时延神经网络。包含带挤压激励的1D卷积块、多头注意力池化，后接线性层降至192维。在VoxCeleb 1+2（2700名说话人，110万条语句）上使用加性角度间隔损失（AAM-softmax）训练。

**WavLM-SV（2022年以后）。** 在预训练的WavLM-large SSL骨干上使用AAM损失进行微调。质量更高但速度较慢——模型大小超过300 MB（对比15 MB）。

**x-vector（基线）。** TDNN + 统计池化。经典方法，在CPU/边缘设备上仍可使用。

**AAM-softmax。** 在角度空间中对正确类别加上间隔 `m` 的标准softmax：`cos(θ + m)`。强制类别间角度分离。典型参数 `m=0.2`，缩放因子 `s=30`。

### 评分

- **余弦**：在注册嵌入与测试嵌入之间计算。基于阈值的决策。
- **PLDA（概率线性判别分析）**：将嵌入投影到潜在空间，使得同说话人与不同说话人具有封闭形式的似然比。在余弦基础上使用可降低EER 10–20%。2020年前为标准方法；现在仅在封闭集设置中使用。
- **评分归一化**：`S-norm` 或 `AS-norm`：针对一组冒名者的均值和标准差对每个分数进行归一化。跨领域评估时必不可少。

### 你应该知道的数值（2026年）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|-----------------|--------|----------------|
| x-vector（经典） | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 分割+嵌入 | 0.65% | 6 M | 100× RT |
| ReDimNet（2024） | 0.39% | 24 M | 100× RT |

### 说话人日志

在多说话人音频中“谁在什么时候说话”。流程：VAD → 分割 → 对每个片段进行嵌入 → 聚类（凝聚聚类或谱聚类） → 平滑边界。现代技术栈：`pyannote.audio` 3.1，它将说话人分割、嵌入和聚类封装在一个调用中。2026年AMI上的SOTA DER约为15%（2022年为23%）。

## 构建它

### 步骤1：从MFCC统计量中获取玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

远非SOTA——仅用于教学。`code/main.py` 将其作为合成说话人数据的概念验证。

### 步骤2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤3：从相似度对计算EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两者均需报告。

### 步骤4：使用SpeechBrain进行生产化

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 步骤5：使用pyannote进行说话人日志

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用它

2026年技术栈：

| 场景 | 选择 |
|------|------|
| 封闭集1:1验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证，云端 | WavLM-SV + AS-norm |
| 说话人日志（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式（关键词唤醒+注册） | Titanet-Small（NeMo） |

## 陷阱

- **信道不匹配。** 在VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。始终在目标信道上评估。
- **短语句。** 测试音频低于3秒时EER急剧下降。
- **带噪声的注册。** 一条噪声注册会毒化锚点。应使用≥3条干净样本并取平均。
- **固定阈值跨条件使用。** 始终在目标域的预留开发集上调优阈值。
- **对非归一化嵌入使用余弦。** 先进行L2归一化，否则幅度会主导结果。

## 交付它

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调优计划以及欺诈防范措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成“说话人”（不同音调轮廓），注册，在100对测试列表上计算EER。
2. **中等。** 使用SpeechBrain ECAPA处理30条VoxCeleb1语句（5名说话人 × 每条6句）。比较余弦与PLDA的EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的注册 → 说话人日志 → 验证流程。评估AMI开发集上的DER。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| EER | 核心指标 | 误接受率等于误拒绝率时的阈值。 |
| 验证 | 1:1 | “这是爱丽丝吗？” |
| 识别 | 1:N | “谁在说话？” |
| 开放集 | 可能包含未知 | 测试集可能含有未注册的说话人。 |
| 注册 | 登记 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角度间隔的softmax；迫使类别间分离。 |
| PLDA | 经典评分 | 概率线性判别分析；在嵌入之上进行似然比评分。 |
| DER | 说话人日志指标 | 说话人日志错误率——漏检+误报+混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典的深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026年主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于说话人识别和日志的SSL骨干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级说话人日志+嵌入技术栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 当前各模型EER排名。
