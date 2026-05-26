# 频谱图、梅尔尺度与音频特征

> 神经网络难以直接处理原始波形。它们处理频谱图，处理梅尔频谱图效果更佳。2026 年的每一个 ASR、TTS 和音频分类器，其成败都取决于这一单一的预处理选择。

**类型：** 动手构建  
**语言：** Python  
**前置条件：** 阶段 6 · 01（音频基础）  
**时长：** 约 45 分钟  

---

## 问题

取一段 10 秒 16 kHz 的片段。那是 160,000 个浮点数，全部在 `[-1, 1]` 范围内，与标签“狗叫”或“单词 cat”几乎完全不相关。原始波形包含了信息，但形式不利于模型提取。相隔 100 ms 的两个相同音素，其原始样本完全不同。

频谱图解决了这个问题。它压缩了人耳感知不到的时间细节（微秒级的抖动），同时保留了感知所关注的结构（在约 10–25 ms 的时间窗口内，哪些频率具有能量）。

梅尔频谱图更进一步。人类对音高的感知是对数性的：100 Hz 与 200 Hz 的“距离感”和 1000 Hz 与 2000 Hz 的“距离感”是相同的。梅尔尺度将频率轴进行了相应的扭曲。从 2010 年到 2026 年，梅尔尺度频谱图一直是语音机器学习中最重要的特征。

---

## 概念

![波形到 STFT 到梅尔频谱图到 MFCC 的阶梯示意图](../assets/mel-features.svg)

**STFT（短时傅里叶变换）** 将波形切割成重叠的帧（典型值：25 ms 窗长，10 ms 步长 = 在 16 kHz 下对应 400 个采样点 / 160 个采样点）。每帧乘以一个窗函数（Hann 窗是默认值；Hamming 窗有略微不同的折衷）。对每帧做 FFT。将幅度谱堆叠成一个 `(帧数, 频率bin数)` 的矩阵。这就是你的频谱图。

**对数幅度** 原始幅度跨越 5–6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产管线都使用对数幅度，而不是原始幅度。

**梅尔尺度** 频率 `f`（以 Hz 为单位）通过公式 `m = 2595 * log10(1 + f / 700)` 映射到梅尔值 `m`。该映射大致在 1 kHz 以下呈线性，在 1 kHz 以上呈对数。80 个梅尔 bin 覆盖 0–8 kHz 是 ASR 的标准输入。

**梅尔滤波器组** 一组在梅尔尺度上等间距分布的三角形滤波器。每个滤波器是相邻 FFT bin 的加权和。将 STFT 幅度与滤波器组矩阵相乘，即可一步得到梅尔频谱图。

**对数梅尔频谱图** `log(mel_spec + 1e-10)`。Whisper 的输入、Parakeet 的输入、SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC** 对对数梅尔频谱图应用 DCT（II 型），保留前 13 个系数。对特征进行去相关并进一步压缩。约在 2015 年之前一直是主导特征，之后 CNN/Transformer 直接处理原始对数梅尔频谱图赶超上来。目前仍用于说话人识别（x-vectors、ECAPA）。

**分辨率权衡** 更大的 FFT = 更好的频率分辨率，但更差的时间分辨率。音频 ML 默认值为 25 ms / 10 ms；音乐为 50 ms / 12.5 ms；瞬态检测（击鼓声、爆破音）为 5 ms / 2 ms。

---

## 动手构建

### 步骤 1：对波形进行分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒 16 kHz 的片段，`frame_len=400, hop=160` 会得到 998 帧。

### 步骤 2：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。消除因端点非零截断引起的频谱泄漏。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。这里的循环是教学性的；它在 `code/main.py` 中对短片段运行。

### 步骤 4：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 个梅尔，覆盖 0–8 kHz，`n_fft=400`，得到 `(80, 201)` 的矩阵。将 `(帧数, 201)` 的 STFT 幅度乘以其转置，得到 `(帧数, 80)` 的梅尔频谱图。

### 步骤 5：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的裁剪 + 归一化例程（参见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每一帧的对数梅尔应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码了总体能量）。

---

## 应用

2026 年的常见搭配：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 个对数梅尔，10 ms 步长，25 ms 窗长 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个梅尔，5–12 ms 步长以实现精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 个对数梅尔，10 ms 步长 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 个对数梅尔或原始波形自监督学习 |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（而非梅尔） |
| 关键词唤醒 | 40 个 MFCC（用于小型设备） |

经验法则：**如果你不是在做音乐，从 80 个对数梅尔开始。** 任何偏离都需要承担举证责任。

---

## 2026 年仍会出现的陷阱

- **梅尔数量不匹配** 训练时用 80 个梅尔，推理时用 128 个梅尔。静默失败。在两端记录特征形状。
- **上游采样率不匹配** 在 22.05 kHz 下计算的梅尔与在 16 kHz 下计算的不同。在特征化之前先固定采样率。
- **dB 与对数** Whisper 期望的是对数梅尔，不是 dB 梅尔。有些 Hugging Face 管线会自动检测；你的自定义代码不会。
- **归一化偏移** 训练时使用每段独立的归一化，推理时使用全局归一化。这种生产级 bug 会使 WER 翻倍。
- **填充导致的泄漏** 在片段末尾补零会产生平坦的频谱（在末尾帧）。应使用对称填充或复制填充。

---

## 存档

保存为 `outputs/skill-feature-extractor.md`。该技能会根据目标模型选择特征类型、梅尔数量、帧/步长和归一化方式。

---

## 练习

1. **简单** 运行 `code/main.py`。它会合成一个啁啾信号（频率从 200 Hz 扫到 4000 Hz），并打印每帧的最大梅尔 bin。可选：绘图并确认它符合扫频结果。
2. **中等** 用 `n_mels` 取 `{40, 80, 128}`、`frame_len` 取 `{200, 400, 800}` 再次运行。测量沿时间轴的尖峰带宽。哪组组合对啁啾信号的分辨率最好？
3. **困难** 实现 `power_to_db`，并在 AudioMNIST 上使用一个小 CNN 分类器比较 ASR 准确率：（a）原始对数梅尔，（b）使用 `ref=max` 的 dB 梅尔，（c）MFCC-13 + delta + delta-delta。报告 top-1 准确率。

---

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------|----------|
| 帧 | 一个切片 | 输入到一次 FFT 的 25 ms 波形块。 |
| 步长 | 步进 | 连续帧之间的采样点数；ASR 默认值为 10 ms。 |
| 窗 | Hann/Hamming 等 | 逐点乘法器，将帧边缘逐渐减至零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗的 FFT；产生时间 × 频率矩阵。 |
| 梅尔 | 扭曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 | 那个矩阵 | 将 STFT 投影到梅尔 bin 上的三角形滤波器。 |
| 对数梅尔 | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已标准化。 |
| MFCC | 老式特征 | 对数梅尔的 DCT；13 个系数，去相关。 |

---

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始梅尔尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram`、步长和窗的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet 和 Canary 模型的生产级管线。
