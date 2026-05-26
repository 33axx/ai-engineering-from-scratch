# 音频基础——波形、采样、傅里叶变换

> 波形是原始信号。频谱图是呈现形式。梅尔特征是对 ML 友好的形态。所有现代 ASR 和 TTS 流水线都遵循这个阶梯，第一步就是理解采样和傅里叶变换。

**类型:** 学习  
**语言:** Python  
**前置知识:** 第一阶 · 06（向量与矩阵）, 第一阶 · 14（概率分布）  
**时间:** 约 45 分钟

## 问题

麦克风产生一个压力随时间变化的信号。你的神经网络使用张量。在这两者之间有一整套约定，一旦违反，就会产生静默的 bug：模型训练正常但 WER 翻倍，或者 TTS 输出嘶嘶声，或者语音克隆系统记住了麦克风而不是说话者。

语音系统中的每个 bug 都可以追溯到以下三个问题之一：

1. 数据录制时使用的采样率是多少？模型期望的采样率又是多少？
2. 信号是否发生了混叠？
3. 你是在原始采样点上操作，还是在频域表示上操作？

把这三个问题搞对，第六阶的剩余部分就能迎刃而解。搞错了，即使是 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、DFT 与频率仓可视化](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数数组，值域 `[-1.0, 1.0]`。以采样点编号作为下标。要转换为秒，除以采样率：`t = n / sr`。一段 10 秒、16 kHz 的片段就是 160,000 个浮点数组成的数组。

**采样率 (sr)。** 每秒的采样点数。2026 年的常见采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话通信、传统 VoIP。奈奎斯特频率为 4 kHz，会丢失辅音。ASR 中应避免。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都使用 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理。** 采样率为 `sr` 的系统能够无歧义地表示最高频率为 `sr/2` 的信号。`sr/2` 这个边界被称为*奈奎斯特频率*。高于奈奎斯特频率的能量会发生*混叠*——折叠到较低的频率中——从而破坏信号。降采样前务必进行低通滤波。

**位深。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用的交换格式。24 位用于音乐，32 位浮点用于内部 DSP。`soundfile` 等库读取 int16 但对外暴露 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换。** 任何有限信号都可以表示为不同频率的正弦波之和。离散傅里叶变换 (DFT) 对 `N` 个采样点计算出 `N` 个复数系数——每个频率仓一个。第 `k` 个频率仓对应于频率 `k · sr / N` Hz。幅度表示该频率的能量大小，角度表示相位。

**FFT。** 快速傅里叶变换：当 `N` 是 2 的幂时，用于计算 DFT 的 `O(N log N)` 算法。每个音频库底层都使用 FFT。在 16 kHz 下，1024 点的 FFT 给出 512 个可用频率仓，跨越 0–8 kHz，分辨率为 15.6 Hz。

**分帧 + 加窗。** 我们不会对整个片段进行 FFT。而是将其切分成重叠的*帧*（通常 25 ms，步长 10 ms），对每一帧乘以窗函数（汉宁窗、汉明窗）以消除边缘不连续性，然后对每一帧进行 FFT。这就是短时傅里叶变换 (STFT)。第 02 课将从这里开始深入。

## 动手构建

### 第一步：读取音频片段并绘制波形

`code/main.py` 仅使用标准库中的 `wave` 模块，以保证示例无依赖。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第二步：从零开始合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

一个 440 Hz（音乐会 A 音）的正弦波，采样率 16 kHz，持续 1 秒，共计 16,000 个浮点数。使用 `wave.open(..., "wb")` 并以 16 位 PCM 编码写入。

### 第三步：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对于 `N=256` 验证正确性没问题，但对真实音频完全不可用。实际代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第四步：找出主导频率

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行应返回峰值出现在 `440 * N / sr` 仓。

### 第五步：演示混叠

以 10 kHz 采样率（奈奎斯特频率 = 5 kHz）采样一个 7 kHz 的正弦波。7 kHz 高于奈奎斯特频率，会折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是所有 ADC/DAC 都配备砖墙低通滤波器的原因。

## 实际应用

2026 年你实际会使用的技术栈：

| 任务 | 库 | 原因 |
|------|----|------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 的封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠滤波。 |
| STFT / 梅尔谱 | `torchaudio` 或 `librosa` | 支持 GPU；与 PyTorch 生态兼容。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | 命令行工具、快速、报告采样率/声道数/编码格式。 |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。如果你传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾结果。

## 交付物

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否与下游模型的期望一致，并在不一致时正确重采样。

## 练习

1. **简单。** 以 16 kHz 合成一段持续 1 秒的混合信号：220 Hz + 440 Hz + 880 Hz。运行 DFT。确认三个峰值出现在预期的仓位上。
2. **中等。** 以 48 kHz 录制一段 3 秒的语音。使用 `torchaudio.transforms.Resample`（带有抗混叠滤波）降采样到 16 kHz，然后再使用简单的抽取（每三个采样点取一个）降采样到 16 kHz。对两者进行 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和第三步的 DFT 从头构建 STFT。帧大小 400，步长 160，汉宁窗。使用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第 02 课中的频谱图。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| 采样率 | 每秒的采样点数 | ADC 测量信号的频率（单位 Hz）。 |
| 奈奎斯特频率 | 能表示的最大频率 | `sr/2`；高于它的能量会混叠回低频。 |
| 位深 | 每个采样的精度 | `int16` = 65,536 级；`float32` = `[-1, 1]` 内的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个采样点 → `N` 个复数频率系数。 |
| FFT | 快速的 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| 频率仓 | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层技术 | 随时间变化的加窗 FFT。 |
| 混叠 | 奇怪的频率幻影 | 高于奈奎斯特频率的能量镜像到较低的频率仓。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理的原始论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费、经典的 DSP 教材。
- [librosa 文档 — 音频入门](https://librosa.org/doc/latest/tutorial.html) — 附带代码的实践指南。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释为什么真实世界的音频不是干净的正弦波。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 十分钟澄清频率仓的直觉理解。
