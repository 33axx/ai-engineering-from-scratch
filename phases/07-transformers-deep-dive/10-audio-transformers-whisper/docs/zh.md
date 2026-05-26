# 音频变换器 —— Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个卷积神经网络（ViT），它摄入梅尔频谱图并输出文本。

**类型：** 学习  
**语言：** Python  
**先修知识：** 第7阶段·05（完整Transformer），第7阶段·08（编码器-解码器），第7阶段·09（ViT）  
**时长：** ~45分钟

## 问题

在 Whisper（OpenAI，Radford 等人，2022）之前，最先进的自动语音识别（ASR）依赖 wav2vec 2.0 和 HuBERT——基于自监督特征提取器加微调的分类头。质量高，但数据管道昂贵，且领域适应性差。多语言语音识别需要为每种语系准备单独的模型。

Whisper 做了三个假设：

1. **使用所有数据进行训练。** 从互联网上爬取的 680,000 小时弱标注音频，涵盖 97 种语言。没有干净的学术语料库，没有音素标签。
2. **单一模型多任务。** 一个解码器通过任务令牌联合训练，完成转录、翻译、语音活动检测、语言识别和时间戳标注。
3. **标准的编码器-解码器 Transformer。** 编码器消费对数梅尔频谱图，解码器自回归生成文本令牌。没有声码器，没有 CTC，没有隐马尔可夫模型（HMM）。

结果：Whisper large-v3 在口音、噪声以及零干净标注数据的语言上表现出色。它是 2026 年所有开源语音助手和多数商业语音助手的默认语音前端。

## 概念

![Whisper 管道：音频 → 梅尔 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 加窗

音频以 16 kHz 采样。裁剪或补零至 30 秒。计算对数梅尔频谱图：80 个梅尔频带，10 毫秒步长 → 大约 3,000 帧 × 80 个特征。这就是 Whisper 所看到的“输入图像”。

### 步骤 2 — 卷积起始层

两个一维卷积层，核大小为 3，步长为 2，将 3,000 帧减少到 1,500。在不增加大量参数的情况下将序列长度减半。

### 步骤 3 — 编码器

一个 24 层（大模型）的 Transformer 编码器，处理 1,500 个时间步。使用正弦位置编码、自注意力、GELU 激活的前馈网络。输出 1,500 × 1,280 的隐藏状态。

### 步骤 4 — 解码器

一个 24 层的 Transformer 解码器。它从 BPE 词表中自回归地生成令牌，该词表是 GPT-2 词表的超集，并添加了一些音频相关的特殊令牌。

### 步骤 5 — 任务令牌

解码器的提示以控制令牌开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按照这种约定训练的。你通过前缀来控制任务。对应于 2026 年的指令微调，但应用于语音领域。

### 步骤 6 — 输出

束搜索（宽度 5）加上对数概率阈值。当缺少 `<|notimestamps|>` 令牌时，每 0.02 秒的音频预测一个时间戳。

### Whisper 模型大小

| 模型 | 参数量 | 层数 | d_model | 头数 | VRAM (fp16) |
|-------|--------|------|---------|------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）将解码器从 32 层缩减到 4 层，解码速度提升 8 倍，WER 仅退化不到 1 个百分点。正是这种解码速度的提升使得 Whisper-turbo 成为 2026 年实时语音智能体的默认选择。

### Whisper 不做什么

- 不做说话人分割（谁在说话）。这需要配合 pyannote 来实现。
- 原生不支持实时流式处理——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过语音活动检测（VAD）和重叠窗口来支持流式。
- 不支持超过 30 秒的上下文，需要外部切分。实践中效果不错，因为人类语音的转录通常不需要长距离上下文。

### 2026 年现状

| 任务 | 模型 | 备注 |
|------|------|------|
| 英文 ASR | Whisper-turbo, Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 毫秒延迟目标 |
| 文本转语音（TTS） | Piper, XTTS-v2, Kokoro | 编码器-解码器模式，但采用 Whisper 形状 |
| 音频+语言 | AudioLM, SeamlessM4T | 一个 Transformer 中同时包含文本令牌和音频令牌 |

## 构建它

参见 `code/main.py`。我们不训练 Whisper——我们构建对数梅尔频谱图管道和任务令牌提示格式化器。这些是你在生产环境中实际会接触的部分。

### 步骤 1：合成音频

生成一个 1 秒、440 Hz 的正弦波，采样率为 16 kHz。共 16,000 个样本。

### 步骤 2：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要 FFT。我们做一个简化的分帧加每帧能量版本，展示管道而不需要 `librosa` 库：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 = 25 ms，步长 = 10 ms。与 Whisper 的加窗方式匹配。每帧能量在教学中代表梅尔频带。

### 步骤 3：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示令牌

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制表面。一个 4 令牌的前缀。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快的、兼容 OpenAI 的版本：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用单一模型进行多语言 ASR。
- 对嘈杂、多样化的音频进行鲁棒转录。
- 研究 / 原型 ASR——最快的起点。

**何时选择其他方案：**

- 边缘设备上的超低延迟流式处理——同等质量下 Moonshine 优于 Whisper。
- 需要 <200 ms 的实时对话 AI——使用专用流式 ASR。
- 说话人分割——Whisper 不做这个；需要搭配 pyannote。

## 交付它

参见 `outputs/skill-asr-configurator.md`。该技能为新的语音应用选择 ASR 模型、解码参数和预处理管道。

## 练习

1. **简单。** 运行 `code/main.py`。确认一个 1 秒信号在 16 kHz、10 ms 步长下的帧数大约为 100 帧。对于 30 秒：大约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整对数梅尔频谱图。验证 80 个梅尔频带与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频切分为 10 秒窗口，重叠 2 秒，在每个块上运行 Whisper，合并转录文本。在 5 分钟的播客样本上测量单词错误率与单次推理的对比。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| 梅尔频谱图 | “音频图像” | 二维表示：一个轴是频率带宽，另一个轴是时间帧；每个格子是对数尺度的能量。 |
| 对数梅尔 | “Whisper 看到的” | 梅尔频谱图经过对数变换；近似人类对响度的感知。 |
| 帧 | “一个时间切片” | 25 ms 的样本窗口；以 10 ms 步长重叠。 |
| 任务令牌 | “语音的提示前缀” | 解码器提示中的特殊令牌，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测（VAD） | “找到语音” | 在 ASR 之前去除静音的门控；大幅降低成本。 |
| CTC | “连接主义时序分类” | 经典 ASR 损失函数，无需对齐训练；Whisper 不使用它。 |
| Whisper-turbo | “小解码器，完整编码器” | large-v3 编码器 + 4 层解码器；解码速度快 8 倍。 |
| Faster-whisper | “生产环境封装” | CTranslate2 重实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 进一步阅读

- [Radford 等人 (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 查看 Conv1D 起始层 + 编码器 + 解码器的完整实现（约 400 行）。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 第 5-6 步描述的束搜索加任务令牌逻辑；500 行，完全可读。
- [Baevski 等人 (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些场景下仍然是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产环境封装，比参考实现快 4 倍。
- [Jia 等人 (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好的 ASR，Whisper 形状但更小。
- [HuggingFace 博客 — “Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers”](https://huggingface.co/blog/fine-tune-whisper) — 标准微调教程，包括梅尔频谱图预处理和令牌时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与本课程的架构图对应。
