# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 Transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上进行训练。一套架构，多项任务，在 99 种语言上表现稳健。2026 年的参考性 ASR。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 04（ASR），阶段 5 · 10（注意力机制），阶段 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一款作为商品提供的 ASR 模型：粘贴音频，获得文本，支持 99 种语言，对噪声鲁棒，可在笔记本电脑上运行。到 2024 年，OpenAI 已推出了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转写到语音助手再到 YouTube 字幕等一切任务的默认基线。

但 Whisper 并不是一个可以永远当做黑盒处理的流水线。领域迁移会对其造成损害——技术术语、说话人口音、专有名词、短片段、静音。你需要了解：

1. 其内部结构究竟是什么。
2. 如何正确地将分段、流式或长格式音频提供给 Whisper。
3. 何时微调以及如何进行。

## 概念

![Whisper 编码器-解码器、任务、分段推理、微调](../assets/whisper.svg)

**架构。** 标准 Transformer 编码器-解码器。

- 输入：30 秒对数梅尔频谱图，80 个梅尔滤波器，10 毫秒步长 → 3000 帧。更短的片段用零填充，更长的片段被切分。
- 编码器：卷积下采样（步长 2）+ `N` 个 Transformer 块。对于 Large-v3：32 层，1280 维，20 个注意力头。
- 解码器：`N` 个 Transformer 块，包含因果自注意力和对编码器输出的交叉注意力。大小与编码器相同。
- 输出：基于 51865 个 token 词汇表的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（从 32 层缩减），延迟降低 8 倍，词错误率增加小于 1%。

**提示格式。** Whisper 是一个多任务模型，通过解码器提示中的特殊 token 进行引导：

```python
prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
```

- `<|en|>` — 语言标签；强制决定翻译与转写行为。
- `<|transcribe|>` 或 `<|translate|>` — 从任意语言输入生成英文输出，或直接转写。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

提示就是让一个模型能完成多项任务的关键。将 `<|en|>` 改为 `<|fr|>`，它就会转写法语。

**30 秒窗口。** 一切都固定在 30 秒上。更长的片段需要切分；更短的片段会被填充。窗口本身不支持原生流式传输——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔归一化。** `(log_mel - mean) / std`，其中统计数据来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数 | 延迟 (A100) | 词错误率 (LibriSpeech-clean) |
|------|-------|-------------|------------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准工作流程：

1. 收集 10–100 小时目标领域的音频及对齐转录文本。
2. 使用带 `generate_with_loss` 回调的 `transformers.Seq2SeqTrainer`。
3. 参数高效：对注意力层的 `q_proj`、`k_proj`、`v_proj` 进行 LoRA，可将 GPU 内存减少 4 倍，词错误率损失小于 0.3。
4. 如果数据少于 10 小时，冻结编码器。仅调整解码器。
5. 使用 Whisper 自己的分词器和提示格式，切勿替换分词器。

社区成果：在 20 小时医学听写数据上微调 Medium，医学词汇上的词错误率从 12% 降至 4.5%。在 4 小时冰岛语上微调 Turbo，词错误率从 18% 降至 6%。

## 动手构建

### 步骤 1：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3")
result = model.transcribe("podcast.mp3", temperature=0.0, condition_on_previous_text=False, no_speech_threshold=0.6)
```

你应该总是指定的关键默认值：`temperature=0.0`（采样默认使用 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 步骤 2：切分长格式音频

```python
import whisperx
model = whisperx.load_model("large-v3", device="cuda")
audio = whisperx.load_audio("podcast.mp3")
result = whisperx.transcribe(model, audio, batch_size=16, vad_filter=True, vad_onset=0.5, vad_offset=0.363)
```

WhisperX 额外提供了 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 实现的词级对齐，(3) 通过 `pyannote.audio` 实现的说话人分离。这是 2026 年生产环境转录的主力工具。

### 步骤 3：使用 LoRA 微调

```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)
```

然后使用标准的 Trainer 循环。每 1000 步保存一次检查点。在保留集上用词错误率评估。

### 步骤 4：检查每层学到什么

```python
import torch
# 获取编码器第 10 层的注意力图
attention_map = model.encoder.layers[10].self_attn.attn_map.detach().cpu().numpy()
```

用热图可视化——你会看到解码器步骤扫描编码器帧时形成的对角线对齐。这条对角线就是 Whisper 对词时间戳的理解。

## 使用场景

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 通用英文，离线 | Large-v3-turbo 通过 `whisperx` |
| 移动端/边缘设备 | Whisper-Tiny 量化版（int8）或 Moonshine |
| 多语言长格式 | Large-v3 通过 `whisperx` + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原生版本快 4 倍，输出完全相同。

## 2026 年仍然存在的陷阱

- **静音时产生幻觉文本。** Whisper 在字幕上训练，会包含“Thanks for watching!”、“Subscribe!”、歌词等内容。调用前务必使用 VAD 门控。
- **`condition_on_previous_text` 级联。** 一次幻觉会污染后续窗口。除非需要跨片段的流畅性，否则设为 `False`。
- **短片段填充。** 一个 2 秒的片段填充到 30 秒后，可能在末尾静音处产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计数据。** 使用 librosa 而非 Whisper 自己的梅尔频谱会导致输出近乎随机。请使用 `whisper.audio.log_mel_spectrogram`。

## 交付

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计一个 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会 tokenize 一个 Whisper 风格的提示，计算解码后的形状预算，并输出一个 10 分钟音频的切分计划。
2. **中等。** 安装 `faster-whisper`，转写一个 10 分钟的播客，与人工转录文本比较词错误率。尝试 `language="auto"` 与强制 `language="en"` 的区别。
3. **困难。** 使用 HF `datasets`，选择一种 Whisper 表现不佳的语言（例如乌尔都语），使用 LoRA 在 2 小时数据上微调 Medium 2 个 epoch，并报告词错误率变化。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 30 秒窗口 | Whisper 的限制 | 硬性输入上限；长音频需要切分。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| 时间戳 token | 时间对齐 | 每个 0.02 秒的偏移在 51k 词汇表中对应一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，快 8 倍，词错误率退化小于 1%。 |
| WhisperX | 长格式封装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 在注意力层添加低秩适配器；训练约 0.3% 的参数。 |
| 幻觉 | 静默失败 | Whisper 从噪声/静音中产生流畅的英文。 |

## 进一步阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练方法。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长格式、词级对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 标准的 LoRA / 全微调教程。
