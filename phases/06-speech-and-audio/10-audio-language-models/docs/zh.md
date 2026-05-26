# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型能够对语音、环境声音和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上媲美 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源与闭源之间的差距已基本消除——除了多音频任务，在该任务上所有模型都接近随机水平。

**类型：** 学习  
**语言：** Python  
**预备知识：** 第六阶段 · 04（ASR）、第十二阶段 · 03（视觉-语言模型）、第七阶段 · 10（音频 Transformer）  
**时间：** 约45分钟

## 问题

你有5秒的音频：狗叫声，有人喊"停下！"，然后一片寂静。有意义的问题涉及多个维度：

- **转录。** "说了什么？"——ASR 领域。
- **语义推理。** "这个人是否处于危险中？"——需要联合理解狗叫声、喊声和寂静。
- **音乐推理。** "旋律由哪些乐器演奏？"
- **长音频检索。** "在这段90分钟的讲座中，讲师在哪一部分解释了梯度下降？"

能够用一个提示回答所有这些问题的一个模型就是**音频-语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，而不仅仅是转录文本。

## 概念

![音频-语言模型：音频编码器 + 投影器 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

每个2026年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型自定义的编码器。
2. **投影器。** 线性层或 MLP，将音频编码器特征桥接到大语言模型的词元嵌入空间。
3. **大语言模型。** 基于 Llama / Qwen / Gemma 的解码器。接收交错文本 + 音频词元；生成文本。

训练：

- **阶段 1。** 冻结编码器和大语言模型；仅在 ASR / 字幕生成数据上训练投影器。
- **阶段 2。** 在指令跟随型音频任务（QA、推理、音乐理解）上进行全参数 / LoRA 微调。
- **阶段 3（可选）。** 语音输入/语音输出增加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 实现了这一点。

### 2026年模型地图

| 模型 | 骨干网络 | 音频编码器 | 输出模态 | 访问方式 |
|------|----------|------------|----------|----------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro (闭源) | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio (闭源) | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试真实情况（2026年）

**MMAU-Pro。** 1800个QA对，涵盖语音、声音、音乐和混合类型。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|------|------|------|------|------|--------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | 在 LongAudioBench 上 SOTA | — | — | — | — |

**多音频列对所有人来说都是致命打击。** 4选1多项选择的随机概率 = 25%；大多数模型得分在此附近。LALM 仍然难以比较两个片段。

### 2026年 LALM 的实用场景

- **呼叫中心录音合规审计。** "座席是否提到了所需的披露信息？"
- **无障碍访问。** 向听障用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言、威胁性语调以及背景上下文。
- **播客/会议章节划分。** 语义摘要，不仅仅是说话人轮次。
- **音乐目录分析。** "找出所有存在 B 段转调的音轨。"

### 它们目前（尚）不实用的场景

- 细致入微的音乐理论（低于和弦级别）。
- 长对话的说话人归属推理（超过10分钟性能下降）。
- 多音频比较（22-26% 仅略高于随机）。
- 实时流式推理（大多数为离线批量推理）。

## 动手构建

### 第一步：查询 Qwen2.5-Omni

```python
# examples/inference_qwen_omni.py
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

audio, sr = librosa.load("dog_bark.wav", sr=16000, mono=True)
inputs = processor(
    text="Describe the sound events in this clip.",
    audios=[audio],
    sampling_rate=sr,
    return_tensors="pt"
)
output_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(output_ids[0], skip_special_tokens=True))
```

### 第二步：投影器模式

```python
# examples/projector_pattern.py
import torch.nn as nn

class AudioProjector(nn.Module):
    """Simple 2-layer MLP that maps audio encoder features to LLM embedding dim."""
    def __init__(self, audio_dim: int, llm_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, audio_feats: torch.Tensor) -> torch.Tensor:
        return self.net(audio_feats)
```

就是这样。投影器通常是1-3个线性层。在 ASR 对（音频→转录）上训练它是第一阶段的预训练任务。

### 第三步：基准测试 MMAU / LongAudioBench

```python
# examples/benchmark_lalm.py
# Pseudocode for MMAU-Pro evaluation
from datasets import load_dataset

dataset = load_dataset("MMAU-Pro/MMAU-Pro", split="test")
model = load_your_lalm("Qwen/Qwen2.5-Omni-7B")

results = {"speech": [], "sound": [], "music": [], "multi": []}
for example in dataset:
    answer = model.query(audio=example["audio"], question=example["question"])
    category = example["category"]  # 'speech' | 'sound' | 'music' | 'multi-audio'
    results[category].append(answer == example["correct_answer"])

for cat, acc in results.items():
    print(f"{cat}: {sum(acc)/len(acc)*100:.1f}%")
```

按类别（语音/声音/音乐/多音频）分别报告结果。汇总数字会掩盖模型失败之处。

## 使用指南

| 任务 | 2026年推荐 |
|------|------------|
| 自由形式的音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源模型 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入/语音输出代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（擅长音乐的 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，并结合策略文档进行 RAG |

## 常见陷阱

- **过度信任多音频任务。** 如果你的任务需要判断"哪个片段包含X"，随机水平的表现是真实存在的。
- **长音频性能退化。** 超过10分钟，大多数模型的说话人归属能力崩溃。先进行说话人日志化（第6课），再汇总。
- **对静音产生幻觉。** 与 Whisper 风格相同的问题，由使用 Whisper 编码器的 LALM 继承。使用 VAD 进行门控。
- **基准测试的花样选取。** 厂商博客文章突出最佳类别。请自己运行 MMAU-Pro 多音频子集。

## 实战部署

保存为 `outputs/skill-alm-picker.md`。针对给定的音频理解任务，选择 LALM、基准测试子集和输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py`，查看一个玩具投影器模式以及虚假的 LALM 路由（音频嵌入、文本词元）→输出词元。
2. **中等。** 在100个 MMAU-Pro 语音项目上评估 Qwen2.5-Omni-7B 的得分。与论文报告的数字进行比较。
3. **困难。** 构建一个最小化的音频字幕生成基线：BEATs 编码器 + 2层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。在 Clotho-AQA 上与 SALMONN 进行比较。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------|----------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + 大语言模型解码器。 |
| 投影器 | 适配器 | 小型 MLP，将音频特征映射到大语言模型嵌入空间。 |
| MMAU | 基准测试 | 10000个音频QA对，涵盖语音、声音、音乐。 |
| MMAU-Pro | 更难的MMAU | 1800个多音频/重推理题目。 |
| LongAudioBench | 长格式评估 | 多分钟片段，带语义查询。 |
| 语音输入/语音输出 | 语音原生 | 模型接收语音并发出语音，无需经过文本绕路。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领导者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026年实时排名。
