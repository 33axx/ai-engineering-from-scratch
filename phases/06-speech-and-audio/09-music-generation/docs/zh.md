# 音乐生成 — MusicGen, Stable Audio, Suno 与许可地震

> 2026 年音乐生成现状：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 引领开源。技术问题基本解决。法律问题（华纳音乐 5 亿美元和解、UMG 和解）在 2025–2026 年重塑了行业格局。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 6 · 02（声谱图），阶段 4 · 10（扩散模型）
**时长：** 约 75 分钟

## 问题

文本 → 30 秒至 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成**。诸如 "lo-fi hip-hop drums with warm keys" 之类的文本 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声 + 歌词）**。"Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件式 / 可控生成**。扩展现有片段、重新生成过渡段落、更换风格、音源分离或修补。Udio 的修补 + 音源分离是 2026 年的标杆功能。

## 概念

![音乐生成：token-LM 与扩散，2026 年模型全景图](../assets/music-generation.svg)

### 基于神经编解码器 token 的 token LM

Meta 的 **MusicGen**（2023，MIT 许可证）及其众多衍生模型：以文本/旋律嵌入为条件，自回归地预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。参数量 3 亿到 33 亿。强基线；超过 30 秒后表现不佳。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）将其扩展为带歌词条件的全歌生成。开源社区中最接近 Suno 的方案。

### 基于梅尔谱或潜在空间的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：对压缩音频进行潜在扩散。擅长循环片段、声音设计、环境纹理。不擅于结构完整的歌曲。

**AudioLDM / AudioLDM2**：通过类似 T2I 的潜在扩散实现文本到音频，泛化到音乐、音效、语音。

### 混合（生产级）—— Suno、Udio、Lyria

闭源。可能是 AR 编解码器 LM + 基于扩散的声码器，带有专门的人声/鼓/旋律头部。Suno v5（2026）是 ELO 1293 质量领先者。Udio v4 增加了修补 + 音源分离（贝斯、鼓、人声可单独下载）。

### 评估

- **FAD（弗雷歇音频距离）**。使用 VGGish 或 PANNs 特征，在生成音频与真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上 FAD 为 4.5；SOTA 约 3.0。
- **音乐性（主观）**。人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐**。提示与输出之间的 CLAP 分数。
- **音乐性伪影**。节拍错位的过渡、人声短语漂移、超过 30 秒后结构丢失。

## 2026 年模型概览

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability 非商业 |
| ACE-Step XL（2026年4月） | 4B | > 2 min | yes | Apache-2.0 |
| YuE | 7B | > 2 min | yes, 多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 min | yes, ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 min | yes + 分轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | yes | 商业 |
| MiniMax Music 2.5 | ? | 4 min | yes | 商业 API |

## 法律环境（2025–2026）

- **华纳音乐 vs Suno 和解**。5 亿美元。WMG 现在对 Suno 上的 AI 相似性、音乐权利和用户生成曲目拥有监督权。类似地，UMG 与 Udio 达成和解。
- **欧盟 AI 法案** + **加州 SB 942**：AI 生成的音乐必须披露。
- **Riffusion / MusicGen** 采用 MIT 许可证，没有合规负担，但也没有商用的人声。

安全交付的模式：

1. 仅生成器乐（使用 MusicGen、Stable Audio Open，输出适用 MIT/CC0 许可证）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），按生成量获取许可。
3. 在自有或授权曲库上训练（多数企业最终选择此方案）。
4. 为生成内容添加水印和元数据标签。

## 构建

### 第一步：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个尺寸：`small`（3 亿，速度快）、`medium`（15 亿）、`large`（33 亿）。对于“点子是否可行”，small 足够。

### 第二步：旋律条件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收一个色度图，保留旋律的同时更换音色。适用于“把这个旋律变成弦乐四重奏”。

### 第三步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级别的回归测试；不能替代人类听众。

### 第四步：集成到 LLM-音乐工作流中

结合第 7-8 课的理念：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用场景

| 目标 | 技术栈 |
|------|-------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4，配合明确许可 |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告音效 | MusicGen 以哼唱参考进行旋律条件生成 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍然存在的陷阱

- **版权洗白提示词**。“以 Taylor Swift 风格写一首歌”——商业 Suno/Udio 现在会过滤这些，开源模型不会。请自行添加过滤列表。
- **超过 30 秒后的重复/漂移**。AR 模型会循环。交叉淡化多个生成结果，或使用 ACE-Step 获得结构连贯性。
- **速度漂移**。模型会偏离 BPM。在提示词中使用 BPM 标签，并用 librosa 的 `beat_track` 进行后过滤。
- **人声清晰度**。Suno 效果出色；开源模型在歌词上往往模糊。如果歌词重要，请使用商业 API 或进行微调。
- **单声道输出**。开源模型生成单声道或假立体声。使用适当的立体声重建工具（如 ezst、Cartesia 的立体声扩散）升级。

## 交付

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、时长/结构计划以及披露元数据。

## 练习

1. **简单**。运行 `code/main.py`。它会生成一个“生成式”和弦进行 + 鼓点模式（ASCII 符号形式）——一个音乐生成的卡通演示。如果有 MIDI 渲染器，可以播放它。
2. **中等**。安装 `audiocraft`，用 MusicGen-small 针对 4 个不同风格的提示词生成 10 秒片段，并与参考风格集计算 FAD。
3. **困难**。使用 ACE-Step（或 MusicGen-melody），用不同音色提示词生成同一旋律的三种变体。计算提示词与输出的 CLAP 相似度以验证对齐度。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频 FID | 真实音频与生成音频嵌入分布之间的弗雷歇距离。 |
| 色度图 | 旋律音高 | 每帧 12 维向量；用于旋律条件输入。 |
| 分轨 | 乐器轨道 | 分离后的贝斯/鼓/人声/旋律 WAV 文件。 |
| 修补 | 重新生成某一段 | 掩蔽时间窗口；模型仅重新生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比式音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，被 MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) —— 声音设计的默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) —— 开源 4B 全歌生成器，2026 年 4 月。
- [Suno v5 平台文档](https://suno.com) —— 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) —— 用于音乐 + 音效的潜在扩散。
- [WMG-Suno 和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) —— 2025 年 11 月先例。
