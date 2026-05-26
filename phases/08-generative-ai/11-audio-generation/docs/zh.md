# 音频生成

> 音频是一个 16-48 kHz 的一维信号。一个五秒的片段包含 80-240k 个样本。没有 Transformer 能直接处理这么长的序列。2026 年所有生产级音频模型的解决方案都一样：一个神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，然后由 Transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 02（音频特征）、阶段 6 · 04（ASR）、阶段 8 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

三项音频生成任务：

1. **文本转语音（TTS）。** 给定文本，生成语音。干净的语音是窄带的，并且具有很强的语音结构——通过 token 上的 Transformer 可以很好地解决。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。分布要宽泛得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效/声音设计。** 给定提示，生成环境音或拟音。AudioGen、AudioLDM 2、Stable Audio Open。

这三者都运行在相同的基座上：神经音频编解码器 + token-AR 或扩散生成器。

## 概念

![音频生成：编解码器 token + Transformer 或扩散模型](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。一个卷积编码器将波形压缩为每时间步的向量；残差向量量化（Residual Vector Quantization, RVQ）将每个向量转换为 K 个码本索引的级联。解码器反向还原。24 kHz 音频在 2 kbps 下使用 8 个 RVQ 码本（75 Hz）= 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 其上的两个生成范式

**token 自回归（Token-autoregressive）。** 将 RVQ token 展开为一个序列，运行一个仅解码器 Transformer。MusicGen 使用“延迟并行”（delayed parallel）机制，通过每个流各自的偏移并行发出 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**潜在扩散（Latent diffusion）。** 将编解码器 token 打包为连续潜在变量，或用分类扩散对其建模。Stable Audio 2.5 在连续音频潜在变量上使用流匹配（flow matching）。AudioLDM 2 使用文本到梅尔到音频的扩散。

2024-2026 年趋势：流匹配在音乐方面胜出（推理更快、样本更干净），而 token-AR 在语音方面仍占主导地位，因为它天然是因果的且流式效果良好。

## 生产现状

| 系统 | 任务 | 主干网络 | 延迟 |
|------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首 token 约 300ms |
| OpenAI GPT-4o 音频 | 全双工语音 | 端到端多模态 AR | 约 200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜在变量上的流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | 在 Encodec 32kHz 上的 token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5s 片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 构建它

`code/main.py` 模拟核心思想：在合成的“音频 token”序列上训练一个极小的下一个 token Transformer，这些序列由两种不同的“风格”生成（风格 A 为交替的低和高 token，风格 B 为单调递增）。以风格为条件进行采样。

### 第一步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第二步：训练一个小型 token 预测器

一个以风格为条件的二元模型（bigram）风格预测器。重点是模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 第三步：条件采样

给定风格 token 和一个起始 token，从预测的分布中采样下一个 token。继续生成 20-40 个 token。

## 陷阱

- **编解码器质量限制了输出质量。** 如果编解码器不能忠实地表示一个声音，那么再好的生成器也无济于事。DAC 是目前公开的最佳选择。
- **RVQ 误差累积。** 每一层 RVQ 都对上一层的残差进行建模。第一层的错误会向下传播。在较高级别层使用温度 0 采样有所帮助。
- **音乐结构。** 30 秒的 token 在 75 Hz 下超过 20k 个 token。对 Transformer 来说很困难。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用较短的片段 + 交叉淡化。
- **边界处的伪影。** 生成的片段之间的交叉淡化需要仔细的重叠相加。
- **干净数据的胃口。** 音乐生成器需要成千上万小时的授权音乐。Suno / Udio 的 RIAA 诉讼（2024）将这一点推到了台面上。
- **语音克隆的伦理问题。** 一个 3 秒的样本加上一个文本提示就足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每个生产模型都需要滥用检测 + 退出名单。

## 使用建议

| 任务 | 2026 年技术栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（经同意验证） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音助手 | GPT-4o 实时或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付

保存为 `outputs/skill-audio-brief.md`。技能接受一个音频简报（任务、时长、风格、声音、许可），并输出：模型 + 托管方式、提示格式（风格标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议、评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 个 token 流，它们必须保持 1 步的偏移。训练一个联合预测器。
3. **困难。** 使用 HuggingFace Transformers 在本地运行 MusicGen-small。用三个不同的提示生成一个 10 秒的片段，对风格遵循度进行 A/B 测试。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| Codec（编解码器） | “神经压缩” | 音频的编码器/解码器；典型输出为 50-75 Hz 的 token。 |
| RVQ | “残差 VQ” | K 个量化器的级联；每个量化器对前一个的残差进行建模。 |
| Token | “一个编解码器符号” | 一个码本中的离散索引；通常为 1024 或 2048。 |
| Delayed parallel（延迟并行） | “偏移码本” | 发出 K 个 token 流，具有交错的偏移量，以减少序列长度。 |
| Flow matching（流匹配） | “2024 年音频的赢家” | 比扩散更直接的路径；采样速度更快。 |
| Voice prompt（语音提示） | “3 秒样本” | 引导克隆声音的说话人嵌入或 token 前缀。 |
| Mel spectrogram（梅尔频谱图） | “那幅图” | 对数幅度感知频谱图；被许多 TTS 系统使用。 |
| Vocoder（声码器） | “从梅尔到波形” | 将梅尔频谱图转换回音频的神经组件。 |

## 生产注记：音频是一个流式问题

音频是用户期望**在生成过程中就到达**的输出模态，而不是一次性全部到达。在生产中，这意味着 TPOT（每个输出 token 的时间）很重要，因为用户的聆听速度就是目标吞吐量——而不是他们的阅读速度。对于 16kHz 音频以约 75 tokens/秒（Encodec）进行 token 化，服务器必须为每个用户生成 ≥75 tokens/秒以保持播放流畅。

两个架构上的后果：

- **流匹配音频模型无法简单地进行流式生成。** Stable Audio 2.5 和 AudioCraft 2 一次性渲染固定长度的片段。要流式生成，你需要将片段分块并重叠边界——类似于滑动窗口扩散——相对于编解码器 AR 模型会增加 100-300ms 的延迟开销。

如果产品是“实时语音聊天”或“实时音乐延续”，请选择编解码器 AR 路径。如果是“提交后渲染一个 30 秒片段”，那么流匹配在质量和总延迟方面更胜一筹。

## 进一步阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年使用流匹配的文本转音乐。
