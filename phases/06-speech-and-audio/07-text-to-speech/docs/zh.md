# 文本转语音（TTS）——从 Tacotron 到 F5 与 Kokoro

> ASR 将语音反转为文本；TTS 将文本反转为语音。2026 年的技术栈分为三部分：文本 → 令牌，令牌 → 梅尔谱，梅尔谱 → 波形。每部分都有一个可在笔记本上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 02（语谱图与梅尔谱），阶段 5 · 09（Seq2Seq），阶段 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题

你有一个字符串："Please remind me to water the plants at 6 pm." 你需要一段 3 秒的音频片段，听起来自然、韵律正确（停顿、重音）、"plants" 的元音发音准确，并且用于实时语音助手时在 CPU 上运行时间不超过 300 毫秒。你还需要能切换语音、处理混合输入（"remind me at 6 pm, daijoubu?"），并且在处理人名时不出丑。

现代 TTS 流水线如下所示：

1. **文本前端。** 文本规范化（日期、数字、邮件），转换为音素或子词令牌，预测韵律特征。
2. **声学模型。** 文本 → 梅尔谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** 梅尔谱 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024 年后的神经编解码器声码器。

到 2026 年，随着端到端扩散和流匹配模型的出现，声学模型与声码器的界限变得模糊。但三部分的思维模型在调试时仍然有效。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro 对比示意图](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器输出梅尔帧。速度慢（自回归），长文本不稳定。仍被引用为基线。

**FastSpeech 2（2020）。** 非自回归。时长预测器输出每个音素得到多少梅尔帧。单次前向传播，比 Tacotron 快 10 倍。牺牲了一些自然度（单调对齐），但无处不在。

**VITS（2021）。** 联合训练编码器 + 基于流的时长 + HiFi-GAN 声码器，使用变分推理端到端。质量高，单一模型。2022–2024 年占主导地位的开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。韵律自然，使用 5 秒参考音频实现零样本语音克隆。2026 年开源 TTS 排行榜顶尖。335M 参数。

**Kokoro（2024）。** 小巧（82M），可在 CPU 上运行，实时使用的同类最佳英语 TTS。封闭词汇，仅英语，Apache 2.0 许可。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业最先进技术。ElevenLabs v2.5 的情绪标签（"[whispered]"、"[laughing]"）和角色语音在 2026 年主导有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近人类 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与自回归模型集成 | 离散令牌，比特高效 |

到 2026 年，大多数 "TTS" 模型是从文本到波形的端到端；梅尔谱图是内部表示。

### 评估

- **MOS（平均意见分）。** 1–5 分，众包。仍为黄金标准；速度慢得令人痛苦。
- **CMOS（比较 MOS）。** A 对 B 偏好。每标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜。
- **CER（字符错误率）通过 ASR。** 将 TTS 输出送入 Whisper，计算与输入文本的 CER。作为可懂度的代理。
- **SECS（说话人嵌入余弦相似度）。** 语音克隆质量。

2026 年 LibriTTS test-clean 上的数字：

| 模型 | UTMOS | CER（通过 Whisper） | 大小 |
|-------|-------|-------------------|------|
| 真实数据 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建

### 步骤 1：音素化输入

```python
import phonemizer
from phonemizer.backend import EspeakBackend

backend = EspeakBackend(language='en-us')
text = "Please remind me to water the plants at 6 pm."
phonemes = backend.phonemize([text], strip=True)[0]
print(phonemes)
# 'p l iː z  ɹ ɪ m aɪ n d  m iː  t uː  w ɔː t ɚ  ð ə  p l æ n t s  æ t  s ɪ k s  p iː  ɛ m .'
```

音素是通用桥梁。避免在低于 VITS 级别质量的模型中使用原始文本。

### 步骤 2：运行 Kokoro（2026 年 CPU 默认）

```python
from kokoro import KPipeline
from IPython.display import Audio

pipeline = KPipeline(lang_code='a')  # 'a' = American English
gen = pipeline(
    "Please remind me to water the plants at 6 pm.",
    voice='af_bella',  # female American voice
    speed=1.0
)
audio_data = list(gen)[0][0]  # first segment, first audio
Audio(audio_data, rate=24000)
```

离线运行，单文件，82M 参数。

### 步骤 3：使用 F5-TTS 进行语音克隆

```python
from f5_tts.model import F5TTS
from f5_tts.infer import infer_batch

model = F5TTS.from_pretrained("SWivid/F5-TTS")
ref_audio = "reference.wav"      # 5-second clip of target speaker
ref_text = "This is the reference transcript."
gen_text = "Please remind me to water the plants at 6 pm."
wav = infer_batch(model, [gen_text], ref_audio, [ref_text])[0]
```

传入一段 5 秒参考片段及其转录文本；F5 会克隆其韵律和音色。

### 步骤 4：HiFi-GAN 声码器（底层）

篇幅太大无法放入教程脚本，但大致结构为：

```python
class HiFiGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # multi‑receptive‑field fusion after upsampling
        self.ups = nn.ModuleList([...])   # transpose convs
        self.mrf = MRF()                  # multiple residual blocks
    def forward(self, mel):
        x = self.ups(mel)
        return self.mrf(x)
```

训练：对抗损失（短窗口判别器）+ 梅尔谱图重建损失 + 特征匹配损失。已商品化——使用来自 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：完整流水线（伪代码）

```python
text = "Please remind me to water the plants at 6 pm."
normalized = normalize_text(text)                        # dates, numbers, etc.
phonemes = phonemize(normalized)                          # espeak / gruut
mel = acoustic_model(phonemes, voice='af_bella')          # Kokoro internally
wav = vocoder(mel)                                        # HiFi‑GAN decoder
save_wav(wav, "reminder.wav")
```

## 使用

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 基于 5 秒参考语音克隆 | F5-TTS |
| 商业角色语音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 富有表现力/情绪标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领导者：**F5-TTS 用于质量，Kokoro 用于效率**。除非你是历史学家，否则不要碰 Tacotron。

## 陷阱

- **缺少文本规范化器。** "Dr. Smith" 读作 "Doctor" 还是 "Drive"？"2026" 读作 "twenty twenty six" 还是 "two zero two six"？在音素化之前进行规范化。
- **未登录专有名词。** "Ghumare" → "ghyu-mair"？为未知令牌配备一个备用的字素到音素模型。
- **削波。** 声码器输出很少削波，但推理时的梅尔缩放不匹配可能导致超出 ±1.0。始终使用 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；下游流水线期望 16 kHz → 重采样，否则会出现混叠。

## 交付

保存为 `outputs/skill-tts-designer.md`。针对给定的语音、延迟和语言目标设计一个 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从一个小型词汇表构建音素字典，估计每个音素的时长，并打印一个假的 "梅尔" 调度。
2. **中等。** 安装 Kokoro，使用 `af_bella` 和 `am_adam` 两种语音合成同一句话。比较音频时长和主观质量。
3. **困难。** 录制一段 5 秒的参考片段（你自己的声音）。使用 F5-TTS 克隆它。报告参考片段与克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Phoneme（音素） | 声音单位 | 抽象声音类别；英语中使用 39 个（ARPABet）。 |
| Duration predictor（时长预测器） | 每个音素持续多久 | 非自回归模型输出；每个音素的整数帧数。 |
| Vocoder（声码器） | 梅尔谱 → 波形 | 将梅尔谱映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年占主导地位。 |
| MOS（平均意见分） | 主观质量 | 来自人类评分者的 1–5 平均意见分。 |
| SECS（说话人嵌入余弦相似度） | 语音克隆指标 | 目标嵌入与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 82M 参数模型，Apache 2.0 许可。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好的英语 TTS。
