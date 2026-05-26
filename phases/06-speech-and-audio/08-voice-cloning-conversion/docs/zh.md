# 语音克隆与语音转换

> 语音克隆用他人的声音朗读你的文本。语音转换在保留你所说内容的前提下，将你的声音改写成他人的声音。两者都基于同一原理：将说话者身份与内容分离。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第 6 阶段 · 06（说话人识别），第 6 阶段 · 07（TTS）  
**时长：** 约 75 分钟  

## 问题

到 2026 年，一段 5 秒的音频片段足以用消费级 GPU 高质量克隆任何人的声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供了零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助声音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两项紧密相关的任务：

- **语音克隆（TTS 侧）：** 文本 + 5 秒参考语音 → 以该声音生成的音频。
- **语音转换（语音侧）：** 源音频（A 说 X 的内容）+ B 的参考语音 → B 说 X 的音频。

两者都将波形分解为（内容、说话者、韵律），然后将来自一个源的内容与来自另一个源的说话者重新组合。

2026 年你必须要满足的关键约束：**水印和同意门控制是欧盟（AI 法案，2026 年 8 月生效）和加利福尼亚州（AB 2905，2025 年生效）的法律要求。** 你的管道必须输出不可听的水印，并拒绝未经同意的克隆。

## 概念

![语音克隆 vs 转换：分解，交换说话者，重新组合](../assets/voice-cloning.svg)

**零样本克隆。** 传递一段 5 秒的片段给一个已在数千名说话者上训练过的模型。说话人编码器将片段映射到一个说话人嵌入；TTS 解码器基于该嵌入和文本进行条件生成。

使用于：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调。** 录制 5-30 分钟的目标语音。对基座模型进行 LoRA 微调约一小时。质量从“还行”跃升至“难以区分”。Coqui 和 ElevenLabs 都支持这种模式；社区将其用于 F5-TTS。

**语音转换（VC）。** 两个家族：

- **识别-合成。** 运行类 ASR 模型提取内容表示（如软音素后验概率 PPG），然后用目标说话人嵌入重新合成。对语言和口音鲁棒。用于 KNN-VC (2023)、Diff-HierVC (2023)。
- **解耦。** 训练一个自编码器，在瓶颈处的潜空间中将内容、说话者和韵律分离。在推理时交换说话人嵌入。质量较低但速度更快。用于 AutoVC (2019)、VITS-VC 变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为来自 SoundStream / EnCodec 的离散 token，训练一个大型自回归或流匹配模型处理编解码器 token。在短提示下的质量可与 ElevenLabs 媲美。

### 伦理部分，不是附加品

**水印。** PerTh 和 SilentCipher (2024) 将约 16-32 位的 ID 不可听地嵌入音频中。能抵抗重新编码、流媒体传输和常见编辑。生产级开源可用。

**同意门控制。** 必须为每个克隆输出配对一个可验证的同意记录。“我，Rohit，于 2026-04-22，授权此声音用于 X 目的。”存储在一个防篡改的日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛公布了针对 ElevenLabs、VALL-E 2 和 Bark 输出的最新检测器的 EER 为 0.8–2.3%。

### 数据（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对于大多数听众来说通常与目标声音难以区分。

## 构建

### 步骤 1：使用识别-合成进行分解（仅在 main.py 中演示代码）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上简单；实现工作量主要在 `tts_model` 和说话人编码器上。

### 步骤 2：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与音频完全匹配；不匹配会破坏对齐。

### 步骤 3：使用 KNN-VC 进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 提取源和目标池的逐帧嵌入，然后用源帧在池中的最近邻替换每一帧。非参数化，只需要一分钟的目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 位的负载，经过 MP3 重新编码和轻度噪声后仍可检测。

### 步骤 5：同意门控制

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用

2026 年的选择：

| 场景 | 选择 |
|-----------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业级生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话者微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转录不匹配。** F5-TTS 等要求参考文本与参考音频完全匹配，包括标点。
- **混响的参考。** 回声会破坏克隆效果。使用近距离麦克风录制干声。
- **情感不匹配。** 训练参考是“愉悦的”会产生所有内容都愉悦的克隆。参考情感应与目标用途匹配。
- **语言泄漏。** 克隆英语说话者然后要求模型说法语通常仍带口音；使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 自 2026 年 8 月起在欧盟法律上不可发布。

## 提交

保存为 `outputs/skill-voice-cloner.md`。设计一个带有同意门控制、水印和质量目标的克隆或转换管道。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算两个“说话者”在交换前后的余弦相似度来演示说话人嵌入交换。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考与克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 对 20 个克隆应用 SilentCipher 水印，经过 128 kbps MP3 编码解码，检测负载。报告位准确率。

## 关键术语

| 术语 | 大家常说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验概率图 | 逐帧的 ASR 后验概率，用作语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 用目标池中最近的帧替换每个源帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 基于 EnCodec/SoundStream token 的自回归模型。 |
| 水印 | 不可听的签名 | 嵌入在音频中的比特，能抵抗重新编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的语音转换。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 生产级 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器与合成器的军备竞赛，2026 年更新。
