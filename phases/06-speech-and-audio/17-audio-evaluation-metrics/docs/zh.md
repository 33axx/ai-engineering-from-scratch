# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 与公开排行榜

> 无法衡量，就无法交付。本节课列举了 2026 年每项音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）以及说话人（EER）。还包括用于对比的排行榜。

**类型：** 学习  
**语言：** Python  
**前置知识：** 阶段 6 · 04、06、07、09、10；阶段 2 · 09（模型评估）  
**时长：** ~60 分钟

## 问题

每个音频任务都有多个指标，每个指标衡量不同的维度。使用错误的指标，就会发布一个在仪表盘上看起来很漂亮、但在生产环境中表现糟糕的模型。2026 年的规范列表如下：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首位令牌延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 语音克隆 | SECS（ECAPA 余弦相似度） | MOS · CER |
| 说话人确认 | EER | minDCF · FAR / FRR（工作点） |
| 说话人日志 | DER | JER · 说话人混淆度 |
| 音频分类 | top-1 · mAP | 宏平均 F1 · 每个类别的召回率 |
| 音乐生成 | FAD | CLAP · 听觉小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音 | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 — 指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。先转小写、去掉标点、标准化数字，再评分。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5%  = 人类水平的朗读语音。

**CER（字符错误率）。** 相同公式，字符级别。用于声调语言（普通话、粤语），因为词切分存在歧义。

**RTFx（逆实时因子）。** 每墙钟秒处理的音频秒数。越高越好。Parakeet-TDT 可达 3380×。Whisper-large-v3 约 30×。

**首位令牌延迟。** 从音频输入到第一个转录令牌的墙钟时间。对流式至关重要。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS（平均意见分数）。** 1-5 分的人工评分。黄金标准，但速度慢。每个样本收集 20+ 名听众，每个模型收集 100+ 个样本。

**UTMOS（2022-2026）。** 学习的 MOS 预测器。在标准基准上与人工 MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；真实音频：4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆。ECAPA 嵌入与参考音频及克隆输出之间的余弦相似度。> 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 对 TTS 输出运行 Whisper，计算与输入文本之间的 WER。用来捕捉可懂度退化。2026 SOTA：< 2% CER。

**TTFA（首次音频时间）。** 墙钟延迟。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### 语音克隆专用

**SECS + MOS + CER** 三元组。SECS 高但 MOS 低意味着音色正确但不自然；反之意味着声音自然但说话人错误。

### 说话人确认

**EER（等错误率）。** 误接受率等于误拒绝率时的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测成本函数）。** 在选定的工作点（通常 FAR=0.01）上的加权成本。比 EER 更贴合生产环境。

### 说话人日志

**DER（说话人日志错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏判语音 + 虚警语音 + 说话人混淆，各部分占总说话时间的比例。AMI 会议：DER 约 10-20% 是合理的。pyannote 3.1 + Precision-2 商业系统：在良好录音下 DER < 10%。

**JER（Jaccard 错误率）。** DER 的替代方案，对短片段偏差更为鲁棒。

### 音频分类

多标签：**mAP（平均准确率均值）** 在所有类别上。AudioSet：BEATs-iter3 的 mAP 为 0.548。

多类别互斥：**top-1、top-5 准确率**。Speech Commands v2：Audio-MAE 的 top-1 为 99.0%。

不平衡：**宏平均 F1** + **每个类别的召回率**。报告每个类别——聚合准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（弗雷歇音频距离）。** 真实音频与生成音频的 VGGish 嵌入分布之间的弗雷歇距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP 分数。** 使用 CLAP 嵌入的文本-音频对齐分数。> 0.3 = 合理对齐。

**听觉小组 MOS。** 对于消费级音乐，仍然是最终裁决。Suno v5 在 TTS Arena 上的 ELO 为 1293（来自配对人工偏好）。

### 音频语言模型基准

**MMAU（大规模多音频理解）。** 10k 个音频问答对。

**MMAU-Pro。** 1800 个困难条目，四个类别：语音 / 声音 / 音乐 / 多音频。四选一随机猜测正确率为 25%。Gemini 2.5 Pro 整体约 60%；多音频在所有模型中约 22%。

**LongAudioBench。** 多分钟片段，包含语义查询。Audio Flamingo Next 超过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 描述基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到首次可听响应的墙钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**输出上的 WER / MOS。**

**打断响应能力。** 从用户打断到助手静音的时间。目标 < 150 ms。

### 2026 年排行榜

| 排行榜 | 赛道 | 网址 |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语言 + 长格式 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，基于配对投票的 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （包含在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 动手实践

### 步骤 1：带标准化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人确认的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用它

将每次部署与一个固定的评估工具配对，该工具在每次模型更新时自动运行。三条基本规则：

1. **在评分前做标准化处理。** 转小写、去标点、扩展数字。报告所使用的标准化规则。
2. **报告分布，而非平均值。** 延迟报告 P50/P95/P99。分类报告每个类别的召回率。MMAU 报告每个类别。
3. **运行一个规范的公开基准。** 即使你的生产数据不同，报告 Open ASR / TTS Arena / MMAU 可以让评审者进行公平比较。

## 常见陷阱

- **UTMOS 外推。** 该模型基于 VCTK 风格的干净语音训练，对含噪、克隆或情感语音的评分效果较差。
- **MOS 小组偏差。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。如果风险较高，请支付领域小组费用。
- **FAD 依赖于参考集。** 在模型之间比较时，应使用相同的参考分布。
- **聚合 WER。** 整体 5% 的 WER 可能掩盖了口音语音上 30% 的 WER。请按人口统计切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上已接近天花板。构建反映你流量的内部保留集。

## 发布

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布版本选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD 近似值 / MMAU 近似值。
2. **中等。** 构建一个 TTS 往返 WER 工具。将你的 Kokoro 或 F5-TTS 输出通过 Whisper 运行。计算 50 个提示的 WER。标记 WER > 10% 的提示。
3. **困难。** 在第 10 课的 LALM 选择模型上，对 MMAU-Pro 语音 + 多音频子集（各 50 项）进行评分。报告每个类别的准确率，并与公开数据进行比较。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------|----------|
| WER | ASR 分数 | 标准化后单词级别的 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | 人工意见 | 1-5 分；20+ 名听众 × 100 个样本。 |
| UTMOS | 机器学习 MOS 预测器 | 学习模型；与人工 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考音频与克隆音频的 ECAPA 余弦相似度。 |
| EER | 说话人确认分数 | FAR = FRR 时的阈值。 |
| DER | 说话人日志分数 | (FA + Miss + Confusion) / 总长。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的弗雷歇距离。 |
| RTFx | 吞吐量 | 每墙钟秒处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带有标准化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督基准。
