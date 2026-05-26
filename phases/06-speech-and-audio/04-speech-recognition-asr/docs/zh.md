# 语音识别 (ASR) — CTC, RNN-T, 注意力机制

> 语音识别本质上是每个时间步的音频分类，再通过一个理解英语和静音的序列模型将这些分类结果粘合起来。CTC、RNN-T 和注意力机制是三种实现方式。选择一种并深入理解它。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 6 · 02（语谱图与梅尔频谱），阶段 5 · 08（用于文本的 CNN 与 RNN），阶段 5 · 10（注意力机制）  
**时长：** ~45 分钟  

## 问题

你有一段 10 秒、16 kHz 的音频片段。你希望得到文本输出："turn on the kitchen lights"。挑战在于结构：音频帧与字符并非一一对齐。单词 "okay" 可能持续 200 毫秒或 1200 毫秒。语音间的静默间隔不固定。某些音素比其他音素更长。输出 token 的数量事先未知。

有三种方法可以解决这个问题：

1. **CTC（连接时序分类）。** 为每帧输出 token 概率，其中包括一个特殊的 *blank*（空）。解码时合并重复标记并移除 blank。非自回归，速度快。用于 wav2vec 2.0、MMS。
2. **RNN-T（循环神经网络换能器）。** 联合网络根据编码器帧和之前 token 预测下一个 token。可流式处理。用于 Google 设备端 ASR、NVIDIA Parakeet。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力生成 token，自回归。用于 Whisper、SeamlessM4T。

至 2026 年，LibriSpeech test-clean 上的 SOTA WER 分别为 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异极小，但部署差异巨大。

## 概念

![三种 ASR 方法：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC 直观理解。** 让编码器输出 `T` 个帧级别的分布，每个分布对应 `V+1` 个 token（V 个字符 + 一个 blank）。对于长度为 `U`（`U < T`）的目标序列 `y`，任何能折叠成 `y` 的帧对齐方式都被计算在内。CTC 损失对所有这样的对齐求和。推理时：取每帧最大概率值，折叠重复，移除 blank。

优点：非自回归、可流式、零前瞻。缺点：*条件独立假设*——每帧预测独立于其他帧，因此不存在内部语言模型。通过外部语言模型进行波束搜索或浅层融合来解决。

**RNN-T 直观理解。** 增加了 *预测器* 网络来编码 token 历史，以及一个 *联合器* 将预测器状态与编码器帧结合，输出针对 `V+1` 个标签（其中 `+1` 表示空/不发射）的联合分布。显式建模了 CTC 忽略的条件依赖性。可流式处理，因为每一步仅依赖过去的帧和过去的 token。

优点：可流式 + 内部语言模型。缺点：训练更复杂、更耗内存（3D 损失栅格）；RNN-T 损失内核本身就是一个完整的库类别。

**注意力编码器-解码器。** 编码器（6-32 层 Transformer）处理对数梅尔帧。解码器（6-32 层 Transformer）通过交叉注意力关注编码器输出，自回归地生成 token。没有对齐约束——注意力可以查看音频中的任何位置。除非限制注意力（如分块的 Whisper-Streaming，2024），否则不可流式处理。

优点：离线 ASR 的最高质量，使用标准的 seq2seq 工具易于训练。缺点：自回归延迟与输出长度成正比；不经工程改造无法流式处理。

### WER：唯一指标

**单词错误率** = `(S + D + I) / N`，其中 S = 替换，D = 删除，I = 插入，N = 参考文本的单词数。在单词级别计算编辑距离。越低越好。WER 高于 20% 通常不可用；低于 5% 则与人类朗读水平相当。2026 年标准基准测试上的数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 参数量 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

以上均为基于编码器-解码器或 RNN-T 的模型。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 构建

### 步骤 1：贪婪 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续的重复、移除 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：波束搜索 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境中使用带语言模型融合的前缀树波束搜索；此处为概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：使用 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强大的通用 ASR 的一行代码。可在 24 GB GPU 上以约 20 倍实时速度运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和状态延续；使用支持此功能的库（Parakeet 使用 NeMo，wav2vec 2.0 使用 `transformers` pipeline 的 `chunk_length_s`）。

## 使用

2026 年的选型建议：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备、移动端、<500 ms 延迟 | Whisper-Tiny 量化版 或 Moonshine（2024） |
| 长音频 | 基于 VAD 分段的 Whisper（WhisperX） |
| 领域特定（医疗、法律） | 微调 wav2vec 2.0 + 领域语言模型融合 |

## 2026 年仍然存在的陷阱

- **缺少 VAD。** 对静音运行 Whisper 会产生幻觉（如"Thanks for watching!"）。务必用 VAD 进行门控。
- **字符 vs 单词 vs 子词 WER。** 报告单词级 WER *在*标准化（小写、去除标点）之后。
- **语言 ID 漂移。** Whisper 的自动语言识别会将噪声较多的片段误判为日语或威尔士语；在确定场景下强制设置 `language="en"`。
- **长片段不分块。** Whisper 有 30 秒的窗口。对于更长的内容，使用 `chunk_length_s=30, stride=5`。

## 交付

保存为 `outputs/skill-asr-picker.md`。根据给定的部署目标选择模型、解码策略、分块方式和语言模型融合方案。

## 练习

1. **简单。** 运行 `code/main.py`。它会贪婪地解码一个手动构造的 CTC 输出，并对照参考文本计算 WER。
2. **中等。** 实现步骤 2 中的前缀树波束搜索（正确处理 blank 合并规则）。在一个包含 10 个示例的合成数据集上与贪婪解码进行比较。
3. **困难。** 使用 `whisper-large-v3-turbo` 处理 [LibriSpeech test-clean](https://www.openslr.org/12)。计算前 100 条语句的 WER。与已发表的结果进行比较。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | 带 blank token 的损失函数 | 对所有帧到 token 对齐的边缘求和；非自回归。 |
| RNN-T | 流式损失函数 | CTC + 下一个 token 预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意力的解码器；最佳离线质量。 |
| WER | 你报告的那个数值 | 单词级别的 `(S+D+I)/N` |
| Blank | 空白 | CTC 中特殊的 token，表示"该帧无输出" |
| 语言模型融合 | 外部语言模型 | 在波束搜索期间添加加权的语言模型对数概率。 |
| VAD | 静音门控 | 语音活动检测；去除无语音片段。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年经典论文；v3-turbo 扩展在 2024 年。
- [NVIDIA NeMo — Parakeet-TDT 模型卡](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年开源 ASR 排行榜领先者。
- [Hugging Face — 开源 ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 涵盖 25+ 个模型的实时基准测试。
