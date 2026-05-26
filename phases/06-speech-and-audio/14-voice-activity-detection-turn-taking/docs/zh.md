# 语音活动检测与话轮转换 — Silero、Cobra 与 Flush 技巧

> 每个语音助手的成败都取决于两个决策：用户现在是否在说话，以及他们是否说完了？VAD 回答第一个问题。话轮检测（VAD + 静音悬停 + 语义端点模型）回答第二个问题。若其中任何一个出错，你的助手要么打断用户，要么永远不停嘴。

**类型：** 构建
**语言：** Python
**先决条件：** 第 6 阶段 · 11（实时音频）、第 6 阶段 · 12（语音助手）
**时间：** 约 45 分钟

## 问题

语音助手在每个 20 毫秒的音频块上需要做出三项截然不同的决策：

1. **这一帧是语音吗？** — VAD。二分类，逐帧判断。
2. **用户开始新的发言了吗？** — 起音检测。
3. **用户说完了吗？** — 终点定位（话轮结束）。

朴素方法（能量阈值）在任何噪声环境下都会失效——车流声、键盘声、人群嘈杂。2026 年的答案：Silero VAD（开源、深度学习） + 话轮检测模型（语义端点检测） + 经 VAD 校准的静音悬停。

## 概念

![VAD 级联：能量 → Silero → 话轮检测器 → Flush 技巧](../assets/vad-turn-taking.svg)

### 三级 VAD 级联

**第一级：能量门控。** 成本最低。将 RMS 阈值设为 -40 dBFS。过滤明显静音，但任何高于阈值的噪声都会触发。

**第二级：Silero VAD**（2020-2026，MIT 许可证）。100 万参数。在 6000 多种语言上训练。在单 CPU 线程上，每 30 毫秒块仅需约 1 毫秒。在 5% 假阳性率下真阳性率为 87.7%。开源默认选项。

**第三级：语义话轮检测器。** LiveKit 的话轮检测模型（2024-2026）或你自己的小型分类器。区分“句中停顿”与“说完”。利用语言上下文（语调 + 近期词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；当 > 0.5（默认）或 > 0.3（敏感）时判定为语音。阈值越低，首字被剪切的概率越小，但误报越多。
- **最小语音时长。** 拒绝短于 250 毫秒的语音——通常是咳嗽或椅子噪音。
- **静音悬停（终点定位）。** VAD 恢复为 0 后，等待 500-800 毫秒再宣布话轮结束。太短会打断用户；太长会感觉迟钝。
- **预卷缓冲。** 在 VAD 触发前保留 300-500 毫秒的音频。防止“喂”字被剪掉。

### Flush 技巧（Kyutai 2025）

流式 STT 模型存在前视延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常需要在说话结束后等待那么久才能得到转录结果。Flush 技巧：当 VAD 触发话轮结束信号时，**向 STT 发送一个冲刷信号**，强制立即输出。STT 处理速度约为实时 4 倍，因此 500 毫秒的缓冲在约 125 毫秒内完成。

端到端延迟：125 毫秒（VAD）+ 冲刷 STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下的 TPR | 延迟 | 许可证 |
|-----|----------------|------|--------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业许可 |
| pyannote 分割 | 95% | ~10 ms | 类 MIT |

Silero 是合适的默认选择。Cobra 是合规性/准确性升级。纯能量 VAD 在 2026 年的生产环境中没有立足之地。

## 动手实践

### 步骤1：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤3：话轮结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤4：Flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持冲刷功能才能生效。Whisper 流模式不支持——它是基于块的，总是等待完整的片段。

## 使用建议

| 场景 | VAD 选择 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/说话人分割 | pyannote 分割 |
| 零依赖回退 | WebRTC VAD（遗留） |
| 需要高质量的结束检测 | Silero + LiveKit 话轮检测器分层 |

经验法则：除非真无其他选择，否则永远不要只使用纯能量 VAD。

## 注意事项

- **固定阈值。** 在安静环境下有效，在嘈杂环境下失效。要么在设备上校准，要么切换为 Silero。
- **静音悬停过短。** 助手在句中被断。对话语音的合适区间是 500-800 毫秒。
- **悬停过长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **没有预卷缓冲。** 用户音频的前 200-300 毫秒丢失。始终保持滚动预卷。
- **忽略语义端点检测。** “嗯，让我想想……” 包含长停顿。用户讨厌在思考时被打断。使用 LiveKit 的话轮检测器或类似方案。

## 交付产出

保存为 `outputs/skill-vad-tuner.md`。选择一个工作负载，选择 VAD 模型、阈值、悬停、预卷和话轮检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟一个“说话+静音+说话+咳嗽”的序列，并测试三个 VAD 层级。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟的录音，调整阈值以最小化首字截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个小型话轮检测器：Silero VAD + 一个基于最后 10 个词嵌入（使用 sentence-transformers）的 3 层 MLP。在一个手工标注的话轮结束数据集上训练。将 F1 分数比单纯 Silero 提升 10%。

## 术语表

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| VAD | 语音检测器 | 逐帧二分类：这是语音吗？ |
| Turn detection | 终点定位 | VAD + 静音悬停 + 语义端点。 |
| Silence hangover | 说话后等待时间 | 宣布话轮结束前等待的时间；500-800 毫秒。 |
| Pre-roll | 说话前缓冲 | VAD 触发前保留 300-500 毫秒音频。 |
| Flush trick | Kyutai 技巧 | VAD → 冲刷 STT → 125 毫秒而非 500 毫秒延迟。 |
| Semantic endpoint | “他们打算停吗？” | 基于机器学习、观察词汇而非仅静音的分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## 扩展阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 开源 VAD 参考实现。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确性领导者。
- [Kyutai — 静音解除 + Flush 技巧](https://kyutai.org/stt) — 低于 200 毫秒的工程技巧。
- [LiveKit — 话轮检测](https://docs.livekit.io/agents/logic/turns/) — 生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基准。
- [pyannote 分割](https://github.com/pyannote/pyannote-audio) — 说话人分割级别的分割。
