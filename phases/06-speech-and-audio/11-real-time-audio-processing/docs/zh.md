# 实时音频处理

> 批处理管道处理一个文件。实时管道需要在下一个 20 毫秒数据到达前处理完前 20 毫秒。每个对话式 AI、广播工作室和电话机器人的存活都取决于这个延迟预算。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 6 · 02（语谱图），阶段 6 · 04（ASR），阶段 6 · 07（TTS）
**时间：** ~75 分钟

## 问题

你希望一个语音助手感觉有活力。人类对话轮换延迟约为 230 毫秒（静音到响应）。超过 500 毫秒会显得机械；超过 1500 毫秒则感觉有故障。在 2026 年，完整的 **听到 → 理解 → 响应 → 说话** 循环的预算是：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 毫秒 |
| VAD | 10 毫秒 |
| ASR（流式） | 150 毫秒 |
| LLM（第一个 token） | 100 毫秒 |
| TTS（第一个块） | 100 毫秒 |
| 渲染 → 扬声器 | 20 毫秒 |
| **总共** | **~400 毫秒** |

Moshi（Kyutai，2024）实现了 200 毫秒全双工。GPT-4o-realtime（2024）约为 320 毫秒。2022 年的级联管道发货时延迟为 2500 毫秒。这 10 倍的提升来自三种技术：(1) 处处流式处理，(2) 带有部分结果的异步流水线，(3) 可中断生成。

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小的块流动。常见选择：20 毫秒（16 kHz 下 320 个样本）。下游所有环节都必须跟上这个节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径上的内存分配。大小 ≈ 最大延迟 × 采样率；一个 2 秒的 16 kHz 环 = 32,000 个样本。

**VAD（语音活动检测）。** 当无人说话时，门控下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 毫秒帧运行时间 <1 毫秒。`webrtcvad` 是较老的替代方案。

**流式 ASR。** 随着音频到达就发出部分转录的模型。Parakeet-CTC-0.6B 流式模式（NeMo，2024）在 320 毫秒延迟下达到 2–5% 的词错误率。Whisper-Streaming（Macháček 等人，2023）将 Whisper 分块以实现近流式，延迟约 2 秒。

**打断。** 当用户在助手说话时发言，你必须 (a) 检测插话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。所有操作需在 100 毫秒内完成，否则用户会感觉助手失聪。

**WebRTC Opus 传输。** 20 毫秒帧，48 kHz，自适应比特率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的栈。

**抖动缓冲区。** 网络数据包乱序/迟到到达。抖动缓冲区重新排序并平滑；太小 → 可听间隙，太大 → 延迟。典型值为 60–80 毫秒。

### 常见陷阱

- **线程竞争。** Python 的 GIL 加上重型模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio）并将 Python 保持在热路径之外。
- **采样率转换延迟。** 在管道内重采样会增加 5–20 毫秒。要么提前重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是像 Kokoro 这样的快速 TTS，首次请求也有 100–200 毫秒的预热。缓存模型，并在第一个真实轮次之前用一次虚拟运行预热。
- **回声消除。** 没有 AEC，TTS 输出会重新进入麦克风并触发生成机器人自身声音的 ASR。WebRTC AEC3 是开源默认方案。

## 构建它

### 步骤 1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。32,000 个样本在 16 kHz 下 = 2 秒。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产环境中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 作用于音频轨道是标准方法。

## 使用它

2026 年的栈：

| 层 | 选择 |
|-------|------|
| 传输 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为了安全缓冲 500 毫秒。** 缓冲区 *就是* 你的延迟下限。缩小它。
- **未绑定线程。** 音频回调在优先级低于 UI 的线程上 = 负载下出现卡顿。
- **TTS 块太小。** 低于 200 毫秒的块会使声码器伪影可听。320 毫秒的块是最佳点。
- **没有抖动缓冲区。** 真实网络有抖动；没有平滑会得到爆音。
- **一次性错误处理。** 音频管道必须防崩溃。一个异常就会杀死会话。

## 交付

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频管道，包含每个阶段的具体延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；针对一个虚拟的 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个直通循环，以 20 毫秒帧处理你的麦克风，并在每帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建一个全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端玻璃到玻璃延迟。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 固定大小、无锁（或单生产者单消费者锁定）的音频帧 FIFO。 |
| VAD | 静音门控 | 模型或启发式算法标记语音与非语音。 |
| 流式 ASR | 实时语音转文字 | 音频到达时输出部分文本；有限前向回看。 |
| 抖动缓冲区 | 网络平滑器 | 对乱序数据包进行重排序的队列；典型 60–80 毫秒。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| Barge-in | 用户打断 | 系统在 TTS 播放过程中检测到用户语音；必须取消播放。 |
| 全双工 | 同时双向通信 | 用户和机器人可以同时说话；Moshi 是全双工的。 |

## 进一步阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 毫秒延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚 1 毫秒 VAD，Apache 2.0 许可。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。
