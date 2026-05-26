# 顶点项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉自然的语音智能体，端到端延迟需低于 800 毫秒，能判断用户何时停止说话，支持打断，且调用工具时不会卡顿。Retell、Vapi、LiveKit Agents 和 Pipecat 都在 2026 年达到了这个标准。它们采用相同的架构：流式 ASR、话轮检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，并在每个环节严格控制延迟预算。构建这样一个系统，测量 WER、MOS 和误切断率，并在丢包条件下运行测试。

**类型：** 顶点项目  
**语言：** Python（智能体 + 流水线）、TypeScript（Web 客户端）  
**先决条件：** 阶段 6（语音与音频）、阶段 7（Transformers）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 17（基础设施）  
**涉及的阶段：** P6 · P7 · P11 · P13 · P14 · P17  
**时间：** 30 小时

## 问题

语音一直是 2025-2026 年发展最快的 AI 用户体验类别。每个季度的技术门槛都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都使得首次音频输出延迟低于 800 毫秒成为可能。但标准不仅仅是延迟。更重要的是交互体验：不要打断用户、不被用户打断、从中途打断中恢复、在对话中调用工具而不卡顿音频、在抖动的移动网络中保持稳定。

仅通过拼接三次 REST 调用无法实现这一目标。架构必须是端到端的流式流水线。构建它之后，失效模式就会显现出来：为电话音频调优的 VAD 在背景电视噪音下被触发、等待标点符号的话轮检测器始终等不到、TTS 缓冲 400 毫秒后才开始输出。这个顶点项目的目标就是逐个修复这些问题，并在负载下进行测试，然后发布一份延迟与质量报告。

## 概念

该流水线包含五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**话轮检测**（VAD 加上一个小型话轮检测模型，该模型读取部分转录以判断是否完成）、**LLM**（一旦话轮被判定完成，立即流式输出 token）、**TTS**（在第一个 LLM token 出现后约 200 毫秒内流式输出音频）。

三个横切关注点：**打断**：当用户在智能体说话时开始讲话，TTS 立即取消，ASR 立即接管。**工具使用**：对话中的函数调用（天气、日历）必须在侧通道上运行，不造成音频卡顿；如果延迟超过 300 毫秒，智能体预填充一个确认 token（“请稍等……”）。**背压**：在丢包情况下，部分转录被暂存，VAD 提高语音门限阈值，智能体避免在确认消息未被接收时说话。

测量指标是定量的。在 Hamming VAD 基准测试、信噪比 15 dB 条件下，WER 低于 8%。在 100 次测量通话中，首次音频输出 p50 低于 800 毫秒。误切断率低于 3%。TTS 的 MOS 高于 4.2。在单个 g5.xlarge 上支持 50 路并发通话。这些数字是交付成果。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- **传输**：LiveKit Agents 1.0（WebRTC）加上 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- **ASR**：Deepgram Nova-3（流式，首个部分转录低于 300 毫秒）或自行托管的 faster-whisper Whisper-v3-turbo
- **VAD**：Silero VAD v5 加上 LiveKit 话轮检测器（读取部分转录的小型 Transformer）
- **LLM**：OpenAI GPT-4o-realtime 用于紧密集成、Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，独立音频路径）
- **TTS**：Cartesia Sonic-2（最低首个字节延迟）、ElevenLabs Flash v3，或开源的 Orpheus 用于自托管
- **工具**：FastMCP 侧通道用于天气/日历/预订；如果工具耗时超过 300 毫秒，智能体预发填充语
- **可观测性**：OpenTelemetry 语音跨度、Langfuse 语音追踪（含音频回放）
- **部署**：单个 g5.xlarge（24GB VRAM）用于自托管的 Whisper + Orpheus；托管的 API 用于最低延迟

## 构建步骤

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器上，附加一个加入房间的智能体工作进程。

2. **ASR 流式传输。** 将 20 毫秒 PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录结果。记录每个部分的延迟。

3. **VAD 与话轮检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件发生时，针对最新的部分转录触发 LiveKit 话轮检测器。仅当 VAD 检测到 500 毫秒静音且话轮检测器评分 > 0.6 时才判定为“话轮结束”。

4. **LLM 流式传输。** 话轮结束判定后，使用对话历史加最终转录启动 LLM 调用。流式输出 token。在第一个 token 出现时，移交给 TTS。

5. **TTS 流式传输。** Cartesia Sonic-2 流式返回音频块。第一个音频块必须在第一个 LLM token 出现后 200 毫秒内离开服务器。将音频块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲播放。

6. **打断。** 当 TTS 正在播放时 VAD 检测到新的用户语音，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启用 ASR。发布一个 `tts_canceled` 跨度。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时，并发执行该调用；如果在 300 毫秒内未解决，让 LLM 输出“请稍等，让我查一下”作为填充语；工具返回后恢复。

8. **评估工具。** 录制 100 次通话。计算 WER（对照保留转录）、误切断率（用户说话中途 TTS 被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 使用合成呼叫者在单个 g5.xlarge 上驱动 50 路并发通话。测量稳定的首次音频输出 p95。

## 使用方法

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付成果

`outputs/skill-voice-agent.md` 是最终的交付文件。给定一个领域（客户支持、日程安排或自助服务终端），它将搭建一个 LiveKit 智能体，其 ASR/VAD/LLM/TTS 流水线根据测量标准进行调优。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 在 100 次录制通话中，首次音频输出 p50 低于 800 毫秒 |
| 20 | 话轮切换质量 | 在 Hamming VAD 基准测试中，误切断率低于 3% |
| 20 | 工具使用正确性 | 对话中的工具调用能返回正确数据且不造成音频卡顿 |
| 20 | 丢包下的可靠性 | 在注入 3% 数据包丢失时，WER 和话轮切换稳定性 |
| 15 | 评估工具完整性 | 可重现的测量结果，带有公开配置 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差异。确定 CPU 与 GPU 决策的关键点。

2. 添加一个打断仲裁策略：当用户在工具调用期间打断时，智能体应该怎么做？比较三种策略（硬取消、完成工具后停止、排队下一话轮）。

3. 运行对抗性话轮检测器测试：让用户在句子中间长时间停顿。调整 VAD 静音阈值和话轮检测器评分阈值，以在误切断率最低且不超过 900 毫秒的前提下取得最佳效果。

4. 通过 Twilio 将同一智能体部署到 PSTN。比较 PSTN 首次音频输出与 WebRTC。解释抖动缓冲和编解码器的差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 的误触发率与针对特定语言微调的模型之间的差异。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------------|------------------------|
| Turn detection | “话语结束” | 根据 VAD 静音和部分转录判断用户是否说完了的分类器 |
| Barge-in | “打断处理” | 当 VAD 检测到新的用户语音时，中断 TTS 播放 |
| First-audio-out | “延迟” | 从用户停止说话到第一个音频数据包离开服务器的时间 |
| VAD | “语音门控” | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| Jitter buffer | “音频平滑” | 客户端缓冲区，短暂持有数据包以吸收网络波动 |
| Filler | “确认 token” | 工具响应慢时，智能体为避免静音发出的简短短语 |
| MOS | “平均意见得分” | 感知语音质量评分；NISQA 是自动化的代理评分 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选的以 Python 为优先的流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 备选的生产级参考
