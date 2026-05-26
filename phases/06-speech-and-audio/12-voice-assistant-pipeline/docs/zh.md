# 构建语音助手指令流水线 — 第六阶段毕业设计

> 整合第01-11课的所有内容。构建一个能听、能思考、能回应的语音助手。到2026年，这已是一个可解决的工程问题，而非研究问题——但集成细节决定了它能否真正落地。

**类型：** 构建  
**语言：** Python  
**前置课程：** 第六阶段·第04、05、06、07、11课；第十一阶段·第09课（函数调用）；第十四阶段·第01课（Agent循环）  
**预计用时：** ~120分钟

## 问题描述

构建一个端到端的助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录文本传递给能调用工具（定时器、天气、日历）的LLM。
5. 将LLM输出文本流式传输给TTS。
6. 将音频播放给用户。
7. 如果用户在回应中途打断，则停止播放。

延迟目标：在笔记本CPU上，用户说完话后800毫秒内输出第一个TTS音频字节。质量目标：无漏词、无静音时幻觉字幕、无不必要的语音克隆泄漏、无提示注入成功。

## 概念图

![语音助手指令流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 ms 块。Python中通常使用`sounddevice`，生产环境中使用原生AudioUnit/ALSA/WASAPI。
2. **VAD（第11课）。** Silero VAD，阈值0.5，最短语音250 ms，静默挂起500 ms。发出“开始”和“结束”信号。
3. **流式STT（第4-5课）。** Whisper-streaming、Parakeet-TDT或Deepgram Nova-3（API）。支持部分结果和最终转录。
4. **带工具调用的LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具使用JSON Schema。流式输出token。
5. **流式TTS（第7课）。** Kokoro-82M（最快的开源方案）或Cartesia Sonic（商业方案）。收到20个LLM token后开始TTS。
6. **播放。** 扬声器输出；在低带宽网络上使用Opus编码。
7. **打断处理。** 如果在TTS播放期间VAD触发，停止播放、取消LLM、重启STT。

### 你会遇到的三种失败模式

1. **首词截断。** VAD启动稍晚，用户的“嘿”字丢失。将起始阈值设为0.3而非0.5。
2. **回应中途打断混乱。** 用户打断后LLM仍在生成；助手会与用户说话重叠。将VAD连接到取消LLM的逻辑。
3. **静音幻觉。** Whisper在静音预热帧上输出“Thanks for watching”。始终用VAD进行门控。

### 2026年生产参考技术栈

| 技术栈 | 延迟 | 许可 | 备注 |
|-------|------|------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业API | 2026年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源 | 适合DIY |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单模型；不同架构，见第15课 |
| Vapi / Retell（托管服务） | 300-500 ms | 商业 | 上线最快；定制化有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私/边缘计算 |

## 构建过程

### 步骤1：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤2：VAD门控的回合捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤4：在LLM循环中调用工具

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用方式

参见 `code/main.py`，这是一个可运行的模拟程序，将所有七个组件与桩模型连接起来，即使没有硬件也能看到流水线结构。如需真实实现，请将桩组件替换为：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 或 `openai-whisper`
- `openai` (`gpt-4o`) 或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于输入/输出

## 陷阱

- **永久记录PII。** 完整回合的音频在大多数司法管辖区属于个人身份信息。建议保留30天，加密存储。
- **没有插话功能。** 用户会打断，你的助手必须停止说话。
- **TTS阻塞。** 同步TTS会阻塞事件循环。请使用异步或独立线程。
- **没有工具调用错误处理。** 工具会失败。LLM必须接收错误并重试一次，然后优雅降级。
- **过于激进的幻觉过滤。** 过滤过度，助手会重复说“我无法处理这个”。过滤不足则会乱说。需在保留集上校准。
- **没有唤醒词选项。** 始终监听是一种隐私风险。请添加唤醒词门控（Porcupine或openWakeWord）。

## 交付

保存为 `outputs/skill-voice-assistant-architect.md`。根据预算、规模、语言和合规性约束，产出一份完整的技术栈规范。

## 练习

1. **简单。** 运行 `code/main.py`。它将使用桩模块模拟一次完整的端到端回合，并打印每阶段延迟。
2. **中等。** 将STT桩替换为真实Whisper模型，处理预先录制的 `.wav` 文件。测量WER和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意API）和 `set_timer`。让LLM通过工具进行路由，并验证当用户说“设置5分钟定时器”时，正确的函数被触发并口头回复确认。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 回合 | 用户与助手的往返 | 一个由VAD界定的用户语音 + 一个LLM-TTS回应。 |
| 插话 | 打断 | 用户在助手说话时说话；助手停止。 |
| 唤醒词 | “嘿，助手” | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| 端点检测 | 回合结束 | VAD + 最小静默判断用户说完话。 |
| 预缓冲 | 说话前缓冲区 | 在VAD触发前保留200-400 ms音频，避免首词截断。 |
| 工具调用 | 函数调用 | LLM发出JSON；运行时调度；结果反馈回循环。 |

## 延伸阅读

- [LiveKit — 语音代理快速入门](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — 语音代理示例](https://github.com/pipecat-ai/pipecat) — 适合DIY的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管式的语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考实现（第15课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM函数调用。
