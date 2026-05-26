# 语音代理：Pipecat 与 LiveKit

> 语音代理在 2026 年已成为一类重要的生产级产品。Pipecat 为您提供基于 Python 帧的管道（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 将 AI 模型通过 WebRTC 桥接到用户。优质技术栈的端到端延迟目标为 450–600ms。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 01（智能体循环），阶段 14 · 12（工作流模式）
**时长：** 约 60 分钟

## 学习目标

- 描述 Pipecat 基于帧的管道：DOWNSTREAM（源→接收端）和 UPSTREAM（控制）。
- 列举经典的语音管道阶段以及 Pipecat 支持的传输层。
- 解释 LiveKit Agents 的两类语音代理类（MultimodalAgent、VoicePipelineAgent）及其适用场景。
- 总结 2026 年生产环境的延迟期望及其对架构选择的影响。

## 问题所在

语音代理并非简单的文本循环加上 TTS。延迟预算极为苛刻（约 600ms），部分音频是默认情况，话轮检测本身就是一个模型，传输层从电话 SIP 到 WebRTC 不一而足。要么您构建一个基于帧的管道（Pipecat），要么依赖一个平台（LiveKit）。

## 概念

### Pipecat (pipecat-ai/pipecat)

- 基于 Python 帧的管道框架。
- `Frame` → `FrameProcessor` 链。
- 两种流方向：
  - **DOWNSTREAM** — 源 → 接收端（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并支持观察者用于指标/追踪/RTVI。

典型管道：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输层：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型桥接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音代理类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效技术直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级别的控制。
- 基于 transformer 模型的语义话轮检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话业务。
- 通过 LiveKit Inference 提供 50+ 模型且无需 API 密钥；通过插件支持 200+ 模型。

### 商业平台

Vapi（优化后的高级技术栈约 450–600ms）和 Retell（180 次测试通话中端到端约 600ms）构建于这些基础之上。当您希望拥有托管语音技术栈而又无需组建 WebRTC 团队时，可以选择平台。

### 这种模式容易出错的地方

- **未处理打断。** 用户打断时，代理仍在说话。需要在 Pipecat 中使用 UPSTREAM 取消帧，LiveKit 中则使用等效机制。
- **忽略 STT 置信度。** 将低置信度的转录文本当作准确信息传给 LLM。应当根据置信度设限或要求确认。
- **TTS 在句子中间被截断。** 当管道在话语中间取消时，TTS 需要感知到这一点并切断音频。
- **忽视延迟预算。** 每个组件都会增加 50–200ms。在交付前请累加整条链路。

### 2026 年典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 第一个 token：150–400ms
- TTS 第一段音频：100–200ms
- 传输层往返时间：30–80ms

端到端 450–600ms 属于优质。800–1200ms 为常见范围。任何超过 1500ms 都会让人感觉卡顿。

## 动手构建

`code/main.py` 是一个基于帧的玩具管道，包含：

- `Frame` 类型（音频、转录文本、文本、TTS 音频、控制）。
- `Processor` 接口，包含 `process(frame)` 方法。
- 一个五阶段管道（VAD → STT → LLM → TTS → 传输层），通过脚本化的处理器实现。
- 一个演示打断的 UPSTREAM 取消帧。

运行：

```
python3 code/main.py
```

追踪结果会显示正常流程以及一次在 TTS 讲话中间将其打断的取消操作。

## 使用场景

- **Pipecat** 适用于完全控制——自定义处理器、Python 优先、可插拔提供者。
- **LiveKit Agents** 适用于以 WebRTC 为主的部署和电话业务。
- **Vapi / Retell** 适用于无需 WebRTC 团队的托管语音代理。
- **OpenAI Realtime / Gemini Live** 适用于直接的音频输入/输出（MultimodalAgent）。

## 交付成果

`outputs/skill-voice-pipeline.md` 构建了一个 Pipecat 风格的语音管道框架，包括 VAD + STT + LLM + TTS + 传输层以及打断处理。

## 练习

1. 为你的玩具管道添加一个指标观察者：统计每阶段每秒处理的帧数。延迟累积在哪个环节？
2. 实现基于置信度门槛的 STT：低于阈值时，要求“能再说一遍吗？”
3. 添加语义话轮检测：简单规则——如果转录文本以“？”结尾，则话轮结束。
4. 阅读 Pipecat 的传输层文档。将标准库传输层替换为 SmallWebRTCTransport 配置（占位实现）。
5. 针对同一查询，测量 OpenAI Realtime 与 STT+LLM+TTS 级联方案的延迟。文本级别控制带来了多少额外延迟成本？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| 帧（Frame） | “事件” | 管道中带类型的数据单元（音频、转录文本、文本、控制） |
| 处理器（Processor） | “管道阶段” | 包含 process(frame) 的处理程序 |
| DOWNSTREAM | “正向流” | 源到接收端：音频输入，语音输出 |
| UPSTREAM | “反馈流” | 控制：取消、指标、打断 |
| VAD | “语音活动检测” | 检测用户是否在说话 |
| 语义话轮检测 | “智能话轮结束” | 基于模型的决策，判断用户是否说完 |
| MultimodalAgent | “直接音频代理” | 音频输入，音频输出；中间没有文本 |
| VoicePipelineAgent | “级联代理” | STT + LLM + TTS；文本级别控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输层
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，经过延迟基准测试
