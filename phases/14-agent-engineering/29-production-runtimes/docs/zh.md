# 生产运行时：队列、事件、定时任务

> 生产级智能体运行在六种运行时形态上：请求-响应、流式、持久化执行、基于队列的后台、事件驱动和定时调度。先选形态再选框架。可观测性在所有形态下都是关键支撑。

**类型：** 学习  
**语言：** Python（标准库）  
**前置条件：** 阶段14 · 13（LangGraph），阶段14 · 22（语音）  
**时长：** 约60分钟

## 学习目标

- 列举六种生产运行时形态，并将每种形态与框架/产品模式对应。
- 解释为什么持久化执行（LangGraph）对长周期任务至关重要。
- 描述事件驱动运行时以及 Claude Managed Agents 适用的场景。
- 解释“可观测性即关键支撑”这一论断在多步骤智能体中的应用。

## 问题

生产环境中的智能体会以 Jupyter Notebook 无法体现的方式失败：第37步网络超时、用户在语音通话中途挂断、定时任务在机器重启后中止、后台工作进程耗尽内存。运行时形态决定了哪些失败是可恢复的。

## 概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel spans。

### 流式

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将其扩展到 WebRTC 以支持语音/视频（第22课）。
- 技术栈：任何支持流式输出的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块耗时、首 token 延迟、尾延迟。

### 持久化执行

- 每一步后检查点状态；失败时自动恢复。
- AutoGen v0.4 的 actor 模型将故障隔离到单个智能体（第14课）。
- LangGraph 的核心差异化特性（第13课）。
- 当步骤数未知且恢复成本高时至关重要。

### 基于队列 / 后台

- 任务进入队列，工作进程拾取，结果通过 webhook 或 pub/sub 返回。
- 对长周期智能体至关重要（根据 Anthropic 的计算机使用公告，每个任务需要数十到数百步）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义方案。
- 可观测性：队列深度、每任务延迟分布、死信队列大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 创建、定时任务触发。
- Claude Managed Agents 开箱即用覆盖此模式（第17课）。
- CrewAI Flows（第15课）结构化事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动延迟、智能体延迟。

### 定时调度

- 按定时任务（cron）模式运行的智能体，周期性执行。
- 结合持久化执行，使失败的夜间运行在下一次触发时恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管服务（Render cron、Vercel cron）。

### 2026年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第22课）。
- **Claude Managed Agents** 用于托管的长时间运行异步任务。

### 可观测性即关键支撑

如果没有 OpenTelemetry GenAI spans（第23课）加上 Langfuse/Phoenix/Opik 后端（第24课），你将无法调试在第40步失败的多步骤智能体。这在生产环境中不是可选项。这是“快速调试”与“从头重放并添加更多日志”之间的区别。

### 生产运行时常见的失败点

- **错误形态选择。** 为5分钟的任务选择请求-响应。用户挂断；工作进程堆积；重试叠加。
- **没有死信队列。** 没有死信队列的队列工作进程。失败的任务消失。
- **不透明的后台工作。** 没有导出追踪的后台智能体运行。失败直到用户报告才被发现。
- **跳过持久化状态。** 任何运行超过30秒且无法承受重启成本的任务都需要持久化执行。

## 构建

`code/main.py` 是一个标准库的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带有死信队列的基于队列的工作进程。
- 事件触发注册表。
- 定时任务形态的调度器。

运行：

```bash
python3 code/main.py
```

输出：五条追踪记录，展示每种形态在同一任务上的行为。相同的智能体逻辑，不同的外部外壳。持久化执行（第六种形态）有意在第13课中与 LangGraph 检查点一起介绍。

## 使用

- **请求-响应** 用于聊天式用户体验。
- **流式** 用于渐进式响应。
- **持久化** 用于长周期任务。
- **队列** 用于批量/异步/长时间运行。
- **事件** 用于智能体的响应式行为。
- **定时任务** 用于运维任务（内存整合、评估、成本报告）。

## 交付

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并配置可观测性要求。

## 练习

1. 将你在第1课中的 ReAct 循环移植到你技术栈中的所有六种形态。哪种形态适合哪种产品场景？
2. 在基于队列的演示中添加死信队列。模拟10%的任务失败；展示死信队列大小。
3. 编写一个定时触发的评估智能体，每天夜间针对当天的前20条追踪记录运行。
4. 实现带背压的流式：如果客户端速度慢，则暂停智能体。这与轮次预算如何交互？
5. 阅读 Claude Managed Agents 文档。何时你会将自托管的长时间运行智能体迁移到托管方案？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 请求-响应 | “同步” | 用户等待；仅短任务 |
| 流式 | “SSE / WS” | 渐进式输出；更好的用户体验；可逐块观测延迟 |
| 持久化执行 | “从失败中恢复” | 状态检查点；从最后一步重启 |
| 基于队列 | “后台任务” | 生产者/工作进程池/死信队列 |
| 事件驱动 | “基于触发器” | 智能体响应外部事件 |
| 死信队列 | “死信队列” | 失败任务的停车场 |
| Claude Managed Agents | “托管运行框架” | Anthropic 托管的长时间运行异步任务，支持缓存和压缩 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行细节
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长时间运行异步任务
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每个任务数十到数百步”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型故障隔离
