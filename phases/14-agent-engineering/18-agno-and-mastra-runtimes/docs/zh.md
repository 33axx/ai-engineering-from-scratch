# Agno 与 Mastra：生产级运行时

> Agno（Python）与 Mastra（TypeScript）是 2026 年的生产运行时搭档。Agno 专注于微秒级 Agent 实例化和无状态 FastAPI 后端。Mastra 则在 Vercel AI SDK 基础上提供 Agent、工具、工作流、统一模型路由和复合存储。

**类型：** 学习
**语言：** Python, TypeScript
**前置知识：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 13（LangGraph）
**时长：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标及其适用场景。
- 说出 Mastra 的三个原语——Agent、工具、工作流——以及支持的服务器适配器。
- 解释为什么无状态会话作用域的 FastAPI 后端是推荐的 Agno 生产路径。
- 针对给定技术栈（Python 优先 vs TypeScript 优先）选择 Agno 还是 Mastra。

## 问题所在

LangGraph、AutoGen、CrewAI 都是框架重量级选手。那些想要“仅 Agent 循环，快速运行，在我自己的运行时中”的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都牺牲了一些框架拥有的原语，以换取原始速度和与周边技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身为 Phi-data。
- “无需图、链或复杂模式——只需要纯粹的 Python。”
- 其文档中的性能目标：约 2μs 的 Agent 实例化，每个 Agent 约 3.75 KiB 内存，约 23 个模型提供商。
- 生产路径：无状态会话作用域的 FastAPI 后端。每个请求启动一个全新的 Agent；会话状态存储在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和 Agentic RAG。

当每秒需要成千上万个短生命周期 Agent（如聊天扇入、评估管道）时，速度目标至关重要。当单个 Agent 运行 10 分钟时，这些目标就不那么重要了。

### Mastra

- TypeScript，基于 Vercel AI SDK。
- 三个原语：**Agent**、**工具**（Zod 类型化）、**工作流**。
- 统一模型路由——截至 2026 年 3 月，覆盖 94 个提供商的 3300+ 模型。
- 复合存储：内存、工作流、可观测性可分别指向不同后端；大规模可观测性推荐使用 ClickHouse。
- Apache 2.0 许可，`ee/` 目录采用源码可用的企业许可。
- 服务器适配器支持 Express、Hono、Fastify、Koa；与 Next.js 和 Astro 提供一级集成。
- 附带 Mastra Studio（localhost:4111）用于调试。
- 截至 1.0 版本（2026 年 1 月），GitHub 星标 22k+，npm 周下载量 300k+。

### 定位

两者都不试图成为 LangGraph。它们在以下方面竞争：

- **语言契合度。** Agno 适合 Python 优先的团队；Mastra 适合 TypeScript 优先的团队。
- **运行时人体工程学。** Agno = 接近零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是自有产品。

### 何时选择各框架

- **Agno**——Python 后端，大量短生命周期 Agent，强性能需求，FastAPI 技术栈。
- **Mastra**——TypeScript 后端，Next.js / Vercel 部署，统一多提供商模型路由，Zod 类型化工具。
- **LangGraph**（第 13 课）——当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK**——当你想要提供商的成品形状时（第 16–17 课）。

### 这种模式出问题的情况

- **为性能而性能。** 仅仅因为“2μs”听起来不错而选择 Agno，但工作负载是每个请求调用一个慢速 Agent。此时开销并非瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是优势，在其他地方则是劣势。
- **企业许可混淆。** Mastra 的 `ee/` 目录是源码可用，而非 Apache 2.0。如果打算分叉，请仔细阅读许可条款。

## 动手构建

本课主要是对比性质的——没有单一的代码工件能公平地展示两个框架。请参见 `code/main.py` 中的并排示例：一个最小的“运行 Agent，流式输出，持久化会话”流程，用两种方式实现（一次 Agno 风格，一次 Mastra 风格）。

运行它：

```
python3 code/main.py
```

两条结构不同但功能等价的轨迹。

## 使用它

- **Agno**——需要速度和 FastAPI 风格的 Python 后端。
- **Mastra**——拥有多个提供商和工作流原语的 TypeScript 后端。
- 两者都提供一级可观测性钩子，并且都与 Langfuse 集成。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运营形态选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 的文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。哪些功能消失了？哪些保留了下来？
2. 阅读 Mastra 的文档。将相同的循环移植到 Mastra。工具类型化（Zod vs 无类型化）发生了什么变化？
3. 基准测试：在你的技术栈上测量 Agent 实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计迁移：如果你一直在 Python 中使用 CrewAI，迁移到 Agno 会破坏哪些东西？
5. 阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源分叉？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|---------|
| Agno | “快速 Python Agent” | 无状态会话作用域 Agent 运行时 |
| Mastra | “基于 Vercel AI SDK 的 TypeScript Agent” | Agent + 工具 + 工作流 + 模型路由 |
| 统一模型路由 | “多提供商访问” | 单个客户端覆盖 94 个提供商的 3300+ 模型 |
| 复合存储 | “多后端” | 内存/工作流/可观测性分别存储到不同数据存储 |
| Mastra Studio | “本地调试器” | localhost:4111 UI，用于内省 Agent |
| 源码可用 | “非开源” | 许可允许阅读源码，但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) —— 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) —— 原语、服务器适配器、模型路由
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成中引用的可观测性对比
