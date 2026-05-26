# A2A — 智能体间通信协议（Agent-to-Agent Protocol）

> 谷歌于2025年4月宣布A2A；到2026年4月，规范位于 https://a2a-protocol.org/latest/specification/，已有超过150个组织支持。A2A是MCP（第13课）的水平互补协议：MCP是垂直方向（智能体 ↔ 工具），而A2A是对等方向（智能体 ↔ 智能体）。它定义了Agent Card（发现）、带有工件（文本、结构化数据、视频）的任务、不透明的任务生命周期以及认证。生产系统越来越多地将MCP与A2A结合使用。2025-2026年间，Google Cloud将A2A支持集成到Vertex AI Agent Builder中。

**类型：** 学习 + 构建
**语言：** Python（标准库，`http.server`，`json`）
**前置要求：** 阶段16 · 04（原始模型）
**时长：** ~75分钟

## 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个HTTP端点，定义一个自定义JSON模式，然后希望对方能理解。每一对智能体都变成了一个定制集成。

A2A就是针对这种调用的通用线路协议。标准化的发现、任务模型、传输协议、工件。就像HTTP+REST，但智能体是一等公民。

## 概念

### 四大要素

**Agent Card（智能体名片）。** 位于 `/.well-known/agent.json` 的一个JSON文档，描述智能体的名称、技能、端点、支持的模式、认证要求。发现过程就是读取该卡片。

```json
{
  "name": "MyAgent",
  "version": "1.0.0",
  "capabilities": {
    "skills": ["translate", "summarize"],
    "modalities": ["text", "json"]
  },
  "endpoints": {
    "tasks": {
      "url": "https://example.com/a2a/tasks",
      "streaming": false,
      "auth": { "type": "bearer" }
    }
  }
}
```

**任务（Task）。** 工作的单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送一个任务，然后轮询或订阅以获取更新。

**工件（Artifact）。** 任务产生的结果类型。文本、结构化JSON、图像、视频、音频。工件具有类型，因此不同的模式都是一等公民。

**不透明的生命周期（Opaque lifecycle）。** A2A不规定远程智能体如何解决任务。客户端只看到状态转换和工件；实现上可以使用任何框架。

### MCP/A2A的区分

- **MCP**（第13课）：智能体 ↔ 工具。智能体通过JSON-RPC对工具服务器进行读写。默认为无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是具有自己推理能力的智能体。

生产环境的多智能体系统两者都用。一个A2A对等体可以在自己这一侧调用MCP工具。这种分离保持了两个关注点的清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON───────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts───
```

或者使用流式传输：通过SSE订阅 `/tasks/{id}/events` 以接收推送更新。

### 认证

A2A支持三种常见模式：

- **Bearer令牌** —— OAuth2或不透明令牌。
- **mTLS** —— 双向TLS；各组织相互验证身份。
- **签名请求** —— 对载荷进行HMAC签名。

认证在Agent Card中声明；客户端发现并遵守。

### 到2026年4月已有超过150个组织支持

企业采用推动了A2A的规模化。要点：A2A成为企业智能体系统跨越信任边界的标准方式。Google Cloud在Vertex AI Agent Builder中集成了A2A支持；Microsoft Agent Framework支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了A2A适配器。

### A2A的优势

- **跨组织调用。** A公司的智能体调用B公司的智能体。没有A2A，每对智能体就是一个定制契约。
- **异构框架。** LangGraph智能体调用CrewAI智能体，再调用自定义Python智能体。A2A使之规范化。
- **类型化工件。** 视频结果、结构化JSON、音频——都是一等公民。
- **长时间运行的任务。** 不透明的生命周期加上轮询，使耗时数小时的任务变得简单。

### A2A的不足

- **对延迟敏感的微调用。** A2A的生命周期是异步的。亚毫秒级的智能体间通信不适合；应使用直接RPC。
- **紧密耦合的进程内智能体。** 如果两个智能体运行在同一个Python进程中，A2A的HTTP往返就过于繁琐。
- **小团队。** 规范的开销是真实存在的；仅用于内部的智能体可能不需要这种形式化。

### A2A与ACP、ANP、NLIP的对比

在2024-2026年间出现了几个相关规范：

- **ACP**（IBM/Linux基金会）—— A2A的前身，范围较窄。
- **ANP**（智能体网络协议）—— 侧重对等发现，去中心化优先。
- **NLIP**（Ecma自然语言交互协议，2025年12月标准化）—— 自然语言内容类型。

截至2026年4月，A2A是采纳最广泛的对等协议。参见arXiv:2505.02279（Liu等人，《智能体互操作性协议综述》）以进行对比。

## 动手构建

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小化的A2A服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回工件。

客户端：

- 获取Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取工件。

运行：

```bash
python code/main.py
```

该脚本在后端线程中启动服务器，然后客户端对其进行操作。你可以看到完整的流程：发现、提交、轮询、工件。

## 使用它

`outputs/skill-a2a-integrator.md` 设计了一个A2A集成方案：Agent Card内容、任务模式、认证选择、流式传输 vs 轮询。

## 投产检查清单

- **锁定规范版本。** A2A仍在演进中；Agent Card应声明协议版本。
- **任务创建幂等。** 重复提交（网络重试）应只产生一个任务。
- **工件模式。** 声明智能体返回的是什么形状；消费者应进行校验。
- **速率限制 + 认证。** A2A面向公网；应应用标准的Web安全措施。
- **失败任务的死信队列。** 定期检查模式以发现重复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现了服务器并收到了正确的工件。
2. 为服务器添加第二个技能（例如“summarize”）。更新Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现一个SSE流式端点：`/tasks/{id}/events`，用于发出状态变更事件。客户端需要做哪些不同的处理？
4. 阅读A2A规范（https://a2a-protocol.org/latest/specification/）。指出该规范强制要求但本演示未实现的三个事项。
5. 比较A2A（Agent Card发现）与MCP（通过 `listTools` 进行服务端能力列表）。自我描述的智能体与通过能力探测相比，各有什么权衡？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| A2A | “智能体到智能体” | 跨系统智能体调用智能体的对等协议。Google 2025年提出。 |
| Agent Card | “智能体的名片” | 位于 `/.well-known/agent.json` 的JSON文件，描述技能、端点、认证。 |
| Task | “工作单元” | 具有生命周期的异步有状态对象；完成时产生工件。 |
| Artifact | “结果” | 类型化的输出：文本、结构化JSON、图像、视频、音频。媒体是一等公民。 |
| Opaque lifecycle | “如何解决是智能体自己的事” | 客户端只看到状态转换；服务器可自由选择框架/工具。 |
| Discovery | “寻找智能体” | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | “工具 vs 对等体” | MCP：垂直方向 智能体 ↔ 工具。A2A：水平方向 智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | “兄弟协议” | 相邻规范；A2A是2026年采纳最广泛的协议。 |

## 延伸阅读

- [A2A规范](https://a2a-protocol.org/latest/specification/) — 权威规范
- [Google Developers Blog — A2A发布公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025年4月启动博文
- [A2A GitHub仓库](https://github.com/a2aproject/A2A) — 参考实现和SDK
- [Liu等人 — 智能体互操作性协议综述](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP对比
