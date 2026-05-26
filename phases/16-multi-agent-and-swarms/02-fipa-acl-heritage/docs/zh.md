# FIPA-ACL 与言语行为的传承

> 在 MCP 和 A2A 之前，存在着 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（IEEE Foundation for Intelligent Physical Agents）批准了一种代理通信语言，包含二十种言语行为（performatives）、两种内容语言以及一套交互协议——合同网（contract net）、订阅/通知（subscribe/notify）、条件请求（request-when）。由于本体论的负担对于网络来说过于沉重，它逐渐从工业界淡出。然而，基于 LLM 的多智能体系统的复兴，正在悄然重新实现同样的思想，只是不再使用形式语义：JSON 合同替代了言语行为，自然语言替代了本体论。本课程将认真解读 FIPA-ACL，以便你能看清 2026 年的哪些协议决策是重新发明，哪些是真正的新颖之处，以及当前这波浪潮会在哪里重新发现 2000 年代已经解决过的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 阶段 16 · 01（为什么需要多智能体）
**时长：** ~60 分钟

## 问题

2026 年的智能体协议格局一片繁忙：MCP 用于工具，A2A 用于智能体，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，此外还有 CA-MCP 以及二十多份研究提案。每个规范都宣称自己是基础性的。

诚实地说，它们中的大多数都是在重新发现一个特定的、已有二十年历史的决策树。来自 Austin（1962）和 Searle（1969）的言语行为理论告诉我们“话语即行动”。KQML（1993）将其转化为一种线缆协议。FIPA-ACL（2000 年获批）产生了参考标准化：二十种言语行为、内容语言 SL0/SL1、用于合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。大约在 2010 年左右，由于本体论负担过重且网络占据主导，这些努力逐渐消退。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是一种更轻量、更 JSON 原生的 FIPA 决策重制版。了解这些传承能告诉你两件事：哪些新的“创新”实际上是重新发明，以及新规范将重新发现哪些旧的失败模式。

## 概念

### 言语行为，用一段话概括

Austin 注意到有些句子不是在描述世界——它们是在改变世界。“我承诺。”“我请求。”“我宣布。”他将这些称为施为性话语（performative utterances）。Searle 将其形式化为五个类别：断言类（assertive）、指令类（directive）、承诺类（commissive）、表达类（expressive）、宣告类（declarative）。KQML（Finin 等人，1993）将其用于软件智能体：一条消息是一个言语行为（动作）加上内容（动作的对象）。FIPA-ACL 填补了 KQML 的空白，并围绕二十种言语行为进行了标准化。

### 二十种 FIPA 言语行为（部分列表）

| 言语行为 | 意图 |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 为真吗？” |
| `query-ref` | “X 的值是多少？” |
| `propose` | “我提议我们做 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意做 X” |
| `refuse` | “我拒绝做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “针对 X 的提案征集” |
| `subscribe` | “当 X 发生变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试了 X 但失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL 消息结构）。重点不在于记住它——而在于其中的每一个都对应于 LLM 协议最终会重新添加的一种原语。

### 标准 FIPA-ACL 消息

```
(
:performative request
:sender agent-a@example.org
:receiver agent-b@example.org
:content ((action (agent-b) (perform-task task-123)))
:language fipa-sl0
:ontology auction-ontology-v1
:reply-with msg-001
:conversation-id conv-007
)
```

七个字段构成了协议信封；一个字段（`content`）承载有效载荷。其余字段正是你每次将重试、线程化和本体论绑定到 JSON 协议时需要重新发明的那些东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999–2020 年代）是使用最广泛的 FIPA 兼容运行时。智能体继承一个基类，交换 ACL 消息，在容器内运行，并使用“行为（behaviors）”进行协调。交互协议库配备了合同网、订阅-通知、条件请求以及提出-接受。

**JACK**（Agent Oriented Software，商业软件）强调在 FIPA 消息基础上进行 BDI（信念-愿望-意图，Belief-Desire-Intention）推理。更形式化，但采用度较低。

两者在 Web 栈吞噬了多智能体用例后都走向衰落。MCP 和 A2A 是 2026 年的运行时“容器”。

### FIPA 为何衰落

- **本体论负担过重。** FIPA 要求共享本体论来解析 `content`。就本体论达成一致是一个长达数年的标准化过程。而网络只是使用 HTTP + JSON。
- **形式语义无人使用。** SL（语义语言）提供了严格的真值条件，但大多数生产系统使用自由格式的内容并忽略了形式化。
- **工具锁定。** JADE 仅支持 Java；JACK 是商业软件。多语言团队都绕过了它们。
- **互联网赢得了协议栈。** REST，然后是 JSON-RPC，接着是 gRPC，取代了 ACL 的传输层。

### LLM 复兴是轻量版 FIPA

将 FIPA 的 `request` 与 MCP 的 `tools/call` 进行比较：

```
// FIPA request
(:performative request :sender agent-a :receiver agent-b :content ... :reply-with r1 :conversation-id c1)

// MCP tools/call (2025)
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": { "name": "calculate" },
  "id": "r1",
  "conversation": "c1"
}
```

相同的信封，不同的语法。两者都携带：谁、对谁、意图、有效载荷、关联 ID。两者之间并无革命性的差异——它们是同一设计的不同权衡。

2025 年 Liu 等人的调查（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”，arXiv:2505.02279）明确指出了这一传承：MCP 对应工具使用的言语行为，A2A 对应智能体对等的言语行为，ACP 对应审计跟踪的言语行为，ANP 对应去中心化身份扩展。新规范是 ACL 的后代，采用 JSON 语法和更宽松的语义。

### 权衡，直截了当地说

**FIPA 提供而现代规范舍弃的：**

- 形式语义——你可以证明 `inform` 意味着发送者相信该内容。
- 规范的言语行为目录——你无需重新争论“我们是否应该有一个 `cancel`？”。
- 数十年的交互协议模式——合同网、订阅-通知、提出-接受——具有已知的正确性属性。

**现代规范提供而 FIPA 没有的：**

- JSON 原生负载，兼容所有现代工具。
- 自然语言内容，LLM 无需手工编写的本体论即可解释。
- Web 栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`、A2A Agent Card）。

更宽松的意图语义使得实现更容易。这正是确切的权衡。

### 值得移植的交互协议

FIPA 提供了大约 15 种交互协议。其中三种值得带入 LLM 多智能体系统：

1. **合同网协议（Contract Net Protocol，CNP）。** 管理者发布 `cfp`（提案征集）；投标者回复 `propose`；管理者接受/拒绝。这是标准的任务市场模式（阶段 16 · 16 谈判）。
2. **订阅/通知（Subscribe/Notify）。** 订阅者发送 `subscribe`；发布者在主题变化时发送 `inform`。这就是 2026 年的每个事件总线。
3. **条件请求（Request-When）。** “当条件 Y 成立时做 X。”带有前置条件的延迟动作。2026 年的对应物是持久化工作流引擎中的延迟任务（阶段 16 · 22 生产环境扩展）。

每一种都能干净地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 放弃本体论会带来什么问题

没有共享本体论，智能体通过自然语言内容推断含义。文档记录的 2026 年失败模式是**语义漂移（semantic drift）**：两个智能体使用同一个词（“customer”）但含义略有不同，接收方智能体基于错误的理解行事，而没有模式验证器能捕捉到这一点。FIPA 的本体论要求会在解析时拒绝该消息。

在不完全采用本体论的情况下的缓解措施：

- 对 `content` 使用 JSON Schema——在传输层拒绝结构错误。
- 类型化工件（A2A）——拒绝错误的模态。
- 在信封中明确言语行为——即使内容是自然语言，也能使意图清晰无歧义。

### 2026 年规范映射到言语行为传承

| 现代规范 | FIPA 类比 | 保留了什么 | 舍弃了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 明确意图、关联 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 明确意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 条件请求 | 异步生命周期、状态转换 | 形式完备性保证 |
| A2A 流式事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写入者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上到下阅读这张表，模式是：保留结构原语，舍弃形式化，让 LLM 来掩盖歧义。

## 动手实践

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 转换器。它编解码标准的 ACL 信封，并展示每个 MCP / A2A 消息形状是如何归结为同样的七个字段的。演示内容包括：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等效形式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 运行一个由一位管理者和三位投标者组成的玩具合同网谈判。

运行：

```
python code/main.py
```

输出是一个并排的跟踪，展示每条现代消息的 2026 JSON 形式和 FIPA-ACL 形式，然后是合同网投标的往返过程。相同的协议原语在往返过程中保持不变；只有语法不同。

## 应用

`outputs/skill-fipa-mapper.md` 是一个技能，可以读取任何智能体协议规范并生成 FIPA-ACL 映射。在采用新协议之前使用它来回答：“这是真正的新东西，还是带有 JSON 语法的 `inform`？”

## 落地

不要带回 FIPA-ACL。但请带回它的检查清单：

- 每条消息的意图原语（言语行为）是什么？
- 是否有用于请求-响应和取消的关联 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型化工件）？
- 交互协议是否是一等公民，还是你在从头实现合同网？
- 当两个智能体对内容含义有分歧时会发生什么（语义漂移）？

在你将任何新协议投入生产之前，记录下这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编解码过程。识别哪个 FIPA 言语行为对应 `tools/call`、`resources/read` 以及 A2A 任务创建。
2. 扩展合同网演示，加入一个 `cancel` 言语行为，让管理者可以在投标进行中撤回任务。`cancel` 解决了哪些仅靠重试无法解决的失败情况？
3. 阅读 FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) 第 4.1–4.3 节。选择一个本课未涉及的言语行为，并描述其现代 JSON-RPC 对应物。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP 中的每一个，列出它们保留和舍弃的 FIPA 言语行为族。
5. 为你自己系统中 `request` 言语行为的 `content` 字段设计一个最小 JSON Schema。该模式能给你什么纯自然语言无法提供的优势，以及它需要付出什么代价？

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|------------|----------|
| 言语行为 (Speech act) | “一种有所作为的话语” | Austin/Searle：话语即行动。ACL 的理论基础。 |
| FIPA | “那个旧的 XML 东西” | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | “代理通信语言” | FIPA 的信封格式：言语行为 + 内容 + 元数据。 |
| 言语行为 (Performative) | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | 知识查询与操作语言（1993）。更简单，范围更窄。 |
| 本体论 (Ontology) | “共享词汇” | 内容语言所谈论概念的形式化定义。 |
| SL0 / SL1 | “FIPA 内容语言” | 语义语言级别 0 和 1——形式化内容语言家族。 |
| 合同网 (Contract Net) | “任务市场” | 管理者发布 cfp，投标者提议，管理者接受。标准交互协议。 |
| 交互协议 (Interaction protocol) | “消息模式” | 一系列具有已知正确性的言语行为：条件请求、订阅-通知等。 |

## 扩展阅读

- [Liu 等人 — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 2025 年将现代规范与 FIPA 传承联系起来的权威调查
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年获批的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的言语行为目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 现代工具使用的对应物 `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 现代智能体对等的对应物：合同网和订阅-通知
