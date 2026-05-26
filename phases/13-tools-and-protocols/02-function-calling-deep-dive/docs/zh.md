# 函数调用深度解析 — OpenAI、Anthropic、Gemini

> 2024 年，三家前沿提供商统一了工具调用循环，但在其他方面却分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联。本课程并排对比这三种格式的差异，确保在某一提供商上运行的代码在移植时不会出错。

**类型：** 构建
**语言：** Python（标准库、模式转换器）
**前置条件：** 阶段 13 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 阐述 OpenAI、Anthropic 和 Gemini 函数调用载荷（声明、调用、结果）之间的三个形状差异。
- 将一条工具声明转换为三种提供商格式，并预测严格模式约束在何处存在差异。
- 使用每个提供商的 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的具体硬限制（工具数量、模式深度、参数长度）以及违反限制时返回的错误特征。

## 问题

函数调用请求的形状因提供商而异。以下是 2026 年生产栈中的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是需要你解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码来强制模式合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已被解析（是一个对象，而非字符串）。你通过包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息进行回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用的关联。你通过 `{functionResponse: {name, id, response}}` 进行回复。

相同的循环，不同的字段名称、不同的嵌套方式、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写的天气代理团队，移植到 Anthropic 需要两天，再移植到 Gemini 又需要一天，仅仅为了处理管道差异。

本课程构建一个转换器，将三种格式统一成一种规范的工具声明，并在边缘进行路由。阶段 13 · 17 将相同的模式泛化为一个 LLM 网关。

## 概念

### 通用结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择。** 强制使用特定工具、禁止使用工具，或让模型决定。
3. **调用输出。** 带有工具名称和参数的结构化输出。
4. **调用 ID。** 将响应与正确的调用关联（对于并行调用很重要）。
5. **结果注入。** 将结果与调用绑定在一起的消息或块。

### 形状差异，逐字段比较

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明外壳 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 解析后的对象 | 解析后的对象 |
| ID 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3 以上） |
| 结果块 | 角色 `tool`，`tool_call_id` | 包含 `tool_result` 和 `tool_use_id` 的 `user` | 带有匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | 模式即模式（始终强制） | 请求级别的 `responseSchema` |

### 你实际会遇到的上限

- **OpenAI。** 每次请求最多 128 个工具。模式深度为 5。参数字符串长度不超过 8192 字节。严格模式要求不能有 `$ref`、不能有重叠的 `oneOf`/`anyOf`/`allOf`，且每个属性都必须列在 `required` 中。
- **Anthropic。** 每次请求最多 64 个工具。模式深度实际上没有限制，但实际限制为 10。没有严格模式标志；模式是一种契约，模型倾向于遵守。
- **Gemini。** 每次请求最多 64 个函数。模式类型是 OpenAPI 3.0 的子集（与 JSON Schema 2020-12 略有不同）。自 Gemini 3 起，并行调用使用唯一 ID。

### `tool_choice` 行为

所有三种提供商都支持三种模式，但名称不同。

- **Auto。** 模型选择工具或文本。默认值。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

每个提供商还有一个独有的模式：

- **OpenAI。** 按名称强制使用特定工具。
- **Anthropic。** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志用于区分单次和多次。
- **Gemini。** `mode: "VALIDATED"` 强制每次响应都通过模式验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一次助手消息中发出多个调用。你运行所有调用，并用一个包含每个 `tool_call_id` 对应条目的批处理工具角色消息进行回复。Anthropic 历史上只进行单次调用；`disable_parallel_tool_use: false`（自 Claude 3.5 起为默认）启用多次调用。Gemini 2 允许并行调用但没有稳定的 ID；Gemini 3 添加了 UUID，因此乱序响应也能干净地关联。

### 流式处理

所有三种提供商都支持流式工具调用。线缆格式有所不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你需要累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始 / 块增量 / 块停止事件。`input_json_delta` 块承载部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，使得多个并行调用可以交错传输。

阶段 13 · 03 深入探讨了并行 + 流式重组的细节。本课程聚焦于声明和单次调用的形状。

### 错误与修复

无效参数错误的表现也各不相同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误信息并重新调用。
- **OpenAI（严格）。** 验证在解码期间进行；无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；模式只是建议性的。需要在服务端进行验证。
- **Gemini。** OpenAPI 3.0 的古怪之处：对象字段上的 `enum` 被静默忽略；需要自行验证。

### 转换器模式

在你的代码中，规范的工具声明看起来像这样（你可以选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种提供商的形状。`code/main.py` 中的 harness 正是这样做的，然后通过每个提供商的响应形状对假的工具调用进行往返处理。无需网络 —— 本课程只教授形状，而非 HTTP。

生产团队将这个转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。阶段 13 · 17 提供了一个网关，它在任意三种提供商前面暴露一个 OpenAI 形状的 API。

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类，以及三个输出 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。然后，它将每个提供商的人工构造响应形状解析为相同的规范调用对象，证明这些语义在底层是相同的。运行它，并排比较三个声明。

需要关注的内容：

- 三个声明块仅在外壳和字段名称上不同。
- 三个响应块的区别在于调用的位置（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付它

本课程生成 `outputs/skill-provider-portability-audit.md`。给定一个针对某提供商的函数调用集成，该技能会生成一份可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、以及移植到其他提供商时会出现什么问题。

## 练习

1. 运行 `code/main.py` 并验证三个提供商声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具以添加一个枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 的古怪之处。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，该解析器提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 本身没有这个功能；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商的形状。然后映射 `mode="any"` 和 `mode="none"`。查阅课程的差异表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。在其模式规范中找到一个其他两个提供商不支持的字段。候选：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明模式的工具调用。将其通过每个提供商的验证器（第 01 课中的标准库验证器可作为代理）运行，并记录哪些错误被触发。记录在生产中你会因为严格性选择使用哪个提供商。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 函数调用 | "工具使用" | 提供商级别的 API，用于结构化工具调用输出 |
| 工具声明 | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必须 / 无 / 指定名称等模式 |
| 严格模式 | "模式强制" | OpenAI 的标志，用于约束解码以匹配模式 |
| `tool_use` 块 | "Anthropic 的调用形状" | 包含 id、name、input 的内联内容块 |
| `functionCall` 部分 | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数字符串化 | "字符串化 JSON" | OpenAI 返回参数作为 JSON 字符串，而非对象 |
| 并行工具调用 | "单次轮询扇出" | 一次助手消息中的多个工具调用 |
| 拒绝 | "模型拒绝" | 严格模式下代替调用的拒绝块 |
| OpenAPI 3.0 子集 | "Gemini 模式古怪之处" | Gemini 使用一种类似 JSON Schema 的方言，有细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块的语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 ID 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级界面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式模式强制详情
