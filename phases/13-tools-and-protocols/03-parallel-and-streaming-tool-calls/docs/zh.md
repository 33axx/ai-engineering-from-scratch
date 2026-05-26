# 并行工具调用与流式工具使用

> 三次独立天气查询串行化需要三个往返。并行执行后，总时延缩小为最慢的单次调用。当前每个前沿模型都支持在单轮中发出多个工具调用。收益真实可见；底层实现却颇为微妙。本课涵盖两方面：并行扇出与流式参数重组，重点在于标识符关联这一陷阱。

**类型：** 构建
**语言：** Python（标准库、线程池 + 流式处理框架）
**前置知识：** 阶段 13 · 02（函数调用深入教程）
**时长：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 的存在原因及其何时需要禁用。
- 在并行扇出过程中将流式参数块关联到正确的工具调用标识符。
- 在不提前解析的情况下将部分 `arguments` 字符串重组为完整 JSON。
- 运行一个三城市天气基准测试，演示串行与并行的延迟差异。

## 问题所在

没有并行调用时，回答“班加罗尔、东京、苏黎世天气如何”的智能体执行如下：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还需计算执行器延迟。总耗时约为理想墙钟时间的 4 倍。

使用并行调用后：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间为三者最大值，而非三者之和。在 OpenAI、Anthropic 和 Gemini 的生产基准测试中，扇出式工作负载的墙钟时间减少 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型对齐。当结果流式传输时，你必须在执行前将部分参数片段组装成完整 JSON。Gemini 3 新增唯一标识符正是为了解决一个实际问题：对同一工具的两次并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI.** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic.** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上默认）。设为 `true` 则串行。
- **Gemini.** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

在以下情况禁用并行：工具存在顺序依赖（`create_file` 后跟 `write_file`）、一次调用的输出影响另一次调用的输入、或者速率限制器无法处理扇出。

### 标识符关联

模型发出的每个调用都有一个 `id`。主机返回的每个结果都必须包含相同的 `id`。缺少此信息，结果将不明确。

- **OpenAI.** 每个工具角色消息上的 `tool_call_id`。
- **Anthropic.** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini.** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，导致同名并行调用出错）。

### 并发执行调用

主机在自己的线程、协程或远程工作者上运行每个调用的执行器。最简单的框架使用线程池；生产环境使用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序不可预测——标识符才是关键。

一个常见错误：按调用列表顺序而非完成顺序回复结果。这通常可行，因为模型只关心 `tool_call_id`，但如果某个结果被丢弃或重复，乱序提交会使调试更困难。建议按完成顺序回复并显式附带标识符。

### 流式工具调用

当模型流式输出时，`arguments` 以碎片形式到达。三个并行调用的三个独立流块会在线路上交织。你需要为每个标识符维护一个累加器。

各供应商的形状：

- **OpenAI.** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。按索引累加，首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic.** 流事件包括 `message_start`，然后每个块有一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空的 input）。`content_block_delta` 事件携带 `input_json_delta` 碎片。`content_block_stop` 关闭每个块。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 及以上）发送带有 `functionCallId` 的碎片，使调用能干净地交织。Gemini 3 之前，流式返回一次一个完整调用。

### 部分 JSON 与过早解析陷阱

在 `arguments` 完整之前无法解析。部分 JSON 如 `{"city": "Beng` 不是有效 JSON 会引发错误。正确的触发条件是供应商提供的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只在此之后才尝试 `json.loads`。更健壮的方法是使用增量 JSON 解析器，在结构完成时产生事件；OpenAI 的流式指南推荐此方法用于显示实时“思考”指示器的用户体验。花括号计数作为完整性检查不可靠（引号内或转义内容中的花括号会导致误报），只应作为非正式调试启发式。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

主机的回复仍然需要引用标识符：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对于 OpenAI 和 Anthropic，回复顺序不影响正确性。Gemini 接受任何顺序，只要标识符匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的框架模拟三个执行器，延迟分别为 400、600 和 800 毫秒。串行运行总计 1800 毫秒。并行运行耗时 max(400, 600, 800) = 800 毫秒。差异是恒定的而非比例的，因此随着工具数量增加，节省更多。

实际注意：并行调用会给下游 API 带来压力。对受速率限制的服务进行 10 路扇出会失败。阶段 13 · 17 涵盖网关级背压；重试语义计划在未来阶段实现。

### 流式扇出墙钟时间

如果模型本身流式输出，你可以在一组调用的参数完整后立即开始执行，而不必等待所有调用完成。这是 OpenAI 文档提到的一种优化，但并非所有 SDK 都公开。本课的框架实现了这一点：一旦模拟流产生完整的参数对象，主机就启动该调用。

## 使用方式

`code/main.py` 包含两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 分别以串行和并行方式运行三个模拟天气调用，并打印墙钟时间。第二部分重放一个伪造的流式响应——三个并行调用的 `arguments` 碎片在线路上交织——并使用 `StreamAccumulator` 按标识符重组。无需 LLM、无网络，仅重组逻辑。

需要关注的点：

- 串行计时器显示 1.8 秒。并行计时器使用相同的伪造延迟显示 0.8 秒。
- 累加器通过按标识符缓冲碎片来处理乱序到达的块，并且仅当每个调用的 JSON 完整时才解析。
- 执行器在某个标识符的参数完成时立即启动，而不是等待所有流结束。

## 交付物

本课生成 `outputs/skill-parallel-call-safety-check.md`。给定工具注册表，该技能审核哪些工具可以安全并行化、哪些具有顺序依赖、哪些会压垮下游速率限制——并返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比值约等于 `max/sum`（由于线程调度、序列化和框架开销，真实运行会与理想值略有偏差）。在什么延迟分布下，并行不再有意义？

2. 扩展累加器以处理“调用在中途被取消”的情况：丢弃其缓冲区并发出 `cancelled` 事件。哪个供应商显式记录了这种情况？查看 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 将线程池替换为 `asyncio.gather`。对两者进行基准测试。你会看到异步有小幅收益（因为上下文切换成本更低），但前提是执行器执行真实 I/O。

4. 选择两个不应并行化的工具（例如 `create_file` 后跟 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并基于该图控制并行扇出。这是依赖感知调度的最小机制，未来智能体工程阶段将对其进行形式化。

5. 阅读 OpenAI 的并行函数调用部分和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的一种真实世界工具类型。（提示：对同一资源的连锁突变。）

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|--------------|----------|
| 并行工具调用 | “一轮内的扇出” | 模型在单条助手消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的标志” | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | “Anthropic 的反向标志” | 选择退出标志；默认启用并行 |
| 工具调用标识符 | “关联句柄” | 每个调用的标识符，结果消息必须回显 |
| 累加器 | “流缓冲区” | 每个标识符的字符串缓冲区，用于接收部分 `arguments` 碎片 |
| 乱序完成 | “最快的先完成” | 并行调用以不可预测的顺序结束；标识符是粘合剂 |
| 依赖图 | “顺序约束” | 工具输出作为其他工具输入的情况；不可并行化 |
| 过早解析陷阱 | “JSON.parse 崩溃了” | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 每个调用带有唯一标识符的流式参数碎片 |
| 按完成顺序回复 | “不要等待全部” | 结果到达后立即按标识符回复 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为及选择退出标志
- [Anthropic — 工具使用：实现工具使用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批处理
- [Google — Gemini 函数调用并行部分](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的标识符关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的碎片化参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
