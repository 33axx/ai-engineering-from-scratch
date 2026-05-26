# 结构化输出 — JSON Schema、Pydantic、Zod、受约束解码

> “友好地让模型返回 JSON” 在即便最前沿的模型上也有 5% 到 15% 的失败率。结构化输出通过受约束解码填补了这一差距：模型字面上被禁止发出违反模式的词元。OpenAI 的 strict 模式、Anthropic 的模式化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`，以及 Zod 的 `.parse`，都是同一思想的五种表现形式。本课构建了模式验证器和 strict 模式合约，学习者将在每个生产级提取流水线中使用它们。

**类型：** 构建  
**语言：** Python（标准库，JSON Schema 2020-12 子集）  
**前置条件：** 阶段 13 · 02（函数调用深入探究）  
**时长：** 约 75 分钟  

## 学习目标

- 为提取目标编写一个 JSON Schema 2020-12，使用正确的约束条件（enum、min/max、required、pattern）。
- 解释为何 strict 模式和受约束解码提供的保证与“生成后验证”不同。
- 区分三种失败模式：解析错误、模式违反、模型拒绝。
- 交付一个带有类型化修复和类型化拒绝处理的提取流水线。

## 问题

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。有三种方法。

**方法一：提示生成 JSON。** “请用 JSON 回复，包含字段 customer、line_items、total_usd。”在最具前沿性的模型上，85% 到 95% 的情况下能正常工作。失败有六种方式：缺失大括号、尾随逗号、类型错误、虚构字段、在词元限制处截断、泄漏散文如“这是您的 JSON:”。

**方法二：生成后验证。** 自由生成，解析，根据模式验证，失败时重试。可靠但代价高昂——每次重试都要付费，且截断错误每次出现都会多花费一次交互。

**方法三：受约束解码。** 提供者在解码时强制执行模式。无效词元会从采样分布中被屏蔽掉。输出保证能被解析，且保证通过验证。失败简化为一种模式：拒绝（模型认为输入不符合模式）。

到 2026 年，每个前沿提供者都提供某种形式的第三种方法。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上响应中的 `refusal`（如果模型拒绝回答）。
- **Anthropic。** 对 `tool_use` 输入进行模式强制执行；`stop_reason: "refusal"` 并不存在，但 `end_turn` 且没有工具调用就是信号。
- **Gemini。** 请求级别的 `responseSchema`；到 2026 年，Gemini 对选定类型提供词元级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 输出一个类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod 模式验证提供者输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点：声明一次模式，端到端强制执行。

## 概念

### JSON Schema 2020-12 — 通用语言

每个提供者都接受 JSON Schema 2020-12。你最常用的结构：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的闭合集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子模式。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI strict 模式增加了三个要求：每个属性都必须列在 `required` 中，所有地方必须设置 `additionalProperties: false`，并且不能有未解析的 `$ref`。如果违反这些要求，API 会在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从类似数据类的模型生成 JSON Schema。Pydantic AI 封装了它，因此你可以编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

并且智能体框架会将模式转换为边缘节点的 OpenAI strict 模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。验证错误会引发带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 中的等效方案。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，它转换为 API 的 JSON Schema 载荷。

### 拒绝

Strict 模式无法强制模型回答。如果输入不符合模式（“邮件是一首诗，而不是发票”），模型会发出一个包含原因的 `refusal` 字段。你的代码必须将此作为一等结果处理，而不是失败。拒绝也作为安全信号很有用：当模型被要求从受保护内容的电子邮件中提取信用卡号时，它会返回一个带有安全原因的拒绝。

### 开源中的受约束解码

开放权重的实现使用了三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从模式构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的词元的对数几率。
2. **带 JSON 解析器的对数几率屏蔽**：与模型同步运行流式 JSON 解析器；在每一步，计算有效下一词元集合。
3. **带验证器的投机解码**：便宜的草稿模型提出词元，验证器强制执行模式。

商业提供者在幕后选择其中之一。到 2026 年，对于短结构化输出，最新技术的速度比普通生成更快，对于长输出则大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。在 strict 模式下不可能发生。在非 strict 提供者中仍可能发生。
2. **模式违反。** 输出可以解析但违反了模式。在 strict 模式下不可能发生。在非 strict 模式中很常见。
3. **拒绝。** 模型拒绝回答。必须作为类型化结果处理。

### 重试策略

当你在 strict 模式之外（Anthropic 工具使用、非 strict OpenAI、较旧的 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常就足够了。三次重试可以捕获弱模型的偶尔失误。超过三次是模式糟糕的标志：模型无法针对某些输入满足模式，提示或模式需要修复。

### 小模型支持

受约束解码对小型模型有效。一个 3B 参数的开源模型配合语法强制，在结构化任务上胜过 70B 参数的原始提示模型。这就是结构化输出在生产中重要的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 提供了一个基于标准库的最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` 模式，并通过验证器运行一个虚拟的 LLM 输出，演示了解析错误、模式违反和拒绝路径。在生产中，将虚拟输出替换为任何提供者的真实响应。

关注点：

- 验证器返回一个类型化的 `[ValidationError]` 列表，包含路径和消息。这是你想要暴露给重试提示的形状。
- 拒绝分支不会重试。它会记录并返回一个类型化的拒绝。阶段 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查会在对抗性测试输入上触发，展示了为什么 strict 模式能阻止虚构字段。

## 交付它

本课程生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、工单、简历等），该技能将生成一个符合 strict 模式兼容性的 JSON Schema 2020-12 以及一个镜像它的 Pydantic 模型，并附带类型化拒绝和重试处理存根。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器使用 `minimum` 约束路径将其拒绝。

2. 扩展验证器以支持带鉴别器的 `oneOf`。常见情况：`line_item` 是产品或服务，由 `kind` 标记。Strict 模式对此有微妙规则；请查阅 OpenAI 的结构化输出指南。

3. 将相同的 Invoice 模式编写为 Pydantic BaseModel，并将 `model_json_schema()` 的输出与你手动编写的模式进行比较。找出 Pydantic 默认设置而手动版本忽略的一个字段。

4. 测量拒绝率。构造十个不应被提取的输入（歌词、数学证明、空白邮件）并通过真实提供者运行它们，使用 strict 模式。计数拒绝与虚构输出。这是你进行拒绝感知重试的基准事实。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出一个它在 strict 模式中明确禁止但普通 JSON Schema 允许的结构。然后设计一个非必需使用该禁止结构的模式，并将其重构为 strict 兼容。

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “模式规范” | 每个现代提供者都支持的 IETF 草案模式方言 |
| Strict 模式 | “保证的模式” | OpenAI 标志，通过受约束解码强制执行模式 |
| 受约束解码 | “对数几率屏蔽” | 解码时强制执行，屏蔽无效的下一词元 |
| 拒绝 | “模型拒绝回答” | 当输入无法符合模式时的类型化结果 |
| 解析错误 | “无效 JSON” | 输出未解析为 JSON；在 strict 模式下不可能 |
| 模式违反 | “形状错误” | 已解析但违反类型/required/enum/范围 |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI strict 模式中要求 |
| Pydantic BaseModel | “类型化输出” | 生成并验证 JSON Schema 的 Python 类 |
| Zod 模式 | “TypeScript 输出类型” | 用于提供者输出验证的 TS 运行时模式 |
| 语法强制 | “开放权重受约束解码” | 基于 FSM 的对数几率屏蔽，如 outlines / guidance 中的做法 |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — strict 模式、拒绝和模式要求
- [OpenAI — 在 API 中引入结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布博文，解释解码保证
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 类型化 output_type 绑定，序列化到每个提供者
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和 strict 模式注意事项
