# 结构化输出与约束解码

> 让大模型输出 JSON。大部分时候能拿到 JSON。但在生产环境中，“大部分”就是问题所在。约束解码通过在采样前修改 logits，把“大部分”变成“总是”。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段5·17（聊天机器人），阶段5·19（子词分词）  
**时长：** 大约60分钟

## 问题

一个分类器提示大模型：“返回 {positive, negative, neutral} 中的一个。”模型返回：“The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...”。你的解析器崩溃了。你的分类器的 F1 变成 0.0。

自由形式生成不是契约，而是建议。生产系统需要契约。

到2026年，有三个层次。

1. **提示工程。** 礼貌地请求。“只返回 JSON 对象。” 前沿模型上大约 80% 有效，较小模型上更低。  
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具使用、Gemini JSON 模式。在支持的 schema 上可靠。受供应商锁定。  
3. **约束解码。** 在每个生成步骤修改 logits，使模型*无法*产生无效词元。构造上保证 100% 有效。适用于任何本地模型。

本课程构建对这三种方法的直觉，并说明何时使用哪一种。

## 概念

![在每个步骤屏蔽无效词元的约束解码](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤，LLM 生成了一个覆盖整个词表（约 100k 个词元）的 logit 向量。一个 *logit 处理器* 位于模型和采样器之间。它根据目标文法（JSON Schema、正则表达式、上下文无关文法）中的当前位置，计算哪些词元是有效的，并将所有无效词元的 logits 设为负无穷。对剩余 logits 进行 softmax，将概率质量仅放在有效的延续上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译成有限状态机。每个词元得到 O(1) 的有效下一词元查找。基于 FSM，因此递归 schema 需要展开。  
- **XGrammar / llguidance。** 上下文无关文法引擎。处理递归 JSON Schema。解码开销几乎为零。OpenAI 在其 2025 年结构化输出实现中引用了 llguidance。  
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，后端通过 Outlines、XGrammar 或 lm-format-enforcer 实现。  
- **Instructor。** 基于 Pydantic 的封装，适用于任何 LLM。验证失败时重试。跨供应商，但不修改 logits — 它依赖于重试和感知结构化输出的提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。原因有二。首先，它缩小了下一个词元的搜索空间。其次，巧妙的实现在强制词元（如支架 `{"name": "` — 每个字节都已确定）处完全跳过词元生成。

### 会付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型在思考之前就做出回答。JSON 是有效的。答案是错的。没有验证能捕获这一点。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序关乎逻辑，而非格式。

## 构建

### 步骤 1：从头实现正则约束生成

参见 `code/main.py` 获得一个独立的 FSM 实现。核心思想在 30 行内：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们已经满足文法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词表词元可以使 FSM 前进而不离开一条接受路径。

### 步骤 2：Outlines 用于 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出不可达。

### 步骤 3：Instructor 用于供应商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不接触 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商可移植性是卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对于受支持的 schema，可靠性等同于 Outlines。无需管理本地模型。将你锁定在供应商上。

## 陷阱

- **递归 schema。** Outlines 将递归展开到固定深度。树结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。  
- **大型枚举。** 10000 个选项的枚举编译缓慢或超时。改用检索器：先预测 top-k 候选，再约束到这些候选。  
- **文法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则，模型无法为缺失的日期输出 `"unknown"`。模型将发明一个日期来补偿。允许 `null` 或一个哨兵值。  
- **过早承诺。** 参见上方字段顺序陷阱。始终把推理放在前面。  
- **供应商 JSON 模式不带 schema。** 纯 JSON 模式仅保证有效 JSON，不保证*对于你的用例*有效。始终提供完整 schema。

## 使用

2026 年的选型：

| 情形 | 选择 |
|--------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines (FSM) |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 可接受重试的批处理 | Instructor + 最便宜模型 |

## 交付

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 用一个小的开放权重模型（例如 Llama-3.2-3B）不加约束解码，提示输出 `Review(sentiment, confidence, evidence_span)`。在 100 条评论中测量解析为有效 JSON 的比例。  
2. **中等。** 对同一语料使用 Outlines JSON 模式。比较合规率、延迟和语义准确性。  
3. **困难。** 从零实现一个针对电话号码的正则约束解码器（`\d{3}-\d{3}-\d{4}`）。验证 1000 个样本上 0 个无效输出。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|--------------|----------|
| Constrained decoding | 强制有效输出 | 在每个生成步骤屏蔽无效词元的 logits。 |
| Logit processor | 进行约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的文法表示；O(1) 有效下一词元查找。 |
| CFG | 上下文无关文法 | 处理递归的文法；比 FSM 慢但表达能力更强。 |
| Schema field order | 有关系吗？ | 是的 — 第一个字段会提交；始终把推理放在回答之前。 |
| Guided decoding | vLLM 对这个概念的命名 | 相同的概念，集成到推理服务器中。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON 语法；**不**保证 schema 匹配。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。  
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速基于 CFG 的约束解码。  
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。  
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考和注意事项。  
- [Instructor library](https://python.useinstructor.com/) — Pydantic + 跨供应商重试。  
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 六个约束解码框架的基准测试。
