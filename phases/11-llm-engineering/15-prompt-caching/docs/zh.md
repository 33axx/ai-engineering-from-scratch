# 提示缓存与上下文缓存

> 你的系统提示词是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送这两者。并且每次都为它们付费。提示缓存让提供商在它们那端保持该前缀的热度，并在重用时向你收取正常价格的 10%。使用得当，它可以将推理成本降低 50–90%，并将首 token 延迟降低 40–85%。

**类型：** 构建
**语言：** Python
**前置要求：** 第 11 阶段 · 01（提示词工程），第 11 阶段 · 05（上下文工程），第 11 阶段 · 11（缓存与成本）
**时间：** ~60 分钟

## 问题

一个编码智能体在对话的每一轮都向 Claude 发送相同的 15,000 token 系统提示词。以 $3/M 输入 token 计算，二十轮对话的输入成本就达到了 $0.90 —— 这还没算用户实际发送的任何消息。如果乘以每天 10,000 次对话，单是为这些从不改变的文字，账单就会达到 $9,000/天。

你无法缩减提示词而不损害质量。你无法避免发送它 —— 模型在每一轮都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这个办法就是提示缓存。Anthropic 在 2024 年 8 月推出了它（并在 2025 年推出了 1 小时扩展 TTL 的变体），OpenAI 在那年晚些时候实现了自动化，Google 随着 Gemini 1.5 一起推出了显式上下文缓存，现在这三家都在它们的前沿模型上将其作为一等地（first-class）特性提供。

## 概念

![提示缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**其机制。** 当请求的前缀与近期请求的匹配时，提供商会从之前的运行中提供 KV 缓存，而不是重新对 token 进行编码。你第一次支付少量的写入溢价，之后每次使用都享受大幅的读取折扣。

**2026 年的三种提供商风格。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入价格打 10% 折扣 | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 token (Sonnet/Opus), 2,048 token (Haiku) |
| OpenAI | 自动前缀检测 | 输入价格打 50% 折扣 | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取约为正常价格的 25% | 按 token·小时收取存储费 | 用户设定（默认 1 小时） | 4,096 token (Flash), 32,768 token (Pro) |

**不变原则。** 所有三家都只缓存前缀。如果请求之间的任何 token 不同，则第一个不同 token 之后的所有内容都会缓存未命中。将*稳定*的部分放在顶部，将*可变*的部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反顺序 —— 将用户消息放在系统提示词之上，在少样本示例之间穿插动态检索 —— 则缓存永远不会命中。

### 盈亏平衡计算

Anthropic 25% 的写入溢价意味着一个缓存块至少需要被读取两次才能实现净节省。1 次写入 + 1 次读取平均每次请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取平均成本为 0.205 倍（节省 80%）。经验法则：缓存任何你预计在 TTL 内至少被重用 3 次的内容。

## 构建它

### 步骤 1：使用显式标记的 Anthropic 提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在此窗口内的重用会命中；过期后的重用则会再次写入。

**响应使用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查这两个字段 —— 如果在多次请求中 `cache_read_input_tokens` 一直为零，则你的缓存键正在漂移。

### 步骤 2：一小时扩展 TTL

对于长时间运行的批处理任务，5 分钟的默认 TTL 会在任务之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2 倍（比基准价高 50%，而不是 25%），但在任何重用前缀超过 5 次的批处理上都能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 无需你进行任何配置。任何超过 1,024 个 token 且与近期请求匹配的前缀都会自动享受 50% 的折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样适用缓存友好布局规则。有两件事会破坏 OpenAI 的缓存（但不会破坏 Anthropic 的）：更改 `user` 字段（用作缓存键组件）以及重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为一个你可以创建和命名的一等地（first-class）对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时收取存储费，只要缓存存在就持续计费，读取费用约为正常输入价格的 25%。当你需要在数天内跨多个会话重用同一个巨大提示时，这种形式最为合适。

### 步骤 5：在生产环境中测量命中率

请参阅 `code/main.py`，查看一个模拟的三提供商会计程序，用于跟踪写入/读取/未命中次数，并计算每 1K 请求的混合成本。根据目标命中率来门控部署 —— 大多数生产环境的 Anthropic 设置在工作预热后，读取占比应超过 80%。

## 2026 年仍然存在的陷阱

- **顶部的动态时间戳。** 在系统提示词顶部使用 `"Current time: 2026-04-22 15:30:02"`。每个请求都会未命中。将时间戳移到缓存断点之下。
- **工具重新排序。** 以稳定顺序序列化工具 —— 部署之间的字典重排会破坏每一次命中。
- **自由文本的近似重复。** "You are helpful." 与 "You are a helpful assistant." —— 一个字节的差异 = 完全未命中。
- **块太小。** Anthropic 强制要求 1,024 个 token 的下限（Haiku 为 2,048 个 token）。较小的块会静默不缓存。
- **盲目的成本仪表板。** 将 "输入 token" 拆分为缓存与未缓存。否则，流量下降看起来会像缓存赢了一样。

## 使用它

2026 年的缓存栈：

| 情况 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示词、多轮交互的智能体 | Anthropic `cache_control` 配合 5 分钟 TTL |
| 重用前缀超过 30 分钟的批处理任务 | Anthropic 配合 `ttl: "1h"` |
| 依赖于 GPT-5 的无服务器端点，无自定义基础设施 | OpenAI 自动缓存（只需确保你的前缀稳定且足够长） |
| 持续数天重用大型代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持跨提供商的可缓存前缀布局一致，以便任何命中都能生效 |

将其与语义缓存（第 11 阶段 · 11）结合用于用户消息层：提示缓存处理 *token 完全相同* 的重用，语义缓存处理*含义相同* 的重用。

## 交付

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 以一个包含 10 轮对话和 5,000 token 系统提示词的对话为例，分别针对 Claude 在不使用 `cache_control` 和使用了的情况下运行它。报告每次的输入 token 账单。
2. **中等。** 编写一个测试框架，给定一个提示模板和一个请求日志，计算每个提供商（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和美元节省量。
3. **困难。** 构建一个布局优化器：给定一个提示和一个标记为 `stable=True/False` 的字段列表，重写提示，将单个缓存断点放置在最有利于缓存的位置，且不丢失信息。在真实的 Anthropic 端点上进行验证。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 提示缓存 | "让长提示变便宜" | 重用提供商的 KV 缓存来匹配前缀；对重复的输入 token 提供 50-90% 的折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明 "直到此处的所有内容都是可缓存的"；`{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 首次填充缓存的请求；在 Anthropic 上按约 1.25 倍输入费率计费，在 OpenAI 上免费。 |
| 缓存读取 | "享受折扣" | 匹配前缀的后续请求；按 10%（Anthropic）、50%（OpenAI）、~25%（Gemini）计费。 |
| TTL | "存活时间" | 缓存保持热状态的秒数；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力最长 1 小时，Gemini 由用户设定。 |
| 扩展 TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价 2 倍，但对于批处理重用来说是值得的。 |
| 前缀匹配 | "为什么我的缓存未命中" | 只有当从开头到断点处的每个 token 都字节完全相同时，缓存才会命中。 |
| 上下文缓存 (Gemini) | "显式的那种" | Google 的命名、按存储计费的缓存对象；最适合持续数天重用大型语料库的场景。 |

## 延伸阅读

- [Anthropic — 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1小时 TTL, 盈亏平衡表。
- [OpenAI — 提示缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic 工程博客 — 长上下文工作负载的提示缓存](https://www.anthropic.com/news/prompt-caching) — 原始发布文章，包含延迟数据。
- 第 11 阶段 · 05（上下文工程）—— 如何在何处分割提示以便缓存生效。
- 第 11 阶段 · 11（缓存与成本）—— 将提示缓存与用户消息上的语义缓存配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示缓存所暴露给用户的 KV 缓存内存模型；解释了为何缓存的前缀重新读取比重新计算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 预填充阶段是提示缓存所跳过的阶段；本文解释了为什么缓存命中时 TTFT 急剧下降，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示缓存与推测解码、Flash Attention 和 MQA/GQA 并列，是弯曲推理成本曲线的杠杆；请阅读本文以了解另三种。
