# 聊天机器人——从基于规则到神经网络再到LLM智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证。每个时代都解决了前一个时代最严重的失败。

**类型：** 学习  
**语言：** Python  
**先修知识：** 阶段 5·13（问答），阶段 5·14（信息检索）  
**时长：** 约 75 分钟  

## 问题

用户说“我想改签我的航班”。系统必须弄清楚他们想要什么，缺少哪些信息，如何获取这些信息，以及如何完成操作。然后用户说“等等，如果我取消会怎样？”系统必须记住上下文，切换任务，并保持状态。

对话对于机器学习系统来说很难。输入是开放式的。输出必须在多轮对话中保持连贯。系统可能需要与世界交互（改签航班、扣款）。每一步错误都对用户可见。

聊天机器人架构经历了四个范式，每个都是因为前一个范式失败得太明显而被引入的。2026 年的生产环境是最后两个范式的混合体。

## 概念

![聊天机器人演进：基于规则 → 检索 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器将输入路由到预定义流程。槽填充状态机收集所需信息。在其设计的狭窄范围内表现出色。超出范围立即失败。在无法容忍幻觉的安全关键领域（银行认证、航班预订）中仍然使用。

**基于检索。** 类似 FAQ 的系统。编码每一对（话语，回复）。运行时，对用户消息进行编码并检索最接近的存储回复。类似于 Zendesk 经典的“相似文章”功能。比规则更好地处理同义改写。不生成，所以没有幻觉。

**神经网络（序列到序列）。** 在对话日志上训练的编码器-解码器。从头生成回复。流畅但容易产生通用输出（“我不知道”）和事实偏差。从未可靠地保持主题。这就是谷歌、Facebook 和微软在 2016-2019 年都推出令人失望的聊天机器人的原因。

**LLM 智能体。** 语言模型包装在一个循环中，该循环进行规划、调用工具并验证结果。不是带有长提示的聊天机器人。而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。基于检索的接地（RAG）防止它产生幻觉。工具调用让它实际执行操作。这是 2026 年的架构。

这四个范式并不是顺序替代的。2026 年的生产聊天机器人会通过所有四个范式路由：基于规则用于认证和破坏性操作，基于检索用于常见问题，神经网络生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 构建它

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA 用 20 行代码实现。反射技巧（“我感到难过” → “你为什么感到难过”）是 Weizenbaum 1966 年经典的心理治疗师演示。仍然具有启发性。

### 第 2 步：基于检索（FAQ）

这个说明性片段需要 `pip install sentence-transformers`（会拉入 torch）。本课可运行的 `code/main.py` 使用标准库的 Jaccard 相似度代替，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级。

### 第 3 步：神经网络生成（基线）

使用小型指令微调编码器-解码器（FLAN-T5）或微调的对话模型。2026 年在生产环境中单独使用不可靠（矛盾、偏离主题、事实无意义），但在混合系统中用于自然措辞。DialoGPT 风格的纯解码器模型需要显式的轮次分隔符和 EOS 处理才能生成连贯回复；FLAN-T5 的 text2text 管道开箱即用，适合教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要命名的三件事。工具是 LLM 可以调用的可执行函数。循环在 LLM 返回最终答案而非工具调用时终止。步骤预算防止在模糊任务上无限循环。

真实生产环境还需添加：基于检索的接地（每次 LLM 调用前注入相关文档）、护栏（拒绝未经确认的破坏性操作）、可观测性（记录每一步）和评估（自动检查智能体行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：对任何破坏性操作使用确定性规则，对预定义常见问题使用检索，对其他所有情况使用 LLM 智能体。这就是 2026 年客户支持系统的实际形态。

## 使用它

2026 年技术栈：

| 用例 | 架构 |
|---------|---------------|
| 预订、支付、认证 | 基于规则的状态机 + 槽填充 |
| 客户支持常见问题 | 检索精选答案 |
| 开放式帮助聊天 | LLM 智能体 + RAG + 工具调用 |
| 内部工具 / IDE 助手 | LLM 智能体 + 工具调用（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 微调 LLM + 角色系统提示 + 知识检索 |

在生产环境中始终使用混合路由。没有单一架构能很好地处理所有请求。路由层本身通常是一个小型意图分类器。

## 仍然存在的失败模式

- **自信的捏造。** LLM 智能体声称执行了未实际执行的操作。缓解措施：验证结果、记录工具调用、绝不允许 LLM 在未成功返回工具结果的情况下声称完成了某事。
- **提示注入。** 用户插入覆盖系统提示的文本。在 2025 年 OWASP LLM 应用十大风险中排名 LLM01。两种类型：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击率因场景而异。在通用工具使用和编码基准测试中，前沿模型的实测成功率大约在 0.5-8.5% 之间。特定高风险设置（针对 AI 编码智能体的自适应攻击、易受攻击的编排）已达到约 84%。生产环境中的 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——由攻击者控制的电子邮件触发的 Microsoft 365 Copilot 零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；在工具调用前进行清理；将工具输出与主提示隔离；使用计划-验证-执行（PVE）模式，智能体先规划，然后验证每个操作是否匹配该计划，之后再执行（这会阻止工具结果注入新的非计划操作）；对破坏性操作要求用户确认；对工具范围应用最小权限原则。

  再多的提示工程也无法完全消除此风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体因工具调用返回了边缘相关信息而偏离任务。缓解措施：缩小工具契约；保持系统提示集中；添加离题率评估。
- **无限循环。** 智能体不断调用同一个工具。缓解措施：步骤预算、工具调用去重、LLM 裁判评估“是否在取得进展”。
- **上下文窗口耗尽。** 长对话将最早几轮推出上下文。缓解措施：总结较早轮次、通过相似性检索相关过去轮次、或使用长上下文模型。

## 交付它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 为咖啡店点单机器人用 10 个模式实现上述基于规则的 respond 函数。测试边缘情况：重复订单、修改、取消、未明确的意图。
2. **中等。** 构建一个混合常见问题 + LLM 回退系统。50 个 SaaS 产品的常见问题条目，LLM 回退带有文档网站检索。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。在 50 个测试场景（包括提示注入尝试）上进行评估。报告离题率、失败任务率以及任何注入成功率。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | 分类标签（book_flight, reset_password）。路由到处理程序。 |
| Slot | 一条信息 | 机器人需要的参数（日期、目的地）。槽填充是一连串的询问。 |
| RAG | 检索加生成 | 检索相关文档，然后为 LLM 的响应提供接地。 |
| Tool call | 函数调用 | LLM 发出带有名称和参数的结构化调用。运行时执行并返回结果。 |
| Agent loop | 规划、行动、验证 | 控制器运行 LLM 调用和工具调用交替进行，直到任务完成。 |
| Prompt injection | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始的基于规则聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌的后期神经网络聊天机器人论文，就在 LLM 智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 提出智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指导，在 2026 年仍然有效。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示注入列为头号安全问题的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实际编排层防御，包括计划-验证-执行和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入的标志性零点击数据泄露 CVE。证明写权限智能体为何需要运行时防御的参考案例。
