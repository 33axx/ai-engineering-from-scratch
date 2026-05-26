# 对话状态跟踪

> “我要找一家北边的便宜餐馆……其实改成中等价位……再加个意大利菜。”三个轮次，三次状态更新。DST 保持 slot-value 字典同步，预订才能正常工作。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 5 · 17（聊天机器人），阶段 5 · 20（结构化输出）  
**时间：** 约 75 分钟

## 问题

在任务导向型对话系统中，用户的目标被编码为一组 slot-value 对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户发言都可能增加、修改或删除一个 slot。系统必须读取完整对话并正确输出当前状态。

只要一个 slot 出错，系统就可能订错餐厅、安排错航班或扣错卡片。DST 是连接用户所说与后端执行的枢纽。

尽管已有 LLM，为何在 2026 年它仍然重要：

- 合规敏感领域（银行、医疗、机票预订）需要确定性的 slot 值，而非自由形式的生成。
- 工具agent 在调用 API 之前仍然需要 slot 解析。
- 多轮修正确实比看起来更难：“不不不，改成周四。”

现代流水线：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

![DST：对话历史 → slot-value 状态](../assets/dst.svg)

**任务结构。** 一个 schema 定义了领域（restaurant, hotel, taxi）及其 slot（cuisine, area, price, people）。每个 slot 可以为空、填充为封闭集合中的值（price: {cheap, moderate, expensive}），或自由形式的值（name: "The Copper Kettle"）。

**两种 DST 形式。**

- **分类。** 对每个 (slot, candidate_value) 对，预测是/否。适用于封闭词表的 slot。2020 年之前的标准做法。
- **生成。** 给定对话，生成 slot 值为自由文本。适用于开放词表的 slot。现代默认做法。

**指标。** 联合目标准确率（JGA）——每一轮中 *所有* slot 都正确的比例。全有或全无。MultiWOZ 2.4 排行榜在 2026 年最高约为 83%。

**架构。**

1. **基于规则（slot 正则 + 关键词）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于拷贝的生成 + BERT 编码。LLM 之前的标准。
3. **LDST（LLaMA + LoRA）。** 经过指令微调的 LLM + domain-slot 提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无 ontology（2024–26）。** 跳过 schema；直接生成 slot 名称和值。处理开放领域。
5. **提示 + 结构化输出（2024–26）。** LLM + Pydantic schema + 约束解码。5 行代码，可投入生产。

### 经典失败模式

- **跨轮指代。** “就选第一个选项。”需要解析出是哪个选项。
- **覆盖 vs 追加。** 用户说“加个意大利菜”。你是替换 cuisine 还是追加？
- **隐式确认。** “好吧” —— 这算接受了所给的预订吗？
- **修正。** “改成晚上 7 点。”必须更新时间，但同时不清除其他 slot。
- **对系统上一轮话语的指代。** “对，就那个。”哪个？

## 构建它

### 步骤 1：基于规则的 slot 提取器

参见 `code/main.py`。正则 + 同义词词典覆盖了窄领域典型话语的 70%：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在典型词汇之外很脆弱。适用于确定性 slot 确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 绝不重置用户未触碰的 slot。
- 明确否定（“算了，不要菜系”）必须清空。
- 用户修正（“其实……”）必须覆盖，而非追加。

### 步骤 3：基于 LLM 的 DST 与结构化输出

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证生成有效的状态对象。无需正则，无 schema 不匹配，无幻觉 slot。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多大比例的轮次中让 ALL 的 slot 都正确？对于 MultiWOZ 2.4，2026 年顶级系统：80-83%。你的领域内系统在窄词汇上应该超过该水平，否则 LLM 基线会击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最后更新的 slot 而非追加。没有 LLM 帮助很难做好。现代模式：始终让 LLM 从历史中重新生成整个状态，而非增量更新——这自然处理的修正。

## 陷阱

- **全历史重新生成成本。** 让 LLM 每轮都重新生成状态，总计 token 消耗为 O(n²)。限制历史长度或对较早轮次进行摘要。
- **Schema 漂移。** 事后添加新 slot 会破坏旧训练数据。对 schema 进行版本管理。
- **大小写敏感性。** “Italian” vs “italian” vs “ITALIAN”——处处规范化。
- **隐式继承。** 如果用户之前指定了“4个人”，那么对时间的新请求不应清除人数。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式 slot；菜系和区域是封闭的。在 schema 中混合使用两者。

## 使用它

2026 年技术栈：

| 情况 | 方法 |
|-----------|----------|
| 窄领域（一两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上） |
| 宽领域，无标注，可投入生产 | LLM + Instructor + Pydantic schema |
| 语音 / 口头 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | Schema 引导的 LLM + 每个领域的 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 降级 + 确认流程 |

## 交付它

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个 slot（cuisine, area, price）构建基于规则的状态跟踪器。在 10 个人工编写的对话上测试。测量 JGA。
2. **中等。** 使用相同数据集，采用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的轮次。
3. **困难。** 实现两者并路由：基于规则为主，当基于规则输出 <2 个置信度 slot 时使用 LLM 降级。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在对话轮次间维护 slot-value 字典。 |
| Slot | 用户意图单元 | 后端需要的命名参数（cuisine, date）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车 —— slot 的集合。 |
| JGA | 联合目标准确率 | 每一轮中所有 slot 都正确的比例。全有或全无。 |
| MultiWOZ | 基准数据集 | 多领域 WOZ 数据集；标准 DST 评估。 |
| 无 Ontology 的 DST | 无 schema | 直接生成 slot 名称和值，没有固定列表。 |
| 修正 | “其实……” | 覆盖之前已填充 slot 的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — 用于 DST 的 LLaMA + LoRA 指令微调。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于拷贝的 DST 工作主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) — 经典 DST 结果。
