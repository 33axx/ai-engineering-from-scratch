# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 让 N 个智能体共享同一段对话；由选择函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多智能体对话的典型模式——智能体并不知晓自己在静态图中的角色，它们只是对共享的对话池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的参与者模型。微软于 2026 年 2 月将 AutoGen 置入维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月发布 RC）。GroupChat 这一原语在 AG2 和 Microsoft Agent Framework 中均得以保留——一次学习，随处可用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 阶段 16 · 04（原语模型）
**时间：** 约 60 分钟

## 问题

当工作流已知时，静态图（LangGraph）表现良好。但真实的对话并非静态：有时编码者会询问审核者，有时询问研究人员，有时询问写作者。硬编码每一种可能的交接会导致边爆炸。你希望*智能体对共享池做出反应*，并由某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 所做的事情。

## 概念

### 结构

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体都能看到每一条消息。在每一轮对话中，会调用一个选择函数来选出下一个发言者。

### 三种选择器风格

**轮询。** 固定的循环顺序。确定性强。随 N 线性扩展，但忽略上下文——即使当前话题是法律审查，编码者也可能获得发言权。

**LLM 选择。** 调用 LLM，读取最近的对话池并返回最合适的下一个发言者。具有上下文意识，但速度慢：每一轮对话都会增加一次 LLM 调用。AutoGen 的默认方式。

**自定义。** 一个 Python 函数，包含任意你想要的逻辑。典型做法：LLM 选择加上后备规则（例如，“在编码者发言后，总是把发言权交给验证者”）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成一轮发言后，管理器调用选择器，选择器返回下一个智能体。循环持续，直到满足终止条件。

### 终止

三种常见模式：

- **最大轮数。** 对总对话轮数设置硬上限。
- **“TERMINATE”标记。** 智能体可以发送一个哨兵消息；当该消息出现时，管理器停止。
- **目标达成检查。** 每一轮运行一个轻量级验证器，在目标完成时停止对话。

### AutoGen → AG2 分裂 与 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动的参与者模型对 AutoGen（v0.4）进行重大重写。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，事件驱动的参与者模型被合并入 **Microsoft Agent Framework**（2026 年 2 月 RC，现已与 Semantic Kernel 合并）。GroupChat 概念在两个轨道中均得以保留，但实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### GroupChat 适用场景

- **涌现式对话。** 你不希望预连接每一种可能的下一发言者。
- **角色混合任务。** 编码者询问研究人员，研究人员询问档案管理员，档案管理员再询问编码者。流程不是 DAG。
- **探索性问题解决。** 更像是“头脑风暴会议”，而不是“流水线”。

### 失效场景

- **严格的确定性。** LLM 选择器可能不一致。相同的提示词，不同的运行，不同的下一发言者。
- **奉承级联。** 智能体会顺从发言最自信的那一个。需要明确地对抗提示。
- **上下文膨胀。** 每个智能体读取每一条消息；10 轮之后上下文已非常庞大。使用投影（第 15 课）来限定视野。
- **热门发言者。** 某个智能体主导了对话，因为选择器偏爱它的专长。可以引入发言平衡作为选择器的一项特征。

### 群聊 vs 监督者

相同的原语，不同的默认设置：

- 监督者：一个智能体进行规划，其他智能体执行。选择器是“询问规划者该做什么”。
- 群聊：所有智能体是平等的；选择器是一个作用于共享池的函数。

两者都使用第 04 课中的四个原语。群聊默认采用 LLM 选择的编排方式和全池共享状态。

## 构建

`code/main.py` 使用标准库从头实现了一个 GroupChat。三个智能体（编码者、审核者、管理者），包含轮询和 LLM 选择两种变体，并在出现 `TERMINATE` 标记时终止。

演示程序会打印对话记录以及两种变体下选择器的决策轨迹。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-groupchat-selector.md` 为给定任务配置了一个 GroupChat 选择器——轮询 vs LLM 选择 vs 自定义，以及选择器应使用哪些输入（最近消息、智能体专长、轮次计数）。

## 交付

检查清单：

- **最大轮数上限。** 始终设置。典型任务设为 10-20 轮。
- **发言平衡指标。** 跟踪每个智能体的发言次数；当不平衡超过阈值时发出警报。
- **终止标记。** `TERMINATE` 或一个专用的验证智能体。
- **投影或限定内存。** 大约 10 条消息后，考虑为每个智能体提供限定视野，以防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择的变体，同时记录选择器的输入和它的选择，否则无法调试。

## 练习

1. 运行 `code/main.py`。比较轮询 vs LLM 选择下的对话。哪种选择器下哪个智能体占主导？
2. 在选择器中添加一个“每个智能体最大发言次数”规则。这会如何影响对话记录？
3. 实现一种目标达成终止：当审核者返回“approved”时停止。在达到轮数上限之前，它多久触发一次？
4. 阅读 AutoGen 稳定版关于 GroupChat 的文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 增加了哪些具体属性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 人们通常说的意思 | 实际含义 |
|------|----------------|------------------------|
| GroupChat | “智能体在一个聊天室里” | 共享消息池 + 选择函数。AutoGen / AG2 原语。 |
| Speaker selection | “谁下一个说话” | 选择下一个智能体的函数。轮询、LLM 选择或自定义。 |
| GroupChatManager | “会议主持人” | AutoGen 组件，拥有选择器并循环执行对话轮次。 |
| ConversableAgent | “基础智能体” | AutoGen 基类；一个可以发送和接收消息的智能体。 |
| Termination token | “‘停止’词” | 用于结束对话的哨兵字符串（通常为 `TERMINATE`）。 |
| Hot speaker | “一个智能体主导对话” | 选择器不断选择同一个智能体的失败模式。 |
| Context bloat | “池无限增长” | 每个智能体读取每一条历史消息；上下文随轮次增长。 |
| Projection | “限定视野” | 针对角色的共享池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) —— 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) —— 社区对 AutoGen v0.2 的延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) —— 合并后的后继者，2026 年 2 月 RC
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) —— 事件驱动参与者模型重写详情
