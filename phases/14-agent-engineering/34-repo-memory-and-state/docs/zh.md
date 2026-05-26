# 仓库记忆与持久状态

> 聊天历史是易失的。仓库是持久的。工作台将智能体状态存储在版本化文件中，以便下一个会话、下一个智能体和下一个评审者都从同一权威来源读取。

**类型：** 构建  
**语言：** Python（标准库 + 可选 `jsonschema`）  
**先决条件：** 阶段 14 · 32（最小工作台）  
**时间：** 约 60 分钟  

## 学习目标

- 定义哪些属于仓库记忆，哪些属于聊天历史。
- 编写 `agent_state.json` 和 `task_board.json` 的 JSON Schema。
- 构建一个状态管理器，实现原子化的加载、验证、变更和持久化。
- 使用 schema 在错误写入损坏工作台之前拒绝它。

## 问题

智能体完成了一个会话。聊天关闭。下一个会话打开并询问从何处开始。模型说“让我检查文件”，读取过时的笔记，然后重做已经完成的工作。更糟的是，它重写了一个已完成的文件，因为没有人告诉它这个文件已经完成了。

工作台的解决方案是仓库记忆：状态存在于仓库的 JSON 文件中，在 schema 下写入，原子化持久化，在代码评审中易于比较差异。聊天是瞬时的数据流；仓库是记录系统。

## 概念

```bash
# 仓库记忆 vs 聊天历史
# 仓库记忆：持久、可搜索、可版本化、可评审
# 聊天历史：易失、无结构、LLM 上下文窗口、一次性

agent_state.json          # 当前工作状态
  ├── schema_version: 1
  ├── active_task_id: "T-0042"
  ├── touched_files: ["src/main.py", "tests/test_main.py"]
  ├── assumptions: ["DB 模式继承自阶段 3 的迁移"]
  ├── blockers: ["CI 缓存键与 PR 分支不匹配"]
  └── next_action: "在 CI 脚本中验证缓存键模式"

task_board.json           # 仓库范围的进度视图  
  ├── schema_version: 1
  └── tasks: [
        { id: "T-0040", status: "done",   owner: "agent-a" },
        { id: "T-0041", status: "review", owner: "agent-b" },
        { id: "T-0042", status: "active", owner: "agent-c" }
      ]
```

### 哪些属于仓库记忆

| 属于 | 不属于 |
|---------|-----------------|
| 当前任务 id | 原始聊天记录 |
| 本会话中接触的文件 | Token 级推理痕迹 |
| 智能体做出的假设 | “用户似乎感到沮丧” |
| 未解决的阻塞项 | 采样补全 |
| 下一步操作 | 供应商特定的模型 id |

判断标准是持久性：三个月后在 CI 重新运行中是否有用？如果有用，存入仓库；如果没用，存入遥测。

### Schema 优先的状态管理

JSON Schema 就是契约。没有它，每个智能体都会发明新字段，每个评审者都要学习新结构，每个 CI 脚本都必须为旧版本特殊处理。有了它，错误的写入会被拒绝。

Schema 涵盖：

- 必需键。
- 允许的 `status` 值。
- 禁止的值（例如数组的 `null`）。
- 模式约束（任务 id 匹配 `T-\d{3,}`）。
- 用于迁移的版本字段。

### 原子写入

状态写入需要能够抵御部分失败：写入临时文件，fsync，重命名覆盖目标文件。状态文件是权威来源；一个写入一半的文件比没有文件更糟糕。

### 迁移

当 schema 发生变化时，在 schema 版本提升的同时提供一个迁移脚本。状态文件包含一个 `schema_version` 字段；管理器拒绝加载无法迁移的版本的文件。

## 构建它

`code/main.py` 实现了：

- `agent_state.schema.json` 和 `task_board.schema.json`。
- 一个仅使用标准库的验证器（JSON Schema 的子集：required、type、enum、pattern、items）。
- `StateManager.load`、`StateManager.update`、`StateManager.commit`，使用原子化的 temp 和 rename 写入。
- 一个演示：修改状态，持久化，重新加载，并验证往返过程。

运行它：

```bash
python code/main.py
```

脚本会在 `workdir/agent_state.json` 和 `workdir/task_board.json` 中写入文件，经过两次“轮次”修改它们，并在每一步打印验证后的状态。

## 生产环境中的常见模式

有四种模式可以将本课程的最小实现扩展到多智能体单体仓库可以承受的程度。

**原子化的 temp 和 rename 不是可选项。** 一个 2026 年 3 月的 Hive 项目 bug 报告清晰地记录了这个失败模式：`state.json` 通过 `write_text()` 写入，异常被捕获并静默处理。部分写入导致会话在损坏的状态上恢复而没有信号。修复方法是始终：在与目标相同的目录中使用 `tempfile.mkstemp`，写入，`fsync`，`os.replace`（在 POSIX 和 Windows 上是原子重命名）。本课程中的 `atomic_write` 正是这样做的。

**对每个非幂等的工具调用使用幂等性键。** 如果智能体在调用工具之后、但在检查点记录结果之前崩溃，恢复时会重试工具调用。对于读取操作是安全的；但对于电子邮件、数据库插入、文件上传则危险。模式：在执行之前将每个工具调用 ID 记录到 `pending_calls.jsonl` 中。重试时，检查该 ID；如果存在，则跳过调用并使用缓存的结果。Anthropic 和 LangChain 在 2026 年的指南中都指出了这一点；LangGraph 的检查点记录器也因同样原因持久化待处理写入。

**将大型工件与状态分离。** 不要将 CSV、长记录或生成的文件存储在 `agent_state.json` 中。将工件保存为单独的文件（或上传到对象存储），仅在状态中保留路径。检查点保持小巧快速；工件独立增长。

**事件溯源用于审计，快照用于恢复。** 每次变更时追加到事件日志（`state.events.jsonl`）；定期快照到 `state.json`。恢复时读取快照，然后重新播放快照时间戳之后的任何事件。这会消耗更多磁盘空间，但可以逐字重放智能体的决策——在调试长时间运行的任务时至关重要。这与 Postgres 内部用于 WAL 的方式相同。

**Schema 迁移，否则拒绝加载。** `schema_version` 整数是契约。当管理器加载一个未知版本的文件时，它拒绝读取。在 schema 版本提升时提供一个迁移脚本；`tools/migrate_state.py` 在每次启动时以幂等方式运行。

## 使用它

在生产环境中：

- **LangGraph 检查点记录器：** 同样的思路，不同的存储。检查点记录器将图状态持久化到 SQLite、Postgres 或自定义后端。本课程教授的 schema 是当检查点记录器失效而需要手动读取状态时你会用到的东西。
- **Letta 记忆块：** 具有结构化 schema 的持久化块（阶段 14 · 08）。相同的规范，但作用于长时间运行的智能体角色。
- **OpenAI Agents SDK 会话存储：** 可插拔后端，schema 感知。本课程中的状态文件就是本地文件后端。

## 交付它

`outputs/skill-state-schema.md` 生成一个项目特定的 JSON Schema 对（状态 + 任务板），一个连接到原子写入的 Python `StateManager`，以及一个迁移脚手架，以便下一个 schema 版本提升不会破坏工作台。

## 练习

1. 添加一个 `last_human_touch` 时间戳。拒绝在人工编辑后五秒内的任何智能体写入。
2. 扩展验证器以支持 `oneOf`，使得一个任务可以是构建任务或评审任务，并且具有不同的必填字段。
3. 添加一个 `schema_version` 字段，并编写从 v1 到 v2 的迁移（将 `blockers` 重命名为 `risks`）。
4. 将存储后端从本地文件迁移到 SQLite。保持 `StateManager` API 不变。
5. 以 50 毫秒的写入竞争运行两个智能体操作同一个状态文件。会发生什么问题？原子重命名如何拯救你？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|------------------------|
| 仓库记忆 | “笔记文件” | 存储在仓库受版本控制文件中的状态，受 schema 约束 |
| Schema 优先 | “验证输入” | 在写入之前定义契约，拒绝偏离 |
| 原子写入 | “只是重命名” | 写入临时文件，fsync，重命名，使部分失败无法损坏数据 |
| 迁移 | “Schema 版本提升” | 将 vN 状态转换为 v(N+1) 状态的脚本 |
| 记录系统 | “权威来源” | 工作台视为权威的工件 |

## 延伸阅读

- [JSON Schema 规范](https://json-schema.org/specification.html)
- [LangGraph 检查点记录器](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Letta 记忆块](https://docs.letta.com/concepts/memory)
- [Fast.io, AI Agent State Checkpointing: A Practical Guide](https://fast.io/resources/ai-agent-state-checkpointing/) — 具有幂等性的 schema 优先检查点
- [Fast.io, AI Agent Workflow State Persistence: Best Practices 2026](https://fast.io/resources/ai-agent-workflow-state-persistence/) — 并发控制、TTL、事件溯源
- [Hive Issue #6263 — non-atomic state.json writes silently ignored](https://github.com/aden-hive/hive/issues/6263) — 真实项目中的失败模式
- [eunomia, Checkpoint/Restore Systems: Evolution, Techniques, Applications](https://eunomia.dev/blog/2025/05/11/checkpointrestore-systems-evolution-techniques-and-applications-in-ai-agents/) — 来自操作系统历史的检查点/恢复原语应用于智能体
- [Indium, 7 State Persistence Strategies for Long-Running AI Agents in 2026](https://www.indium.tech/blog/7-state-persistence-strategies-ai-agents-2026/)
- [Microsoft Agent Framework, Compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction) — 供应商检查点管理器
- 阶段 14 · 08 — 记忆块和休眠时计算
- 阶段 14 · 32 — 本课程模式化的三个文件最低要求
- 阶段 14 · 40 — 从同一 schema 读取的移交数据包
