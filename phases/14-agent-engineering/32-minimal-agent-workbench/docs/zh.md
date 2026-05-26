# 最小智能体工作台

> 最小可用工作台由三个文件组成：根指令路由器、一个状态文件和一个任务板。其他所有内容都是在此基础上叠加的。如果一个仓库连这三个文件都无法承载，那么任何模型也救不了它。

**类型：** 构建  
**语言：** Python（标准库）  
**前置条件：** 阶段 14 · 31（为什么强大的模型仍然会失败）  
**时长：** ~45 分钟

## 学习目标

- 定义构成最小可行工作台的三个文件。
- 解释为什么一个简短的根路由器胜过一份冗长的 `AGENTS.md`。
- 构建一个智能体每轮都可以读取、结束时写入的状态文件。
- 构建一个无需聊天历史就能在多会话工作中存活的任务板。

## 问题

大多数团队通过编写一份 3000 行的 `AGENTS.md` 并宣布完成来搭建工作台。模型加载它，忽略它无法总结的部分，然后在它一贯失败的地方继续失败。

你需要相反的做法。一个微小的根文件，只在相关时路由智能体进入更深的文件。持久的状态，智能体在行动前读取、行动后写入。一个任务板，说明哪些在进行中、哪些被阻塞、哪些是下一步。

三个文件。每个都有自己的职责。每个都足够机器可读，以便以后演变成真正的系统。

## 概念

```plaintext
repo/
├── AGENTS.md              # 根路由器
├── agent_state.json       # 状态（持久记录）
├── task_board.json        # 队列（进行中 / 待办 / 阻塞）
└── docs/                  # 更深的规则（仅按需加载）
    └── agent-rules.md
```

### AGENTS.md 是路由器，不是手册

一份好的 `AGENTS.md` 很简短。它指引智能体指向：

- 状态文件（你当前在哪里）
- 任务板（还剩下什么）
- 更深的规则（位于 `docs/agent-rules.md`）
- 验证命令（如何知道它正常工作）

任何更长的内容都放在更深的文档里，仅在需要时加载。长手册会被忽略。短路由器会被遵循。

### agent_state.json 是记录系统

状态携带：当前任务 ID、已触及的文件、已做出的假设、阻塞项以及下一步动作。智能体在每一轮读取它。下一个会话读取它，而不是重放聊天记录。

状态存在于文件中，因为聊天历史不可靠。会话会结束。对话会被截断。文件不会。

### task_board.json 是队列

任务板携带每个任务及其状态 `todo | in_progress | done | blocked`。它是在状态为空时智能体从中拉取任务的队列，也是你想知道智能体是否在正轨上时阅读的队列。

板上的任务有一个 ID、一个目标、一个负责人（`builder`、`reviewer` 或 `human`）以及验收标准。任务板故意设计得短小：当它增长超过一屏时，你面临的不是任务板问题，而是规划问题。

### 三个文件是地板，不是天花板

后续课程会添加作用域合同、反馈运行器、验证门、审查者检查清单以及交接包。这里的三个文件是所有这些内容的基础。

## 构建

`code/main.py` 将最小工作台写入一个空仓库，并演示单个智能体轮次，其中：

1. 读取 `agent_state.json`。
2. 如果状态为空，则从 `task_board.json` 拉取下一个任务。
3. 触及作用域内的单个文件。
4. 写回更新后的状态。

运行它：

```bash
cd code && python main.py
```

脚本会在其旁边创建 `workdir/`，放下三个文件，运行一轮，并打印差异。重新运行可以看到第二轮如何接续第一轮的位置。

## 使用

在生产智能体产品中，同样的三个文件以不同的名称出现：

- **Claude Code：** `AGENTS.md` 或 `CLAUDE.md` 作为路由器，`.claude/state.json` 风格的存储作为状态，钩子作为任务板。
- **Codex / Cursor：** 工作空间规则作为路由器，会话记忆作为状态，聊天侧边栏中的排队任务作为任务板。
- **自定义 Python 智能体：** 你刚刚编写的相同文件。

名称变了，但形态不变。

## 生产中的模式

当将三种模式叠加在最小工作台之上时，它能在真实的大型仓库中存活下来。它们是独立的；选择你的仓库实际需要的那些。

**嵌套的 `AGENTS.md` 以及最近者优先的优先级。** OpenAI 在主仓库中发布了 88 个 `AGENTS.md` 文件，每个子组件一个。Codex、Cursor、Claude Code 和 Copilot 都从工作文件向仓库根目录遍历，并沿途拼接每个 `AGENTS.md`。子目录文件扩展根文件。Codex 添加了 `AGENTS.override.md` 来替换而不是扩展；override 机制是 Codex 特定的，跨工具工作时请避免使用。Augment Code 的衡量标准是重要的：最好的 `AGENTS.md` 文件带来的质量提升相当于从 Haiku 升级到 Opus；最差的文件产生的输出比根本没有文件还要糟糕。

**即使看起来像覆盖也要拒绝的反模式。** 冲突的指令会静默地将智能体从交互模式降级到贪婪模式（ICLR 2026 AMBIG-SWE：解决率 48.8% → 28%）；对指令进行编号优先级，而不是平铺堆放。没有强制执行命令的不可验证的风格规则（“遵循 Google Python Style Guide”）会让智能体编造合规性；每条风格规则都要配上确切的 lint 命令。以风格而非命令开头会隐藏验证路径；命令优先，风格最后。为人类而不是为智能体写作会浪费上下文预算；简洁是一种特性。

**跨工具符号链接。** 一个单一的根文件加上符号链接（`ln -s AGENTS.md CLAUDE.md`，`ln -s AGENTS.md .github/copilot-instructions.md`，`ln -s AGENTS.md .cursorrules`）可以让所有编码智能体保持相同的真相源。Nx 的 `nx ai-setup` 从单一配置自动为 Claude Code、Cursor、Copilot、Gemini、Codex 和 OpenCode 完成这一操作。

## 发布

`outputs/skill-minimal-workbench.md` 为任何新仓库生成三个文件的工作台：一个针对项目调整的 `AGENTS.md` 路由器、一个具有正确键值的 `agent_state.json`，以及一个填入了当前积压工作的 `task_board.json`。

## 练习

1. 在 `agent_state.json` 中添加一个 `last_run` 时间戳。如果文件早于 24 小时，则拒绝运行，除非操作员确认。
2. 在任务板上添加一个 `priority` 字段，并更改拉取者以始终选择最高优先级的 `todo`。
3. 将 `task_board.json` 迁移为 JSON Lines 格式，这样每个任务占据一行，并且在版本控制中差异更清晰。
4. 编写一个 `lint_workbench.py`，如果 `AGENTS.md` 超过 80 行或引用了不存在的文件，则失败。
5. 决定三个文件中哪一个丢失损失最大。为你的选择辩护。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 路由器 | `AGENTS.md` | 简短的根文件，将智能体指向更深的文档和文件 |
| 状态文件 | “笔记” | 机器可读的记录，记录智能体当前的位置，每轮写入 |
| 任务板 | “积压工作” | JSON 格式的工作队列，包含状态、负责人、验收标准 |
| 记录系统 | “真相源” | 工作台在聊天消失时视为权威的文件 |

## 延伸阅读

- [agents.md — the open spec](https://agents.md/) — 被 Cursor、Codex、Claude Code、Copilot、Gemini、OpenCode 采用
- [Augment Code, A good AGENTS.md is a model upgrade. A bad one is worse than no docs at all](https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files) — 测量到的质量提升
- [Blake Crosley, AGENTS.md Patterns: What Actually Changes Agent Behavior](https://blakecrosley.com/blog/agents-md-patterns) — 实证有效与无效的内容
- [Datadog Frontend, Steering AI Agents in Monorepos with AGENTS.md](https://dev.to/datadog-frontend-dev/steering-ai-agents-in-monorepos-with-agentsmd-13g0) — 实践中的嵌套优先级
- [Nx Blog, Teach Your AI Agent How to Work in a Monorepo](https://nx.dev/blog/nx-ai-agent-skills) — 跨六种工具的单一源生成
- [The Prompt Shelf, AGENTS.md Best Practices: Structure, Scope, and Real Examples](https://thepromptshelf.dev/blog/agents-md-best-practices/) — 能通过审查的章节顺序
- [Anthropic, Claude Code subagents and session store](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/sub-agents)
- 阶段 14 · 31 — 这个最小工作台所吸收的失败模式
- 阶段 14 · 34 — 本课预览的持久状态架构
