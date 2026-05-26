# 顶点项目 01 — 终端原生编程代理

> 到 2026 年，编程代理的形态已经定型：一个 TUI 框架、一个有状态计划、一个沙箱化的工具面、一个计划-行动-观察-恢复的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都一样。这个顶点项目要求你从头构建一个端到端的代理——从命令行输入到拉取请求输出——并在 SWE-bench Pro 上将其与 mini-swe-agent 和 Live-SWE-agent 进行对比评估。你将学到难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本上限。

**类型：** 顶点项目  
**语言：** TypeScript / Bun（框架）、Python（评估脚本）  
**先修条件：** 阶段 11（LLM 工程）、阶段 13（工具与协议）、阶段 14（代理）、阶段 15（自主系统）、阶段 17（基础设施）  
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**时长：** 35 小时

## 问题

编程代理在 2026 年成为了主导的人工智能应用类别。Claude Code（Anthropic）、Cursor 3 及其 Composer 2 和 Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（11.2 万星）、Factory Droids 和 Google Jules 都推出了相同架构的变体：一个终端框架、一个有权限的工具面、一个沙箱、以及一个围绕前沿模型构建的计划-行动-观察循环。前沿很窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2%——但工程技艺很广。大多数失败模式并非模型错误，而是工具循环不稳定、上下文中毒、失控的令牌成本以及破坏性的文件系统操作。

你无法从外部推理这些代理。你必须构建一个，观察循环在第 47 轮因 ripgrep 返回 8MB 匹配项而崩溃，然后重建截断层。这就是本顶点项目的意义所在。

## 概念

框架有四个面。**计划** 维护一个 TodoWrite 风格的状态对象，模型每一轮重写该对象。**行动** 分发工具调用（读取、编辑、运行、搜索、git）。**观察** 捕获 stdout / stderr / 退出码，截断，并将摘要反馈回去。**恢复** 处理工具错误，同时不撑爆上下文窗口或陷入无限循环。2026 年的形态增加了一个新东西：**钩子**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——可配置的扩展点，操作者可在此注入策略、遥测和护栏。

沙箱是 E2B 或 Daytona。每个任务在一个全新的 devcontainer 中运行，并挂载一个 git worktree（读写权限）。框架从不接触主机文件系统。工作树在成功或失败时被销毁。成本控制通过三层强制实现：每轮令牌上限、每次会话的美元预算、以及硬性的轮次数限制（通常 50 轮）。可观测性层是采用 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5（终端内 React）
- 模型访问：OpenRouter 统一 API，搭配 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最困难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程、tree-sitter 解析器（支持 17 种语言，预编译）
- 隔离：每个任务使用 `git worktree add`，任务成功或失败后清理
- 评估框架：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己的 30 任务保留集
- 可观测性：OpenTelemetry SDK，带 `gen_ai.*` semconv → 自托管 Langfuse
- PR 发布：GitHub App，使用细粒度令牌，作用域限制在目标仓库

## 构建步骤

1. **TUI 和命令循环。** 使用 Ink 搭建一个 Bun 项目。接受 `agent run <repo> "<task>"` 命令。打印一个分割视图：计划面板（顶部）、工具调用流（中间）、令牌预算（底部）。添加 Ctrl-C 取消功能，该功能在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个类型化的 TodoWrite 模式（待处理 / 进行中 / 已完成项，带备注）。模型每一轮通过工具调用重写完整状态——不允许增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（带差异预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使框架与传输层无关。每个工具返回截断后的输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 创建一个新分支。所有工具调用在沙箱内执行。主机文件系统不可达。

5. **钩子。** 实现全部八种 2026 钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止在工作树外执行 `rm -rf`；(b) `PostToolUse` 令牌记账；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终的 trace 包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个问题子集。对你的框架针对每个问题运行。与 mini-swe-agent（最简基线）在 pass@1、每任务轮数和每任务美元成本上进行比较。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性上限：50 轮、200k 上下文、每任务 5 美元。`PreCompact` 钩子将较早的轮次总结为一个先前状态块（在 150k 标记处），为新的观察腾出空间，同时不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` 加上一个 GitHub API 调用，该调用创建一个 PR，正文中包含计划和差异摘要。

## 使用方法

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 提交要求

交付物位于 `outputs/skill-terminal-coding-agent.md` 中。给定一个仓库路径和一个任务描述，它会在沙箱中运行完整的计划-行动-观察循环，并返回一个 PR URL 和一个 trace 包。本顶点项目的评分标准如下：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 与基线对比 | 你的框架在 30 个匹配的 Python 任务上与 mini-swe-agent 对比 |
| 20 | 架构清晰度 | 计划/行动/观察分离、钩子面、工具模式——参照 Live-SWE-agent 布局进行评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | trace 完整性（100% 工具调用都有 span）、每轮令牌记账 |
| 15 | 开发者体验 | 冷启动 < 2 秒、崩溃恢复后恢复计划、Ctrl-C 能在工具执行中途干净取消 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 替换为 Qwen3-Coder-30B（在 vLLM 上服务）。比较 pass@1 和每任务美元成本。报告开放模型表现不佳的地方。

2. 添加一个 `reviewer` 子代理，在 PR 发布前读取差异，并可请求修订循环。测量假阳性审查是否会使 SWE-bench 通过率低于单代理基线（提示：通常是的）。

3. 对沙箱进行压力测试：编写一个尝试 `curl` 外部 URL 的任务，以及一个尝试写出工作树之外的任务。确认两者都被 PreToolUse 钩子阻止。记录这些尝试。

4. 使用较小的模型（Haiku 4.5）实现 `PreCompact` 总结。测量在 3 倍压缩下丢失了多少计划保真度。

5. 将 MCP StreamableHTTP 传输替换为 stdio。对比冷启动和每次调用延迟。为仅限本地使用选择胜出者。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 框架（Harness） | "代理循环" | 围绕模型的代码，负责分发工具、维护计划状态和执行预算 |
| 钩子（Hook） | "代理事件监听器" | 由框架在八个生命周期事件之一运行的用户编写脚本 |
| 工作树（Worktree） | "Git 沙箱" | 一个位于独立路径的链接 Git 检出；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 一个类型化的待处理/进行中/已完成项列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：长连接 HTTP，支持双向流；取代 SSE |
| 令牌上限（Token ceiling） | "上下文预算" | 每轮或每次会话的输入+输出令牌上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | SWE-bench 任务第一次运行就解决的比例，无重试或测试集窥探 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考框架
- [Cursor 3 变更日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 框架对比的最简基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — 开源框架，11.2 万星
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本顶点项目针对的评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和令牌使用的 span 模式
