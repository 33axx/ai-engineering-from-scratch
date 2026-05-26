# 代理指令作为可执行约束

> 以散文形式撰写的指令是愿望。以约束形式撰写的指令是测试。工作台将每条规则转化为代理可以在运行时检查、审查者可以在事后验证的实体。

**类型：** 构建
**语言：** Python（标准库）
**先决条件：** 阶段 14 · 32（最小工作台）
**时间：** ~50 分钟

## 学习目标

- 将路由散文与操作规则分离。
- 将启动规则、禁止行为、完成定义、不确定性处理以及审批边界表达为机器可检查的约束。
- 实现一个规则检查器，根据规则集对运行进行评分。
- 使规则集对 diff 友好，以便审查可以看到更改的内容。

## 问题

典型的 `AGENTS.md` 看起来像一份入职文档。它告诉代理“小心”、“全面测试”和“如果不确定就提问”。三天后，代理在没有测试的情况下提交了更改，写入禁止目录，并且从未提问，因为它从未知道界限在哪里。

指令在具有操作性时是强大的，在只具愿望性时是脆弱的。解决方法是编写工作台能够解释、审查者能够评分的规则。

## 概念

规则应放在 `docs/agent-rules.md` 中，远离简短的根路由。每条规则都有一个名称、一个类别和一个检查项。

```markdown
# agent-rules.md 示例
## Rule: state-file-must-exist
- Category: startup
- Check: check_state_file_exists
```

### 覆盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|----------|---------------------------|---------|
| Startup（启动） | 工作开始前必须满足什么条件？ | “状态文件存在且是最新的” |
| Forbidden（禁止） | 绝对不能发生什么？ | “不要编辑 `scripts/release.sh`” |
| Definition of done（完成定义） | 什么证明任务已完成？ | “pytest 返回 0 且验收线通过” |
| Uncertainty（不确定性） | 代理不确定时该怎么做？ | “打开一个问题说明，而不是猜测” |
| Approval（审批） | 什么需要人工批准？ | “任何新依赖项，任何生产写入” |

一条不属于这五个类别的规则通常希望被拆分为两条规则。强制拆分。

### 规则是机器可读的

每条规则都有一个短标识符、一个类别、一行描述，以及一个 `check` 字段，该字段命名了 `rule_checker.py` 中的一个函数。添加规则意味着添加一个检查；检查器随工作台一起增长。

### 规则对 diff 友好

规则在单个 markdown 文件中每条占用一个标题。重命名在 diff 中可见。新规则位于其类别的顶部。过时的规则被删除，而不是被注释掉，因为工作台是真理之源，而不是团队上个季度感受的聊天记录。

### 规则与框架护栏

框架护栏（OpenAI Agents SDK 护栏、LangGraph 中断）在运行时级别强制执行规则。本课中的规则集是人类可读、可审查的契约，这些护栏实现该契约。两者都需要：运行时在单次交互期间捕获违规，规则集证明运行时做了正确的事情。

## 构建它

`code/main.py` 包含：

- 解析 `agent-rules.md` 并将规则加载到数据类中的解析器。
- `rule_checker.py` 中的样式检查器函数，每个 `check` 引用对应一个函数。
- 一个演示代理运行，违反两条规则，以及一个捕获它们的检查过程。

运行它：

```bash
cd code && python main.py
```

输出：解析的规则集、运行追踪、每条规则的通过/失败状态，以及脚本旁边保存的 `rule_report.json`。

## 生产环境中的模式

有三种模式能将持续一个季度的规则集与一周内就退化的规则集区分开。

**编写时标记严重性。** 每条规则携带 `severity`：`block`（阻止）、`warn`（警告）或 `info`（信息）。检查器报告所有三种；运行时仅在遇到 `block` 时拒绝。大多数团队早期夸大严重性，然后在截止日期压力下悄悄削弱；编写时标记迫使提前校准。与验证门（阶段 14 · 38）配对，该门将对任何 `block` 规则的覆盖签名记录到 `overrides.jsonl` 审计日志中。

**规则过期作为强制机制。** 每条规则携带一个 `expires_at` 日期（默认为自编写起 90 天）。当一条未过期的规则连续 60 天零违规时，检查器会发出警告；下一次季度审查要么证明保留它的合理性，要么将其降级为 `info`，要么删除。Cloudflare 的生产 AI 代码审查数据（2026 年 4 月，30 天内跨 5,169 个仓库的 131,246 次审查运行）显示，具有明确过期时间的规则集每个仓库保持在 30 条规则以下；没有过期时间的规则集增长到 80+ 条，且大多数从未触发。

**Markdown 作为源码，JSON 作为缓存。** `agent-rules.md` 是作者编写的文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁文件由预提交钩子重新生成。Markdown 的 diff 是可审查的；JSON 解析不会出现在每次交互中。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 形式相同。

## 使用它

在生产中：

- Claude Code、Codex、Cursor 在会话开始时读取规则，并在拒绝操作时引用它们。检查器在 CI 中重新运行它们以捕获静默漂移。
- OpenAI Agents SDK 护栏注册相同的检查作为输入和输出护栏。Markdown 是文档表面；SDK 是运行时表面。
- LangGraph 中断在飞行中的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复。

规则集在三者之间是可移植的，因为它只是 markdown 加函数名。

## 交付它

`outputs/skill-rule-set-builder.md` 采访项目所有者，将现有的散文指令归类为五个类别，并生成一个版本化的 `agent-rules.md` 以及一个检查器存根。

## 练习

1. 如果你的产品确实需要，添加第六个类别。论证为什么它不能归入五个类别之一。
2. 扩展检查器，使得规则可以携带严重性（`block`、`warn`、`info`），并且报告相应地进行汇总。
3. 将检查器接入 CI：如果最新代理运行中某个 block 严重性的规则失败，则构建失败。
4. 为每条规则添加一个“过期”字段。如果超过 90 天没有检查失败，则该规则将被审查。
5. 找一个真正的 `AGENTS.md` 并将其重写为五类别规则。其中多少行是操作性的？多少行是愿望性的？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| 操作性规则 | “真正的指令” | 工作台可以在运行时检查的规则 |
| 愿望性规则 | “小心” | 没有检查的规则；要么删除，要么升级 |
| 完成定义 | “验收” | 客观的、基于文件的证明任务已完成 |
| 阻止严重性 | “硬规则” | 违规会中止运行；没有操作员无法静默 |
| 规则过期 | “过时规则清理” | 在 N 天内没有失败的规则将被退休 |

## 进一步阅读

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重性
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行，规则组合经验
- [microservices.io, GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [Type-Checked Compliance: Deterministic Guardrails (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门实现：范围、变异测试、违规预算
- 阶段 14 · 32 — 该规则集所插入的最小工作台
- 阶段 14 · 38 — 使用规则报告的验证门
- 阶段 14 · 39 — 对规则合规性进行评分的审查者代理
