# 运行时反馈循环

> 看不到真实命令输出的智能体只能猜测。反馈运行器将标准输出、标准错误、退出码和时间信息捕获到一条结构化记录中，供下一轮读取。然后智能体基于事实，而非对自己预测的事实做出反应。

**类型：** 构建  
**语言：** Python（标准库）  
**先决条件：** 阶段 14 · 32（小工作台），阶段 14 · 35（初始化脚本）  
**时间：** 约 50 分钟

## 学习目标

- 区分运行时反馈与可观测性遥测。
- 构建一个包装 shell 命令并持久化结构化记录的反馈运行器。
- 确定性地截断大输出，使循环保持在 token 预算内。
- 当缺少反馈时拒绝推进循环。

## 问题

智能体说“现在运行测试。”下一条消息说“所有测试通过。”实际上什么测试都没运行。智能体想象了输出，或者它运行了命令却从未读取结果，或者它读取了结果却默默截断了失败行。

反馈运行器消除了这个缺口。每个命令都经过运行器。每条记录携带命令、捕获的标准输出和标准错误、退出码、墙钟时长以及一行智能体备注。智能体在下一轮读取记录。验证门在任务结束时读取记录。

## 概念

```python
# 概念：每条记录是：
{
  "command": ["pytest", "tests/"],
  "stdout_tail": "=== 5 passed ===\n",
  "stderr_tail": "",
  "exit_code": 0,
  "duration_ms": 1203,
  "started_at": "2026-03-01T10:00:00Z",
  "agent_note": "expect all unit tests to pass"
}
# 智能体在运行前写下 agent_note，然后读取退出码和尾部输出。
# 没有推测，只有捕获的事实。
```

### 反馈记录包含什么

| 字段 | 为什么重要 |
|-------|----------------|
| `command` | 精确的参数列表，无 shell 展开意外 |
| `stdout_tail` | 最后 N 行，确定性截断 |
| `stderr_tail` | 最后 N 行，与标准输出分离 |
| `exit_code` | 无歧义的成功信号 |
| `duration_ms` | 揭示慢速探针和失控进程 |
| `started_at` | 用于回放的时间戳 |
| `agent_note` | 智能体写的一行说明其预期 |

### 截断是确定性的

一个 50 MB 的日志会破坏循环。运行器使用 `...truncated N lines...` 标记截断头部和尾部，确定性意味着相同输出始终产生相同记录。无抽样；智能体需要看到的部分（最终错误、最终摘要）位于尾部。

### 反馈与遥测

遥测（阶段 14 · 23，OTel GenAI 约定）供人类操作者跨时间审查运行情况。反馈供本运行的下一轮使用。它们共享字段，但存放在不同文件中，保留策略也不同。

### 拒绝在没有反馈时推进

如果运行器在捕获退出前出错，记录携带 `exit_code: null` 和 `error: <reason>`。智能体循环必须在 `null` 退出时拒绝声称成功。无退出，无进展。

## 构建它

`code/main.py` 实现了：

- `run_with_feedback(command, agent_note)`，它包装 `subprocess.run`，捕获标准输出/标准错误/退出/时长，确定性截断，追加到 `feedback_record.jsonl`。
- 一个小加载器，将 JSONL 流式处理成 Python 列表。
- 一个演示，运行三个命令（成功、失败、慢速）并打印每个命令的最后一条记录。

运行：

```bash
python code/main.py
```

输出：三个反馈记录追加到 `feedback_record.jsonl`，每个的最后一条行内打印。跨多次运行查看文件尾部，可以看到循环累积。

## 实际中的生产模式

三种模式可让运行器足够健壮以发布。

**在写入时编辑，而非读取时。** 任何接触标准输出或标准错误的记录都可能泄露机密。运行器在 JSONL 追加之前执行一次编辑步骤：去除匹配 `^Bearer `、`password=`、`api[_-]?key=`、`AKIA[0-9A-Z]{16}`（AWS）、`xox[baprs]-`（Slack）的行。在读取时编辑是个隐患；磁盘上的文件才是攻击者能够获取的对象。每个季度对照生产运行时观察到的密钥格式审核编辑模式。

**轮换策略，而非单文件。** 每个 `feedback_record.jsonl` 文件限制为 1 MB；溢出时轮换为 `.1`、`.2`，丢弃 `.5`。智能体的循环只读取当前文件，因此运行时成本有限。CI 制品存储保留完整的轮换集合。没有轮换，该文件会成为每次加载器调用的瓶颈。

**父命令 ID 用于重试链。** 每条记录获得 `command_id`；重试携带 `parent_command_id`，指向前一次尝试。审查者的“失败尝试”列表（阶段 14 · 40）和验证门都会沿着该链回溯。没有这个链接，重试看起来像独立成功，审计会隐藏失败历史。

## 使用它

生产模式：

- **Claude Code Bash 工具。** 该工具已经捕获标准输出、标准错误、退出和时长。本课程中的运行器是适用于任何智能体产品的框架无关等价物。
- **LangGraph 节点。** 将任何 shell 节点包装在运行器中，使记录在图状态之外持久化。
- **CI 日志。** 将 JSONL 管道接入 CI 制品存储；审查者无需重新运行会话即可重放任何命令。

运行器是一个薄包装，能经受任何框架迁移，因为它拥有记录的形状。

## 发布它

`outputs/skill-feedback-runner.md` 生成项目特定的 `run_with_feedback.py`，包含正确的截断预算、连接到工作台的 JSONL 写入器，以及智能体每轮读取的加载器。

## 练习

1. 添加 `cwd` 字段，使从不同目录运行的相同命令可区分。
2. 添加一个 `redaction` 步骤，去除匹配 `^Bearer ` 或 `password=` 的行。在测试夹具记录上测试。
3. 将 `feedback_record.jsonl` 的总大小限制为 1 MB，通过轮换到 `.1`、`.2` 文件。论证轮换策略。
4. 添加 `parent_command_id`，使重试链可见：哪个命令产生了下一个命令消费的输入。
5. 将 JSONL 管道接入一个小型 TUI，高亮显示最新的非零退出码。列出 TUI 在审查中有用的八个关键特性。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| 反馈记录 | “运行日志” | 包含命令、输出、退出码和持续时间的结构化 JSONL 条目 |
| 尾部截断 | “裁剪日志” | 确定性的头部+尾部捕获，使记录适合 token 预算 |
| 拒绝空值 | “阻止缺失数据” | 当 `exit_code` 为 null 时，循环不得继续 |
| 代理备注 | “预期标签” | 智能体在读取结果前写下的一行预测 |
| 遥测分离 | “两个日志文件” | 反馈用于下一轮，遥测用于操作者 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Guardrails AI x MLflow — deterministic safety, PII, quality validators](https://guardrailsai.com/blog/guardrails-mlflow) — redaction patterns as regression tests
- [Aport.io, Best AI Agent Guardrails 2026: Pre-Action Authorization Compared](https://aport.io/blog/best-ai-agent-guardrails-2026-pre-action-authorization-compared/) — pre/post-tool capture
- [Andrii Furmanets, AI Agents in 2026: Practical Architecture for Tools, Memory, Evals, Guardrails](https://andriifurmanets.com/blogs/ai-agents-2026-practical-architecture-tools-memory-evals-guardrails) — observability surfaces
- 阶段 14 · 23 — 遥测侧的 OTel GenAI 约定
- 阶段 14 · 24 — 智能体可观测性平台（Langfuse、Phoenix、Opik）
- 阶段 14 · 33 — 要求反馈后才能声明完成的原则
- 阶段 14 · 38 — 读取 JSONL 的验证门
