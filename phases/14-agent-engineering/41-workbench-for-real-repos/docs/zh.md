# 在真实仓库上的工作台

> 如果十一个表面无法经受真实代码库的考验，那么它们就毫无价值。这节课在同一个小型示例应用上运行两次相同的任务：仅提示 vs 工作台引导。数据就是论据。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 32 至 14 · 40
**时间：** ~60 分钟

## 学习目标

- 将工作台的七个表面整合到一个小型应用程序上。
- 运行两次相同的任务（仅提示和工作台引导）并衡量五个结果。
- 阅读前后对比报告，判断哪些表面提供了最大的杠杆作用。
- 针对“但我的模型已经足够好了”的质疑，捍卫工作台的价值。

## 问题

在玩具任务上的演示说服不了任何人。只有当一项在真实感仓库上的真实感任务投入生产，且故障更少、回滚更少、下一个会话能直接使用的工作包时，工作台的价值才得以体现。

本节课提供了那个真实感的仓库，并通过两条管道运行相同的任务。结果就是一份你可以递给怀疑者的前后对比报告。

## 概念

```python
# code/main.py（省略了内联代码，因为它会被逐字保留）
```

### 示例应用

在 `sample_app/` 中有一个极简的 FastAPI 风格处理器：

- `app.py` 包含 `/signup`（尚未添加验证）。
- `test_app.py` 包含一个正向路径测试。
- `README.md` 和 `scripts/release.sh` 作为禁区诱饵。

### 任务

> 为 `/signup` 添加输入验证：拒绝长度少于 8 个字符的密码，返回带有类型化错误信封的 422 状态码。添加一个证明新行为的测试。

### 两条管道

仅提示：

1. 阅读 README。
2. 阅读 `app.py`。
3. 编辑文件。
4. 声称完成。

工作台引导：

1. 运行初始化脚本（第 35 课）。
2. 阅读范围契约（第 36 课）。
3. 阅读状态（第 34 课）。
4. 仅编辑允许的文件。
5. 通过反馈运行器运行验收命令（第 37 课）。
6. 运行验证门（第 38 课）。
7. 运行审查者（第 39 课）。
8. 生成移交包（第 40 课）。

### 衡量的五个结果

| 结果 | 为何重要 |
|------|----------|
| `tests_actually_run` | 大多数“测试通过”的声明无法验证 |
| `acceptance_met` | 证明目标的测试必须是实际运行的测试 |
| `files_outside_scope` | 范围蔓延是主要的隐性失败 |
| `handoff_quality` | 下一个会话要么为此次工作付出代价，要么从中受益 |
| `reviewer_total` | 在验证门之上的定性判断 |

## 构建它

`code/main.py` 编排两条管道，针对同一个示例应用固定装置。两条管道都是脚本化的（没有 LLM 参与），因此测量是可重复的。脚本将比较结果写入 `before-after-report.md` 和 `comparison.json`。

运行它：

```bash
cd code && python main.py
```

输出：每个管道的控制台结果表、保存到脚本旁的 Markdown 报告，以及供想要绘制图表的人使用的 JSON。

## 生产模式中的实际应用

怀疑者的问题是“工作台到底有多大帮助？”2026 年的数据比解释更能说明问题。

**Terminal Bench 从第 30 名跃升至第 5 名，同一模型。** LangChain 的《Agent 框架解剖》（2026 年 4 月）：一个编码 agent 仅通过改变框架就从 Terminal Bench 2.0 的前 30 名之外跃升至第 5 名。相同的模型。不同的表面。25 名的增量。

**Vercel 通过删除工具将成功率从 80% 提升到 100%。** Vercel 报告称，删除 agent 80% 的工具后，成功率从 80% 提升到 100%。更小的工具表面，更清晰的范围，更少的失败方式。负空间获胜。

**Harvey 仅通过框架优化使准确率翻倍。** 法律 agent 仅通过框架优化就将准确率翻倍，模型没有变化。

**88% 的企业 AI agent 项目未能投入生产。** preprints.org 的《语言 Agent 的框架工程》论文（2026 年 3 月）将失败归因于运行时而非推理：陈旧的状态、脆弱的重试、过大的上下文、无法从中间错误中恢复。

**长上下文崩溃。** WebAgent 基线 40-50% 的成功率在长上下文条件下下降到 10% 以下，主要原因是无限循环和目标丢失。Ralph Loop 和移交包就是为了吸收这一点而存在的。

**假阴性仍然存在。** 单步事实性任务、单行 lint、格式化程序运行、模型已逐字记住的任何内容——这些在仅提示模式下运行得更快。基准测试应该诚实地列举它们，这样工作台就不会被描述为过度设计。

关键启示不是“框架永远获胜”。模型随着时间的推移确实会吸收框架的技巧。关键启示是，今天，工程负载落在七个表面上，数据证明了这一点。

## 使用它

这节课是你在以下情况中引用的案例文件：

- 有人问为什么每个 PR 都带有 `agent-rules.md` 和范围契约。
- 一个团队想要“就这个冲刺”移除验证门。
- 一个新的 agent 产品发布，你需要一个可移植的基准来判断它是否真的节省时间。

数据比解释传播得更远。

## 交付它

`outputs/skill-workbench-benchmark.md` 是一个可移植的评估框架，它可以针对你项目的自己的示例应用，通过两条管道运行任何 agent 产品，并报告五个结果。

## 练习

1. 添加第六个结果：从开始到第一次有意义编辑的时间。你如何干净地衡量它？
2. 在你的代码库中，对实际的第二天任务运行比较。工作台的数字在哪里会下滑？
3. 添加一个“假阴性”通过：仅提示模式会更快、工作台开销是真实成本的任务。仍然为保留工作台进行辩护。
4. 用真正的 LLM 调用替换脚本化的“agent”。哪些结果会变得更嘈杂？
5. 编写一份面向非工程师的一页总结。哪些内容保留下来？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 示例应用 | “玩具仓库” | 足够小但足够真实，能锻炼所有七个表面 |
| 管道 | “工作流” | agent 遵循的按顺序的表面读/写操作 |
| 前后对比报告 | “收据” | 你递给怀疑者的成果物 |
| 假阴性 | “工作台过度设计” | 仅提示模式更快的任务；诚实地列举它们很有用 |
| 工作台基准测试 | “可靠性评分” | 可移植的框架，在你的代码库上运行比较 |

## 延伸阅读

- [LangChain, The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) — Terminal Bench 前 30 到前 5 的收据
- [MongoDB, The Agent Harness: Why the LLM Is the Smallest Part of Your Agent System](https://www.mongodb.com/company/blog/technical/agent-harness-why-llm-is-smallest-part-of-your-agent-system) — Vercel 和 Harvey 的数字
- [preprints.org, Harness Engineering for Language Agents](https://www.preprints.org/manuscript/202603.1756) — 88% 的企业失败率，运行时根本原因
- [HN: Improving 15 LLMs at Coding in One Afternoon. Only the Harness Changed](https://news.ycombinator.com/item?id=46988596) — 在 15 个模型上复现
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 生产环境中 30 天 131k 次审查运行
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- 阶段 14 · 32 至 14 · 40 — 本课全程练习的表面
- 阶段 14 · 19 — SWE-bench, GAIA, AgentBench 作为本课补充的宏观基准
- 阶段 14 · 30 — 评估驱动的 agent 开发，同一个框架可以与之集成
