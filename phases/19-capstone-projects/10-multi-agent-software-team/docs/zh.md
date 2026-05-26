# 顶点项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化参与者图、Cognition 的 Devin 以及 Factory 的 Droids 都汇聚到了 2026 年相同的形态：一个架构师负责规划，N 个编码者在并行工作树中工作，一个审查者把关，一个测试者验证。并行工作树将墙上时间转化为吞吐量。共享状态和交接协议成为失败面。这个顶点项目是构建该团队，在 SWE-bench Pro 上进行评估，并报告哪些交接环节会失败以及失败频率。

**类型：** 顶点项目
**语言：** Python / TypeScript（智能体）、Shell（工作树脚本）
**先决条件：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 15（自主）、阶段 16（多智能体）、阶段 17（基础设施）
**涵盖阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## 问题

单个智能体的编程工具在大任务上遇到了瓶颈。不是因为单个智能体能力不足，而是因为一个 200k token 的上下文无法同时容纳架构计划、四个并行的代码库切片、审查者评论以及测试输出。多智能体工厂将问题拆分开来：架构师负责计划，编码者负责在并行工作树中实现，审查者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化参与者图——这三种框架都描述了相同的形态。

失败面在于交接环节。架构师规划了编码者无法实现的内容。编码者产生了冲突的差异。审查者批准了一个幻觉修复。测试者与仍在编写的编码者发生竞态。你将构建其中一个团队，在 50 个 SWE-bench Pro 问题上运行，跟踪每一次交接，并发布事后分析报告。

## 概念

角色是类型化的智能体。**架构师**（Claude Opus 4.7）阅读问题，编写计划，并将其分解为具有显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）阅读合并后的差异并批准或要求具体修改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件并报告通过/失败及工件。

通信通过共享任务板（文件支持或 Redis）进行。每个角色消费其允许处理的任务。交接通过 A2A 协议类型化的消息进行。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（一旦编码者启动，计划就冻结；重新规划是单独的事件），以及审查者把关（审查者不能批准自己的修改或自己提议的修改）。

Token 放大是隐藏的成本。每个角色边界都会增加摘要提示和交接上下文。一个 40 轮的单智能体运行在四个角色中变为总共 160 轮。该评分标准特别权衡了 token 效率与单智能体基线，因为问题不在于"多智能体是否有效"，而在于"它是否在每美元成本上胜出"。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每智能体子图
- 消息传递：A2A 协议（Google 2025），用于类型化的智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- 工作树隔离：每个编码者 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 调解的冲突解决
- 评估：SWE-bench Pro（50 个问题）、SWE-AF 场景、HumanEval++（单元测试）
- 可观测性：Langfuse，带角色标记的跨度、每智能体 token 核算
- 部署：K8s，每个角色作为独立 Deployment + 基于积压的 HPA

## 构建步骤

1. **任务板。** 基于文件的 JSONL，包含类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **架构师。** 读取 GitHub 问题，运行 Opus 4.7，使用要求显式子任务接口（涉及的文件、公共函数、测试影响）的计划模板。发出一个包含子任务 DAG 的 `plan_request`。

3. **编码者。** N 个并行工作进程，每个从板上认领一个子任务。每个生成一个新的 `git worktree add` 分支加一个 Daytona 沙箱。实现子任务。发出带有补丁 + 测试变更的 `diff_ready`。

4. **合并协调者。** 在所有编码者完成后，将 N 个分支三方合并到一个暂存分支。仅在存在文件级重叠时进行 LLM 调解的冲突解决。

5. **审查者。** GPT-5.4 读取合并后的差异。不能批准自己编写的差异。发出 `approved`（无操作）或 `review_feedback`，包含路由回相关编码者的具体变更请求。

6. **测试者。** Gemini 2.5 Pro 在干净的沙箱中运行测试套件。捕获工件。发出 `test_passed` 或 `test_failed`（含堆栈跟踪）。失败的测试循环回拥有失败子任务的编码者。

7. **交接核算。** 每次跨越角色边界的消息在 Langfuse 中获得一个跨度，包含负载大小和使用的模型。计算每个子任务的 token 放大（编码者 token + 审查者 token + 测试者 token + 架构师分摊 / 编码者 token）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。与单智能体基线（单个工作树中的一个 Sonnet 4.7）比较 pass@1 和每个已解决问题花费的美元。

9. **事后分析。** 对于每个失败的问题，识别出问题的交接环节（计划过于模糊、合并冲突、审查者误批准、测试波动）。生成交接失败直方图。

## 使用方式

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付成果

`outputs/skill-multi-agent-team.md` 是交付文件。给定一个问题的 URL 和并行度，团队会生成一个可合并的 PR，并附带每角色 token 核算。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 问题子集上的 pass@1 |
| 20 | 并行加速比 | 墙上时间与单智能体基线的对比 |
| 20 | 审查质量 | 在注入 bug 探测中的误批准率 |
| 20 | Token 效率 | 每个已解决问题的总 token 与单智能体的对比 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途向差异中注入一个明显的 bug（主函数体前额外的 `return None`）。测量审查者的误批准率。调整审查者提示，直到误批准率低于 5%。

2. 减少到两个编码者（架构师 + 编码者 + 审查者 + 测试者，编码者顺序执行两个子任务）。比较墙上时间和通过率。

3. 用单写约束（子任务触及不相交的文件集）替换合并协调者。测量架构师的规划负担。

4. 将审查者从 GPT-5.4 换成 Claude Opus 4.7。测量误批准率和 token 成本差异。

5. 添加第五个角色：文档编写者（Haiku 4.5）。审查通过后，它生成一个变更日志条目。衡量文档质量是否值得额外的 token 开销。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|------------------------|
| 并行工作树 | "隔离分支" | `git worktree add` 为每个编码者生成一个全新的工作树 |
| 任务板 | "共享消息总线" | 文件或 Redis 存储，包含智能体订阅的类型化消息 |
| 交接 | "角色边界" | 任何从一个角色上下文跨越到另一个角色的消息 |
| Token 放大 | "多智能体开销" | 所有角色的总 token / 同一任务的单智能体 token |
| A2A 协议 | "智能体到智能体" | Google 2025 年关于类型化智能体间消息的规范 |
| 合并协调者 | "集成者" | 运行三方合并并调解冲突的组件 |
| 误批准 | "审查者幻觉" | 审查者批准包含已知 bug 的差异 |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026 年参考多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — 微软的类型化参与者框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 另一参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离基础
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
