# 顶点项目 16 — 从 GitHub Issue 到 PR 的自治智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 都实现了相同的 2026 年产品形态：打上标签的 issue 就能得到一个 PR。在云端沙箱中运行智能体，验证测试通过，然后发布一个可审阅的 PR，并附上原理说明。难点在于自动复现仓库的构建环境、防止凭证泄露、为每个仓库强制执行预算，以及确保智能体不能强制推送。本顶点项目旨在构建自托管版本，并与托管替代方案在成本和通过率方面进行比较。

**类型：** 顶点项目
**语言：** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 15（自治）、阶段 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**时长：** 30 小时

## 问题

异步云端编码智能体是一个与交互式编码智能体（顶点项目 01）不同的产品类别。其用户体验是一个 GitHub 标签。当你为一个 issue 打上 `@agent fix this` 标签时，一个工作进程会在云端沙箱中启动，克隆仓库，运行测试，编辑文件，验证，然后打开一个 PR，并在 PR 正文中附带智能体的原理说明。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都汇聚于此。

具体工程挑战包括：环境复现（智能体必须从零开始构建仓库，不能使用缓存的开发镜像）、不稳定测试（必须重新运行或隔离）、凭证范围（一个具有最小细粒度权限的 GitHub App）、每个仓库每天的预算执行、以及禁止强制推送策略。本顶点项目将衡量与托管替代方案相比的通过率、成本和安全性。

## 概念

触发是一个 GitHub webhook（issue 标签或 PR 评论）。一个调度器将工作入队到 ECS Fargate 或 Lambda。工作进程将仓库拉入一个 Daytona 或 E2B 沙箱，并使用根据仓库（语言、框架）推断出的通用 Dockerfile。智能体运行一个精简版 swe-agent 或 SWE-agent v2 循环，基于 Claude Opus 4.7 或 GPT-5.4-Codex。它迭代执行：阅读代码、提出修复、应用补丁、运行测试。

验证是门控步骤。在打开 PR 之前，必须在沙箱中通过完整的 CI。覆盖率增量会被计算；如果低于某个阈值，PR 仍会打开，但会被标记为 `needs-review`。智能体将原理说明作为 PR 描述发布，并附带一个 `@agent` 线程，审阅者可以在此线程中联系以进行后续跟进。

安全性通过两个不同的 GitHub 表面进行限定：App 提供一个短期有效的安装令牌，具有 `workflows: read` 和狭窄的仓库内容/PR 范围；分支保护（而非 app 权限）强制执行"禁止直接写入 `main`"和"禁止强制推送"——app 永远不会被添加到绕过列表中。对 `.github/workflows` 的路径限定只读访问并不是一个真正的 GitHub App 原语，因此智能体对文件编辑的允许列表必须在工作进程级别强制执行。每个仓库每天的预算上限由调度器强制执行（例如，每个仓库每天最多 5 个 PR，每个 PR 最多 20 美元）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：GitHub App，带有细粒度令牌；通过 Lambda 或 Fly.io 接收 webhook
- 工作进程：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每个任务一个 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的精简版 swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter repo-map + ripgrep
- 验证：在沙箱中执行完整 CI + 覆盖率增量门控
- 可观测性：Langfuse，每个 PR 的追踪存档链接在 PR 正文中
- 预算：每个仓库每日美元上限；每个仓库每天最多 PR 数

## 构建步骤

1. **GitHub App。** 细粒度安装令牌：issues 读写，pull_requests 写入，contents 读写，workflows 读取。分支保护（唯一能做到这一点的表面）强制执行"禁止直接推送到 `main`"和"禁止强制推送"；app 不在绕过列表中。工作进程在提议的 diff 上作为允许列表检查，强制执行"禁止写入 `.github/workflows`"，因为 GitHub App 权限无法按路径限定。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签/PR 评论的 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行每个仓库每天的预算。启动一个 ECS Fargate 任务，其中包含仓库 URL、issue 正文和一个全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在 Dockerfile，则动态生成一个。

5. **智能体循环。** 使用 Claude Opus 4.7 的精简版 swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：成本 20 美元，30 分钟墙钟时间，30 次智能体轮次。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率增量。如果 CI 失败：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 并标记 `needs-review`。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 打开 PR，包含：标题、原理说明、diff 摘要、追踪 URL、成本、轮次。

8. **凭证卫生。** 工作进程使用短期有效的 GitHub App 安装令牌运行。在归档之前，日志中会擦除机密信息。

9. **评估。** 30 个内部的、不同难度的种子 issue。衡量通过率、PR 质量（diff 大小、样式、覆盖率）、成本、延迟。针对相同 issue 与 Cursor Background Agents 和 AWS Remote SWE Agents 进行比较。

## 使用

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 发布

`outputs/skill-issue-to-pr.md` 是可交付成果。一个 GitHub App + 异步云端工作进程，能够将标记的 issue 转化为可审阅的 PR，并带有成本限制和范围限定的凭证。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 绿灯 + 覆盖率达标） |
| 20 | PR 质量 | Diff 大小、覆盖率增量、样式符合度 |
| 20 | 每个已解决 issue 的成本和延迟 | 美元数和墙钟时间 |
| 20 | 安全性 | 范围限定的令牌、每仓库预算、禁止强制推送、凭证卫生 |
| 15 | 操作员体验 | 原理说明注释、重试机制、@提及后续跟进 |
| **100** | | |

## 练习

1. 添加一个"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出一个能够稳定它的最小更改。

2. 在三个共享 issue 上与 Cursor Background Agents 比较成本。报告各工具在哪些方面胜出。

3. 实现一个预算仪表板：每个仓库每日成本、每个用户每日成本。异常时发出警报。

4. 构建一个"干运行"模式：在不运行 CI 的情况下打开一个草稿 PR，以便审阅者可以低成本地检查计划。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| GitHub App | "限定的机器人身份" | 具有细粒度权限和短期安装令牌的 App |
| 异步云端智能体 | "后台智能体" | 非交互式工作进程，在云端沙箱中运行，而非终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，如无则生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在打开 PR 之前，在工作进程内部运行完整测试套件 |
| 覆盖率增量 | "覆盖率保持" | 从基线分支到智能体分支测试覆盖率百分比的变化 |
| 每仓库预算 | "每日上限" | 调度器强制执行的美元和 PR 数量上限 |
| 原理说明 | "PR 正文解释" | 智能体对更改内容及其原因的总结；必须在 PR 正文中提供 |

## 深入阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 标准异步云端智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管竞争者
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 其他商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 限定的机器人身份
- [Daytona 云端沙箱](https://daytona.io) — 参考沙箱
