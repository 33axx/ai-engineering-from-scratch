# Capstone 09 — 代码迁移代理（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 升级至 17）和 Google 的 App Engine Py2-to-Py3 迁移工具设定了 2026 年的标杆。Moderne 的 OpenRewrite 以可确定的方式大规模执行 AST 重写。Grit 则使用 codemod 风格的 DSL 解决同样的问题。生产模式将两者结合：一个用于安全重写的确定性基座，加上一个处理模糊情况的代理层，一个用于按分支构建的沙箱，以及一个在 PR 打开前就能让测试全部通过（绿色）的测试框架。本收官项目旨在迁移 50 个真实仓库，并发布通过率及失败分类信息。

**类型：** 收官项目
**语言：** Python（代理），Java / Python（目标），TypeScript（仪表盘）
**前置条件：** 阶段 5（NLP），阶段 7（Transformer），阶段 11（LLM 工程），阶段 13（工具），阶段 14（代理），阶段 15（自主），阶段 17（基础设施）
**涉及的阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

大规模代码迁移是 2026 年编码代理最清晰的生产应用之一。基准真相显而易见（迁移后测试套件是否通过？），回报实实在在（Java 8 集群迁移是一个人力级别的项目），基准测试也是公开的（MigrationBench 的 50 个仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。代理层则处理 OpenRewrite 配方无法处理的任何内容：模糊重写、构建系统漂移、长尾语法、传递依赖断裂。

你将构建一个代理，接收一个 Java 8 仓库（或 Python 2 仓库），并生成一个 CI 全部通过（绿色）的迁移分支。你将测量通过率、测试覆盖率保持情况、每个仓库的成本，并构建一个失败分类体系。与仅使用确定性基线的对比，能告诉你代理的价值究竟在哪里。

## 概念

该流水线包含两层。**确定性基座**（Java 使用 OpenRewrite，Python 使用 libcst）安全地执行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、弃用 API 替换。它速度快，并且产生可审计的差异。**代理层**（使用 OpenAI Agents SDK 或基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 LangGraph）处理配方无法解决的问题：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试不稳定、自定义注解。

每个仓库都会获得一个预装了目标运行时的 Daytona 沙箱。代理进行迭代：运行构建、对失败进行分类、应用修复、重新运行。硬性限制：每个仓库 30 分钟，每个仓库 8 美元，20 次代理轮次。如果所有测试通过且覆盖率差异不为负，则分支将开启一个 PR。否则，该仓库将被归档到某个失败类别下，并附上证据。

失败分类体系是交付物。在 50 个仓库中，是什么出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别都有数量和一个示例差异。未来的配方作者可以针对前三名进行改进。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性基座：OpenRewrite（Java）或 libcst（Python）
- 代理：OpenAI Agents SDK 或基于 Claude Opus 4.7 + GPT-5.4-Codex 的 LangGraph
- 沙箱：每个分支的 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准测试：Amazon MigrationBench 的 50 个仓库子集（Java 8 至 17）、Google App Engine Py2-to-Py3 仓库
- 测试框架：并行运行器，通过 Jacoco（Java）或 coverage.py（Python）进行覆盖率分析
- 可观测性：Langfuse + 每个仓库的跟踪包，包含每个差异块
- 仪表盘：失败分类仪表盘，显示每个类别的计数和示例差异

## 构建步骤

1. **配方执行。** 首先运行 OpenRewrite（Java）或 libcst（Python）的配方。抓住 70-80% 的机械性迁移。提交为“recipe”提交。

2. **构建尝试。** Daytona 沙箱：安装目标运行时，运行构建。如果通过（绿色），跳转到测试。如果失败（红色），交给代理处理。

3. **代理循环。** LangGraph 配合工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。代理对失败进行分类（依赖、语法、测试、构建工具）并应用有针对性的修复。重新运行。

4. **预算上限。** 每个仓库限 30 分钟墙钟时间、8 美元成本、20 次代理轮次。任何超出都停止，并将当前差异归档到“预算耗尽”类别。

5. **测试 + 覆盖率门控。** 构建通过（绿色）后，运行测试套件。将覆盖率与基础仓库进行比较。如果覆盖率下降超过 2%，则归档到“覆盖率回归”类别。

6. **开启 PR。** 成功时，推送分支，开启 PR，附带差异以及摘要，说明应用了哪些配方、代理提交了哪些提交。

7. **失败分类。** 对于每个失败的仓库，打上类别标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建一个仪表盘。

8. **50 个仓库运行。** 在 MigrationBench 子集上执行。报告每个类别的通过率、每个仓库的成本、覆盖率保持情况，以及与仅使用确定性基线的对比。

## 使用方法

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付物

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它执行确定性配方，然后进行代理循环，生成一个全部通过（绿色）的迁移分支，或者将该仓库归档到某个分类类别下。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 |
| 20 | 测试覆盖率保持 | 与基础仓库相比的平均覆盖率差异 |
| 20 | 每个迁移仓库的成本 | 通过运行中的 $/repo |
| 20 | 代理/确定性工具集成 | OpenRewrite 处理与代理编写的修复比例 |
| 15 | 失败分析报告 | 分类完整性及示例 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite（无代理）运行迁移流水线。将通过率与完整流水线进行比较。识别哪些情况仅仅依赖代理就能带来差异。

2. 实现一个“lint-清洁”检查：迁移后运行风格检查器（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误，则 PR 失败。测量覆盖率保持但风格退化的比例。

3. 添加一个“最小差异”优化器：在代理分支通过测试后，通过第二次执行来修剪不必要的更改。报告差异大小的缩减。

4. 扩展到第三种迁移：Node 18 到 Node 22。重用沙箱包装；将配方层替换为自定义 codemod。

5. 将“首次通过（绿色）构建时间”（TTFGB）作为用户体验指标进行测量。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|------------------------|
| 确定性基座 | “配方引擎” | OpenRewrite / libcst：带有安全保证的声明式 AST 重写 |
| Codemod | “代码修改程序” | 以机械方式更改源代码的重写规则 |
| 构建漂移 | “工具版本偏差” | Maven / Gradle / uv 在主要版本之间微妙的行為变化 |
| 失败类别 | “分类桶” | 仓库未迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率差异 | “覆盖率保持” | 从基础仓库到迁移分支测试覆盖率百分比的变化 |
| 代理轮次 | “工具调用回合” | 代理循环中的一个计划 -> 行动 -> 观察周期 |
| 预算耗尽 | “触碰上限” | 仓库在未通过的情况下用完了 30 分钟 / 8 美元 / 20 次轮次的限制 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年典型基准测试
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基座参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 另一种 codemod DSL
- [OpenAI 沙箱代码迁移示例](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 至 Py3 迁移器](https://cloud.google.com/appengine) — 另一种迁移基准测试
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基座
- [Daytona 沙箱](https://daytona.io) — 按分支沙箱参考
