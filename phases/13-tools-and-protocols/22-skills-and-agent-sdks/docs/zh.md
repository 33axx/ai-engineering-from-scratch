# 技能与代理 SDK —— Anthropic 技能、AGENTS.md、OpenAI Apps SDK

> MCP 表示“存在哪些工具”。技能表示“如何执行任务”。2026 年的技术栈将两者分层。Anthropic 的 Agent Skills（开放标准，2025 年 12 月发布）以 SKILL.md 形式提供，并支持渐进式揭示。OpenAI 的 Apps SDK 是 MCP 加上小部件元数据。AGENTS.md（现在已存在于 60,000+ 仓库中）作为项目级代理上下文位于仓库根目录。本课程阐明每层覆盖的内容，并构建一个可在不同代理间迁移的最小 SKILL.md + AGENTS.md 组合。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器和加载器）
**先决条件：** 第 13 阶段·07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用的操作知识）、MCP（工具）。
- 编写带有 YAML 前置元数据和渐进式揭示的 SKILL.md。
- 以文件系统方式将技能加载到代理运行时中。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使得一个包能在 Claude Code、Cursor 和 Codex 中工作。

## 问题

一位工程师将一个发布说明编写工作流程提炼为一个多步骤提示：“阅读最新合并的 PR，按领域分组，总结每个组，按照团队风格编写变更日志条目，发布到 Slack 草稿。”他们将其放入团队的 Notion 文档中。

现在他们想从 Claude Code、Cursor 和 Codex CLI 中使用这个工作流程。每个代理加载指令的方式不同：Claude Code 使用斜杠命令、Cursor 使用规则、Codex 使用 `.codex.md`。工程师将工作流程复制了三份并维护三个副本。

AGENTS.md 和 SKILL.md 共同解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的代理在会话启动时读取它：“这个项目如何工作？有哪些约定？哪些命令运行测试？”
- **SKILL.md** 是一个可移植的包：YAML 前置元数据（名称、描述）+ Markdown 正文 + 可选资源。支持技能的代理按名称按需加载它们。
- **MCP**（第 13 阶段·06-14）处理技能需要调用的工具。

三个层次，一个可移植的工件。

## 概念

### AGENTS.md (agents.md)

2025 年末推出，截至 2026 年 4 月已被 60,000+ 仓库采用。一个文件位于仓库根目录。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

代理在会话启动时读取此文件，并用于校准它们在该项目中的行为。2026 年的每个编码代理都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的代理技能（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

前置元数据声明了技能的标识。正文是技能加载时显示给模型的提示。

### 渐进式揭示

技能可以引用子资源，代理仅在需要时获取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中说“有关样式规则，请参阅 style-guide.md”。代理仅在技能实际运行时才拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示。

### 文件系统发现

代理运行时扫描已知目录以查找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载依据文件夹名称和前置元数据的 `name` 字段。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨代理）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载技能，将技能作为可调用的“代理”暴露在运行时内。当用户调用技能时，代理循环分发到该技能。

### OpenAI Apps SDK

2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一在一个开发者界面下。一个 Apps SDK 应用包括：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上 ChatGPT UI 的小部件元数据。
- 加上一个可选的 MCP Apps `ui://` 资源，用于交互界面。

相同的协议，更丰富的用户体验。

### 通过 SkillKit 实现跨代理可移植性

像 SkillKit 及类似的跨代理分发层工具，将单个 SKILL.md 翻译成 32+ AI 代理（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的本地格式。一个真相源；多个消费者。

### 三层栈

| 层次 | 文件 | 加载时机 | 目的 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根目录 | 会话启动 | 项目级约定 |
| SKILL.md | skills 目录 | 技能被调用 | 可复用工作流程 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用的操作 |

这三者协同工作：代理在会话启动时读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，代理通过 MCP 客户端分发。

## 使用它

`code/main.py` 提供了一个标准库 SKILL.md 解析器和加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据和 Markdown 正文，并生成以技能名称为键的字典。然后模拟一个代理循环，按名称调用 `release-notes-writer`。

要关注的内容：

- 使用最小标准库解析器解析 YAML 前置元数据（无 `pyyaml` 依赖）。
- 技能正文原样存储；代理在调用时将其前置到系统提示中。
- 通过 `read_subresource` 函数演示渐进式揭示，该函数按需拉取引用的文件。

## 交付它

本课程生成 `outputs/skill-agent-bundle.md`。给定一个工作流程，技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可跨代理移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，并确认加载器能检测到它。

2. 为本课程仓库编写 AGENTS.md。包含测试命令、样式约定以及第 13 阶段思维模型。

3. 将团队内部文档中的多步骤工作流程移植到 SKILL.md 中。验证它能在 Claude Code 中加载。

4. 手动将技能翻译成 Cursor 和 Codex 的原生规则格式。计算格式之间的差异——这是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课程加载器未覆盖的一个特性。（提示：代理子调用。）

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | “技能文件” | YAML 前置元数据加 Markdown 正文，由代理运行时加载 |
| AGENTS.md | “仓库根代理上下文” | 会话启动时读取的项目级约定文件 |
| 渐进式揭示 | “延迟加载子资源” | 技能正文引用的文件仅在需要时拉取 |
| 前置元数据 | “顶部的 YAML 块” | 位于 `---` 分隔符内的元数据（名称、描述） |
| Claude Agent SDK | “Anthropic 的技能运行时” | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | “MCP + 小部件元数据” | 构建在 MCP 之上并加入 ChatGPT UI 钩子的 OpenAI 开发界面 |
| 技能发现 | “文件系统扫描” | 遍历已知目录查找 SKILL.md，按名称索引 |
| 跨代理可移植性 | “一个技能多个代理” | 通过 SkillKit 风格的工具将单个 SKILL.md 翻译给 32+ 代理 |
| 代理技能 | “可移植的操作知识” | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT UI” | Connectors 和 Custom GPTs 在 MCP 上统一 |

## 延伸阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
