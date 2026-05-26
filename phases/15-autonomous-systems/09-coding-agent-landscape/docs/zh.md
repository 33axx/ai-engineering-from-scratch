# 自主编码智能体格局（2026年）

> SWE‑bench Verified 在不到三年间从 4% 升至 80.9%。同样的 Claude Sonnet 4.5 在 SWE‑agent v1 上得分 43.2%，而在 Cline 自主模式下得分 59.8%——模型周围的脚手架如今与模型本身同等重要。OpenHands（原名 OpenDevin）是 MIT 许可证下最活跃的平台，其 CodeAct 循环直接在沙箱中执行 Python 动作，而非使用 JSON 工具调用。这些头条数字背后隐藏着一个方法论问题：SWE‑bench Verified 的 500 个任务中有 161 个仅需 1–2 行改动，而 SWE‑bench Pro（10+ 行任务）上相同的前沿模型得分仅为 23–59%。

**类型：** 学习  
**语言：** Python（标准库，CodeAct vs JSON 工具调用比较）  
**前置条件：** 阶段 14 · 07（工具使用），阶段 15 · 01（长时程智能体）  
**时间：** ~45 分钟

## 问题

“哪个编码智能体最好”是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布上，使用我将投入生产的脚手架，我能获得怎样的端到端可靠性？

2022 至 2026 年间，该领域认识到脚手架——检索层、规划器、沙箱、编辑‑验证循环、反馈格式——承担着核心负荷。Claude Sonnet 4.5 在 SWE‑agent v1 上得分为 SWE‑bench Verified 的 43.2%；同一个模型在 Cline 的自主脚手架内得分为 59.8%。相差 16.6 个绝对百分点，权重相同。基础模型是一个组件；循环才是产品。

伴随的问题是基准测试饱和掩盖了回归。SWE‑bench Verified 已接近饱和，而简单任务的尾部（500 个任务中有 161 个需要 ≤2 行改动）拉高了最高分数。现实世界的质量更宜在 SWE‑bench Pro（10+ 行改动）这样的分布上衡量，在该分布上同样的领先者仍停留在 23–59%。

## 概念

### SWE‑bench，一段话说明

SWE‑bench（Jimenez 等人）选取带有真实补丁的 GitHub Issue，要求智能体生成一个能使测试套件通过的补丁。SWE‑bench Verified（OpenAI，2024）是一个经过人工整理的 500 任务子集，移除了有歧义和损坏的任务。SWE‑bench Pro 是更难的后续版本——任务需要 10+ 行改动，当前前沿智能体在此得分 23–59%。

### 2022 → 2026 曲线实际展示的内容

- **2022**：研究模型在原始 SWE‑bench 上约为 4%。
- **2024**：GPT‑4 + Devin 风格脚手架约为 14%；SWE‑agent 约为 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE‑agent 内将成绩推至 40–55% 范围。
- **2026**：Claude Sonnet 4.5 及前沿竞品在 SWE‑bench Verified 上达到 70–80%+。Epoch AI 的排行榜实时追踪这一数据。

这条斜率来自三个叠加来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）、以及更好的基准测试（Verified 去除噪声）。

### CodeAct 与 JSON 工具调用对比

OpenHands（All‑Hands‑AI，arXiv:2407.16741，原名 OpenDevin）采取了一项特定的架构赌注：模型不是发出由宿主解码并执行的 JSON 工具调用，而是发出 Python 代码，并由一个类似 Jupyter 的内核在沙箱中执行。智能体可以在一个动作内循环处理文件、链接工具并捕获自己的异常。

权衡如下：

- **JSON 工具调用**：每个动作是一轮对话；易于审计；组合能力有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是完整的程序；组合能力强；需要强化的沙箱（OpenHands 使用 Docker 隔离）；失败模式包括沙箱运行时允许的任何行为。

两种架构都已投入生产。CodeAct 在开放平台（OpenHands、smolagents）中占主导地位。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍占主导地位，因为提供商控制着执行器。

### 2026 年格局中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | CodeAct 在 Docker 中 | 最活跃的开放平台；事件流可重放 |
| SWE‑agent | MIT | 智能体‑计算机接口（ACI） | 首个端到端 SWE‑bench 脚手架 |
| Aider | Apache‑2 | 在本地仓库中通过 diff 编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache‑2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上得分最高的开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 首个“AI 软件工程师”产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍智能体循环 |

### 为什么脚手架占主导地位

一次编码运行是一个长时程轨迹（第 1 课）。可靠性随步骤累积。脚手架提升得分的三个地方：

1. **检索**：找到要读取的正确文件是无声的瓶颈。SWE‑agent 的 ACI、OpenHands 的文件索引以及 Aider 的仓库映射都在解决这个问题。
2. **验证器循环**：运行测试、读取堆栈跟踪并重试，在 SWE‑bench 上产生 10+ 个百分点的差异。
3. **故障隔离**：一个在出错时回滚的沙箱可防止复合损害。带与不带验证器循环的同一个模型看起来像两种不同的产品。

### 基准测试饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE‑bench Verified 有一个简单尾部：500 个任务中有 161 个仅需 1–2 行改动。高分数部分由这个尾部驱动。SWE‑bench Pro 限制为 10+ 行改动，即使前沿系统也仅返回 23–59% 的分数。你的生产分布几乎肯定更接近 Pro 而非 Verified。

对选择智能体的启示：在自己的错误积压中运行一个类似 Pro 的子集。重要的分数是在与你发布内容相符合的任务上的得分。

## 使用它

`code/main.py` 在固定的小型任务分布上比较两个玩具智能体脚手架：

1. 一个 **JSON 工具调用** 脚手架，每轮执行一个动作。
2. 一个 **CodeAct** 脚手架，每个动作可以发出一段小的 Python 代码片段。

两者都使用桩“模型”（确定性规则），使得比较隔离了脚手架与模型质量。输出显示 CodeAct 脚手架以更少的轮次解决了更多任务，但代价是每个动作的爆炸半径更大。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采纳前审计一个提出的编码智能体脚手架：检索质量、验证器存在性、沙箱隔离以及基准测试到分布的匹配度。

## 练习

1. 运行 `code/main.py`。在同样的任务集上，每个脚手架需要多少轮次？每个动作的爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文主张 CodeAct 在复杂任务上胜过 JSON 工具调用。指出论文承认的一个失败模式，并用一句话描述该模式何时会在生产中占主导地位。

3. 从你的错误积压中挑选一个需要跨两个文件进行 10+ 行改动的任务。估算前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。证明差距的合理性。

4. SWE‑bench Verified 有 161 个单文件、1–2 行的任务。构建一个排除它们的分数。排行榜会如何重新排序？

5. 阅读“Introducing SWE‑bench Verified”（OpenAI）。解释用于移除有歧义任务的具体方法论，并指出策展可能会遗漏的一个类别。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|---|---|---|
| SWE‑bench | “编码基准测试” | 带有真实补丁和测试套件的真实 GitHub Issue |
| SWE‑bench Verified | “清理过的子集” | 500 个人工整理的任务，存在简单尾部 |
| SWE‑bench Pro | “更难的子集” | 10+ 行改动；前沿模型得分 23–59% |
| CodeAct | “代码即动作” | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | “函数调用” | 每个动作是一个在执行前经过验证的结构化 JSON 载荷 |
| 脚手架 | “智能体框架” | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（智能体‑计算机接口） | “SWE‑agent 的格式” | 为 LLM 人体工学设计而非人类 shell 的命令集 |
| 验证器循环 | “测试并重试” | 运行测试、读取输出、修改补丁；最大的非模型可靠性提升 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准测试与方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 如何构建整理后子集。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时追踪的分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程编码智能体可靠性框架。
