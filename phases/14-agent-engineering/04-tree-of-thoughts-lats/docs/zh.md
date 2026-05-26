# 思维树与LATS：审慎搜索

> 单条思维链路径没有回溯空间。ToT（Yao 等人，2023）将推理转化为树结构，每个节点上进行自我评估。LATS（Zhou 等人，2024）将 ToT 与 ReAct 和 Reflexion 统一到蒙特卡洛树搜索框架下。24点游戏从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 阶段 14 · 01（智能体循环），阶段 14 · 03（Reflexion）
**时长：** ~75 分钟

## 学习目标

- 将推理视为搜索：节点是“思维”，边是“扩展”，值是“前景如何”。
- 使用标准库实现 ToT 风格的 BFS 树搜索，并包含自我评估评分。
- 扩展为玩具 LATS MCTS 循环，包含选择 / 扩展 / 模拟 / 反向传播。
- 判断何时值得使用搜索（增加 token 乘数，如 24 点游戏、代码生成），何时单条轨迹足够（简单问答）。

## 问题

思维链是一条线性路径。如果第一步出错，后续所有步骤都基于错误的前提。在 24 点游戏（用四个数字通过 + − × ÷ 得到 24）中，GPT-4 CoT 的准确率仅为 4%。模型过早选择了错误的子表达式，无法挽回。

推理需要的能力是：提出多个候选方案，评估它们，选择有前景的，并在遇到死胡同时回溯。这就是搜索。思维树和 LATS 是两种典型形式。

## 概念

### 思维树（Yao 等人，NeurIPS 2023）

每个节点是一个连贯的中间步骤（“一个思维”）。每个节点可以扩展为 K 个子思维。LLM 通过评分提示对每个节点进行自我评估。搜索遍历树——BFS、DFS 或集束搜索。

```python
# 伪代码：ToT BFS
def bfs(state, max_depth, beam_width):
    frontier = [state]
    for depth in range(max_depth):
        candidates = []
        for node in frontier:
            for thought in propose_thoughts(node, K=3):  # 扩展 K 个思维
                score = self_evaluate(node, thought)      # LLM 给出评分
                candidates.append((node, thought, score))
        # 保留评分最高的若干候选
        frontier = select_top_k(candidates, beam_width)
    return best_leaf(frontier)
```

自我评估是关键部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分，以及候选投票。三者均大幅优于 CoT，在 24 点游戏中（GPT-4 从 4% 提升到 74%）。

### LATS（Zhou 等人，ICML 2024）

LATS 将 ToT、ReAct 和 Reflexion 统一到 MCTS 框架下。LLM 扮演三个角色：

- **策略（Policy）**：提出候选下一步动作（ReAct 风格）。
- **价值函数（Value function）**：对部分轨迹进行评分（ToT 风格的自我评估）。
- **自我反思（Self-reflector）**：失败时，写出自然语言反思（Reflexion 风格），并用于为后续的 rollout 重新播种。

环境反馈（观测）融入价值函数，使搜索基于真实工具结果而非仅仅模型观点。论文发表时的结果：HumanEval pass@1 92.7%（GPT-4，SOTA），WebShop 平均 75.9（GPT-3.5，接近基于梯度的微调）。

### MCTS 的最小化实现

每次迭代四个阶段：

1. **选择（Select）**——从根节点走到叶节点，使用 UCT（树置信上界）。
2. **扩展（Expand）**——通过策略生成 K 个子节点。
3. **模拟（Simulate）**——从子节点出发使用策略进行 rollout，用价值函数（或环境奖励）对叶节点评分。
4. **反向传播（Backpropagate）**——沿路径更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。`c` 需要根据任务调整。

### 成本现实

搜索会导致 token 爆炸。ToT 在 24 点游戏上使用的 token 是 CoT 的 100–1000 倍。LATS 类似。这并非免费；应当只在以下场景使用搜索：

- 单条轨迹明显不够的任务（24 点游戏、复杂代码）。
- 正确性比挂钟时间更重要的任务。
- 有廉价且可靠的价值函数的任务（代码的单元测试、数学题的明确目标）。

如果你的任务有唯一正确答案但评估器有噪声，搜索往往适得其反——它可能找到一个“评分高”的错误答案。

### 2026 年的定位

大多数生产环境中的智能体并不运行 LATS。它们运行带有工具验证反馈的 ReAct（CRITIC，第 05 课）。搜索出现在特定领域：

- 将测试作为价值函数的代码生成智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图中的规划密集型工作流。

AlphaEvolve（第 11 课）是 2025 年的极致：对代码进行进化搜索，机器可检查的适应度，前沿收益（56 年来首次 4x4 矩阵乘法改进）。

## 动手构建

`code/main.py` 实现了：

- 一个微型 ToT BFS，用于风格化的“选取算术运算符”任务。
- 一个玩具 LATS MCTS 循环，针对同一任务（选择 / 扩展 / 模拟 / 反向传播），包含 UCT 选择。
- 一个价值函数，结合符号评分与自我评估评分。

运行它：

```bash
cd code && python main.py
```

跟踪输出显示了 ToT 在 BFS 中每层扩展三个候选，而 LATS 通过 MCTS 收敛到最优 rollout。两者均打印了 token 计数。

## 使用指南

LangGraph 将 ToT 风格的探索作为子图模式提供；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供了 `TreeOfThoughts` 智能体。对于大多数 2026 年生产环境的智能体，该模式隐藏在 `if task_complexity > threshold: use_search()` 门后——参见第 05 课的评估器-优化器模式。

## 实战输出

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间进行选择。

## 练习

1. 运行玩具 LATS，分别使用 UCT c=0.1 和 c=2.0。跟踪输出有何变化？
2. 将价值函数替换为更嘈杂的评分器（加入随机抖动）。MCTS 还能找到最优叶节点吗？它能容忍的最低信噪比是多少？
3. 实现集束搜索 ToT（每层保留 top-k），并与 BFS 比较。在 tight token 预算下哪个更好？
4. 阅读 LATS 论文第 5.1 节。重现 HumanEval 的轨迹数量：达到报告的 pass@1 需要多少次 rollout？
5. 阅读 LATS 论文关于“LATS 何时帮助较小”的讨论。写一段决策规则，将任务形态映射到搜索策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “分支 CoT” | Yao 等人 —— 带自我评估的思维节点树 |
| LATS | “LLM 的 MCTS” | Zhou 等人 —— 将 ToT + ReAct + Reflexion 统一到 MCTS 下 |
| UCT | “上置信界” | 平衡利用（Q）和探索（ln N / n）的选择公式 |
| Value function | “这个状态有多好” | 提示 LLM 得到的分数或环境奖励；用于反向传播 |
| Policy | “动作提议器” | ReAct 风格的生成器；输出候选的下一个思维/动作 |
| Rollout | “模拟轨迹” | 从节点到叶节点的策略运行，用价值函数评分 |
| Backpropagate | “更新祖先” | 将叶节点的奖励向上传播，更新访问次数和 Q |
| Search cost | “Token 爆炸” | 在 24 点游戏中是 CoT 的 100–1000 倍；采用前需预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— 经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) —— 结合 Reflexion 反馈的 MCTS
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 带程序化评估器的进化搜索
