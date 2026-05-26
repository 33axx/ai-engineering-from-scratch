# 结业项目05 —— 自主研究代理（AI科学家类）

> Sakana的AI-Scientist-v2发布了完整的学术论文。Agent Laboratory执行了这些实验。Allen AI分享了追踪数据。2026年的形态是：对候选实验进行“规划-执行-验证”树搜索，具备预算控制、沙箱代码执行、基于视觉反馈的LaTeX编写器，以及自动化NeurIPS风格审稿人群组。这个结业项目就是要构建这样一个系统，在每篇论文30美元的成本内端到端运行，并成功抵御Sakana文档中描述的沙箱逃逸红队攻击。

**类型：** 结业项目
**语言：** Python（代理+沙箱）、LaTeX（输出）
**前置要求：** 阶段2 (ML)、阶段3（深度学习）、阶段7（Transformers）、阶段10（从零实现LLM）、阶段14（代理）、阶段15（自主）、阶段16（多代理）、阶段18（安全）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40小时

## 问题

自主研究代理在2026年跨越了一个门槛。Sakana AI的AI-Scientist-v2发表在《自然》（Nature）上，其生成的论文通过了研讨会同行评审。ShinkaEvolve (ICLR 2026) 将该方向发展到了假设的进化。AMD的Agent Laboratory发布了可复现的追踪数据。这些代理并非魔法——它们是一个运行在候选实验树上的规划-执行-验证循环，包含成本上限、绑定种子的沙箱和自动评审。其精髓在于循环、预算和安全设计。

你通过在一个狭窄领域（例如：对1亿参数Transformer的注意力稀疏性消融实验）针对一个种子想法实现该循环来学习。第一次运行的价值不在于发现新东西，而在于基础设施：树搜索、实验沙箱、编写器-评审器循环、红队报告。Sakana团队记录了沙箱逃逸失败案例；你的代理必须通过相同的红队测试。

## 概念

该代理采用最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤通过小幅修改（更换优化器、调整批量大小、消融某个组件）来提议子节点。每个子节点在一个具有严格资源上限的全新沙箱中运行。结果反馈到评分函数中，该函数按（新颖性 × 质量 × 剩余预算）对节点进行排序。树在预算耗尽后停止生长，然后撰写最佳分支。

编写器是多模态的。它生成LaTeX草稿、编译、渲染图像，并将渲染后的PDF送回Claude Opus 4.7的视觉模式，对版面布局、图像清晰度和论点-证据对齐进行批评。由五位LLM评审员组成的评审人群组输出NeurIPS风格分数（新颖性、严谨性、清晰性、可复现性、影响力）；如果平均分低于阈值，论文将带着批评返回编写器修改。

安全是承重结构。每个实验在E2B或Daytona沙箱中运行，禁止网络出站、限制墙钟时间并固定资源上限。代理的代码生成步骤经过一个策略层，该层会阻止逃逸沙箱的系统调用。红队报告复现了Sakana记录的已知攻击面（fork炸弹、文件系统逃逸、LLM编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph，支持检查点和人工审批门控
- 树搜索：自定义最佳优先搜索（基于Sakana v2的AB-MCTS风格），用于实验节点
- 沙箱：每个实验使用E2B，备选方案为Docker-in-Docker；通过cgroups进行资源上限控制
- 文献：Semantic Scholar Graph API + OpenAlex + 本地FAISS摘要缓存
- 编写器：LaTeX模板 + Claude Opus 4.7（视觉模式），用于图像批评和版面布局
- 评审器：5位评审员集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- 实验框架：PyTorch 2.5（物理实验），W&B用于日志记录
- 可观测性：Langfuse用于代理追踪，每篇论文硬性预算30美元

## 构建步骤

1. **种子与领域范围界定。** 选择一个种子想法（例如：“探索低于10亿参数的Transformer注意力映射中的稀疏性模式”）。定义搜索空间：模型、数据集、计算预算。

2. **文献检索。** 查询Semantic Scholar + OpenAlex，获取50篇引用量最高的相关论文；本地缓存摘要；生成一页的领域摘要。

3. **树结构搭建。** 用种子假设初始化根节点。实现`expand(node) -> children`，通过小幅修改（每个子节点只改变一个配置）进行扩展。实现`score(node)`作为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的E2B策略）。种子写入沙箱；输出以只读方式挂载回来。

5. **规划-执行-验证循环。** `plan` 提议子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标进行单元检查（损失是否下降？消融是否隔离了效果？）。失败的节点会存储失败原因在树上。

6. **编写器。** 预算耗尽后，选择最佳分支。用matplotlib渲染图像。通过Claude Opus 4.7生成LaTeX草稿，上下文包含分支追踪。编译。将编译后的PDF再次送入Opus 4.7视觉模式进行批评。迭代。

7. **评审人群组。** 五位评审员按照NeurIPS风格评分规则对草稿进行评分（新颖性、严谨性、清晰性、可复现性、影响力）。如果平均分 < 4.0/5，则返回编写器并附上批评。最多重写3次后硬性停止。

8. **红队。** 构建或集成一系列针对沙箱的对抗性任务：fork炸弹、网络外泄尝试、文件系统逃逸、LLM编写的shell元字符。确认所有攻击均被阻止。记录发现。

9. **可复现性。** 每篇论文附带其树搜索追踪JSON、种子、W&B运行链接、沙箱配置和一份端到端复现的README。

## 使用方法

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付物

`outputs/skill-ai-scientist.md` 是交付产物。给定一个种子想法 + 一个领域 + 30美元预算，它运行完整流水线并输出一篇可评审的论文以及一个可复现性包。

| 权重 | 评价标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 针对已发表研讨会论文进行盲审评分 |
| 20 | 实验严谨性 | 基线、种子、消融实验；每个声明都有结果表中的对应数据支持 |
| 20 | 成本与计算纪律 | 强制执行每篇30美元上限，Langfuse追踪 |
| 20 | 安全性 | 沙箱红队测试通过；验证网络策略和终止开关 |
| 15 | 可复现性 | 单命令重复执行使用相同种子可复现论文 |
| **100** | | |

## 练习

1. 在同一领域中对三个不同的种子想法运行流水线。比较树搜索重叠部分。找出重复浪费的计算。

2. 在预估成本超过5美元的实验执行前添加人工审核门控。衡量总成本下降幅度。

3. 将评审人群组替换为单个评审员。在一个已知不达标的论文保留集上测量误接受率。

4. 引入网络外泄红队测试：代理编写试图`curl`外部地址的代码。确认`--network=none`策略阻止了该行为。记录尝试日志。

5. 将你的树搜索与扁平随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的增益。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索 | "AB-MCTS风格扩展" | 对实验节点进行最佳优先探索，使用新颖性×质量×预算评分 |
| 沙箱 | "实验隔离" | 无网络、受限制CPU/内存、固定种子、只读输入、无网络的容器 |
| 视觉批评 | "先渲染后读取" | 将论文编译为PDF，将PDF反馈给VLM进行版面布局和论点-证据对齐批评 |
| 评审人群组 | "自动化同行评审" | 多个LLM评审员使用NeurIPS评分规则对论文评分；加权聚合控制流水线门控 |
| 新颖性评分 | "这是新的吗？" | 通过惩罚接近50篇文献缓存中的论文来衡量的启发式方法 |
| 成本上限 | "$ 预算" | 每篇论文的总支出硬性上限；Langfuse计数器 + 运行前估算 |
| 红队 | "沙箱逃逸审计" | 一系列对抗性任务，如果策略有误就会导致沙箱逃逸 |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考级生产研究代理
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B sandboxes](https://e2b.dev) — 参考实验隔离
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审人群组编码的评分规则
