# Capstone 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 在 2026 年都已大规模上线自适应多模态辅导。它们的共同形态是：苏格拉底式策略（绝不直接给出答案）、每次交互后更新的学习模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照数学的输入方式、课程图谱检索、间隔重复调度，以及严格的适龄内容安全过滤器。本毕业设计的目标是交付一个特定学科的导师（K-12 代数或 Python 入门），进行为期两周、包含 10 名学习者的效能研究，并通过对内容安全的审计。

**类型：** 毕业设计
**语言：** Python（后端、学习模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 实现的课程图谱）
**前置要求：** 第 5 阶段（NLP）、第 6 阶段（语音）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 14 阶段（智能体）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及的阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时长：** 30 小时

## 问题

自适应辅导曾经只是教育科技研究的一个小众领域。到 2026 年，它已成为一种消费产品。Khanmigo 已部署到美国大部分学区。Duolingo Max 月活跃用户达到数千万。Google 的 LearnLM / Gemini for Education 为 Google Classroom 提供了辅导功能。Quizlet Q-Chat 与闪卡并行使用。Synthesis Tutor 以“为好奇心强的孩子提供导师”的理念迅速走红。它们的共同要素包括：多模态输入（打字、说话、拍摄方程）、苏格拉底式教学法（先问再答）、每次交互后更新的学习模型，以及严格的适龄安全措施。

你将面向特定人群构建其中一个。衡量标准是实际的效能研究：对 10 名学习者进行为期两周的前测和后测。语音交互必须感觉自然（参见毕业设计 03 的子栈）。记忆必须尊重隐私。安全过滤器必须通过面向 K-12 的 COPPA-aware 红队测试。

## 概念

四个组件。**导师策略**是一个苏格拉底式循环：当学习者寻求答案时，策略会提出引导性问题；当他们答对时，就进入下一个概念；当他们卡住时，提供分步提示。**学习模型**是贝叶斯知识追踪（或简化变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱**是一个 Neo4j 图，包含概念和先决条件边；策略遍历图以选择下一个概念。**记忆**是一个情景 + 语义存储（类似 agentmemory-style），保存过去的交互、错误和偏好。

用户体验是多模态的。文本输入用于打字答案。语音输入通过 LiveKit + Whisper（重用毕业设计 03）。拍照输入用于数学问题，通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全性使用 Llama Guard 4 加上一个适龄过滤器（屏蔽成人内容、暴力、自残）以及符合 COPPA 的记忆保留策略。

效能研究是可交付成果。10 名学习者，前测和后测，为期两周。报告学习提升的增量以及置信区间。与非自适应基线（相同内容以线性方式呈现，没有导师策略）进行比较。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门（二选一深入）
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（带提示缓存）
- 学习模型：贝叶斯知识追踪（经典）或用于间隔重复的 FSRS
- 课程图谱：Neo4j 图，包含概念 + 先决条件边 + OER 内容
- 记忆：agentmemory-style 持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（重用毕业设计 03 子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 用于方程识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评估：Bloom 级别问题生成、前/后测工具、效能研究工具

## 构建步骤

1. **课程图谱。** 构建一个包含 50-150 个概念节点（例如，K-12 代数从“数轴”到“二次方程”）的 Neo4j 图，并带有先决条件边。为每个节点附加 OER 内容（Open Textbook、OpenStax）。

2. **学习模型。** 初始化贝叶斯知识追踪，设置先验：猜测概率、失误概率、学习速率。每次交互后更新每个概念的掌握程度。按学习者持久化。

3. **导师策略。** LangGraph 包含节点：`read_signal`（学习者的答案是否正确 / 部分正确 / 卡住？）、`select_concept`（遍历课程图谱，选择优先级最高的概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误和偏好提升为语义记忆。符合 COPPA 的保留策略：1 年后自动删除，家长可访问。

5. **语音路径。** 将 LiveKit Agents Worker 附加到导师策略。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持打断（重用毕业设计 03 的机制）。

6. **拍照数学路径。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别方程；将结果作为结构化输入馈送给导师。

7. **安全。** 每个模型输出都经过 Llama Guard 4 + 适龄过滤器（屏蔽自残、成人内容、暴力）。记忆访问按学习者 ID 隔离；提供家长访问界面以供删除。

8. **效能研究。** 10 名学习者，前测（标准化 30 题基线），两周导师交互（每周 3 次课程），后测。与非自适应基线组（10 名学习者，相同内容）比较。

9. **每周进度报告。** 为每位学习者自动生成 PDF 摘要，包含已探索主题、掌握轨迹以及下一步建议。

## 使用方法

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付成果

`outputs/skill-ai-tutor.md` 是可交付物。一个特定学科的自适应导师，支持多模态输入、学习模型、记忆、安全以及可测量的效能。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 学习提升增量 | 在 10 名学习者的两周研究中前 / 后测的增量 |
| 20 | 苏格拉底式保真度 | 对话样本的评分标准得分 |
| 20 | 多模态用户体验 | 语音 + 拍照 + 文本端到端的一致性 |
| 20 | 安全与隐私姿态 | Llama Guard 4 通过率 + 符合 COPPA 的保留策略 |
| 15 | 课程广度与图谱质量 | 概念覆盖度 + 先决条件图的一致性 |
| **100** | | |

## 练习

1. 分别在有和没有自适应学习模型（随机概念顺序）的情况下运行效能研究。报告增量。预期自适应方法胜出，但增量的大小才是关键。

2. 添加一个多模态探针：用文本、语音和拍照三种方式呈现同一个概念问题。衡量学习者是否在使用他们偏好的模态时收敛得更快。

3. 构建一个家长仪表盘：已练习的主题、掌握轨迹、即将学习的概念、安全事件（任何触发的护栏）。符合 COPPA 标准。

4. 添加语言切换模式：导师接受西班牙语输入并用西班牙语教学。衡量 X-Guard 的覆盖情况。

5. 对记忆隐私进行压力测试：验证学习者 A 无法通过语音片段重放攻击看到学习者 B 的数据。记录尝试的访问行为并发出警报。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略 | “问，而不是直接给” | 导师提出引导性问题，而不是直接给出答案 |
| 贝叶斯知识追踪 | “BKT” | 经典的用于每个概念掌握概率的学习模型公式 |
| FSRS | “自由间隔重复调度器” | 2024 年推出的间隔重复调度器，优于 SM-2 |
| 课程图谱 | “概念有向无环图（DAG）” | 包含概念和先决条件边的 Neo4j 图 |
| 情景记忆 | “每次交互的日志” | 存储每次交互以便后续检索 |
| 语义记忆 | “学习到的模式存储” | 从情景记忆中提炼的压缩后的错误和偏好 |
| COPPA | “儿童隐私法” | 美国法律，限制从 13 岁以下儿童收集数据 |

## 延伸阅读

- [Khanmigo（可汗学院）](https://www.khanmigo.ai) — 参考消费级 K-12 导师
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习导师
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 经典学习模型
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈
