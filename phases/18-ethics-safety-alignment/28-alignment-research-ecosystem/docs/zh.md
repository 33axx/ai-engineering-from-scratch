# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五大机构定义了2026年非实验室对齐研究层。MATS（机器学习对齐与理论学者计划）：自2021年底以来已有527+名研究人员，180+篇论文，10K+引用，h指数47；2024年夏季批次以501(c)(3)形式注册，约90名学者和40名导师；2025年前校友中有80%从事安全/安保工作，200+人分布在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo。Redwood Research：由Buck Shlegeris创立的应用对齐实验室；提出了AI控制（第10课）；与英国AISI合作开展控制安全案例研究。Apollo Research：为前沿实验室提供部署前的欺骗性策略评估；撰写了《上下文中的欺骗性策略》（第8课）和《迈向AI欺骗性策略的安全案例》。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间跨度研究；《前沿AI安全政策的共同要素》比较了各实验室框架。Eleos AI Research：部署前的模型福利评估（第19课）；对Claude Opus 4进行了福利评估。

**类型：** 学习
**语言：** 无
**前提条件：** Phase 18 · 01-27（前Phase 18课程）
**时间：** ~45分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个机构及其核心产出。
- 描述MATS的规模（学者、论文、h指数）及其作为人才输送管道的作用。
- 描述Redwood的AI控制议程及其与英国AISI的合作关系。
- 描述METR基于任务的评估方法论。

## 问题

前沿实验室（第18课）内部进行安全评估并发布部分结果。实验室之外的生态系统是评估得到验证、新型故障模式首次被发现、以及人才得到培养的地方。理解这个生态系统有助于解读哪些研究发现被谁所信任。

## 概念

### MATS（机器学习对齐与理论学者计划）

始于2021年底。研究指导项目；学者们用10-12周时间与一位资深研究员共同解决一个特定的对齐问题。

规模（2026年）：
- 自成立以来已有527+名研究人员。
- 发表180+篇论文。
- 10K+次引用。
- h指数47。
- 2024年夏季：90名学者 + 40名导师；注册为501(c)(3)组织。

职业去向：~80%的2025年前校友从事安全/安保工作。200+人分布在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由Buck Shlegeris创立。提出了AI控制议程（第10课）。与英国AISI合作开展控制安全案例研究。为DeepMind和Anthropic提供评估设计建议。

经典论文：Greenblatt、Shlegeris等人《AI控制》(arXiv:2312.06942, ICML 2024)；《对齐伪装》(Greenblatt、Denison、Wright等人, arXiv:2412.14093, 与Anthropic合作)。

风格：具体威胁模型、最坏情况对手、可压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前的欺骗性策略评估。撰写了《上下文中的欺骗性策略》（第8课, arXiv:2412.04984）。参与2025年OpenAI反欺骗性策略训练合作项目。制作了《迈向AI欺骗性策略的安全案例》（2024年）。

风格：在可能产生欺骗的智能体设置中进行评估；三支柱分解（不一致性、目标导向性、情境感知）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间跨度研究。《前沿AI安全政策的共同要素》（metr.org/common-elements, 2025）比较了各实验室框架。

与Apollo共同撰写了AI欺骗性策略安全案例草案。

风格：长时程任务评估、经验性能力测量、框架综合。

### Eleos AI Research

部署前的模型福利评估。对Claude Opus 4进行了系统卡5.3节中记录的福利评估。为第19课的福利相关主张提供了外部方法论核查。

### 流程

MATS培养研究人员。毕业生进入Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估机构）。外部评估机构与实验室以及英国AISI/CAISI合作。成果发表后反馈到MATS，用于下一批学员。

### 为什么这一层很重要

单一来源的评估并不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估机构能够发现并验证实验室可能低估的故障模式。2024年的《潜伏智能体》论文（第7课）是Anthropic + Redwood；《对齐伪装》是Anthropic + Redwood；《上下文中的欺骗性策略》是Apollo；《反欺骗性策略》是Apollo + OpenAI。多机构结构正是质量控制所在。

### 在Phase 18中的位置

第7-11课引用了Redwood和Apollo的工作；第18课引用了METR的框架比较；第19课引用了Eleos。第28课提供了整个Phase所依赖的生态系统的明确组织图谱。

## 使用它

无需代码。阅读METR的《前沿AI安全政策的共同要素》，作为外部综合如何为实验室内部政策工作增值的示例。

## 交付

本课程产出 `outputs/skill-ecosystem-map.md`。给定一个对齐主张或评估，它能够识别机构、发布渠道和方法论风格，并与已知的对应机构进行交叉验证。

## 练习

1. 从第7-15课中选一篇论文，识别涉及的机构。将作者与MATS校友及当前生态系统中的任职机构进行交叉核对。

2. 阅读METR的《前沿AI安全政策的共同要素》。找出他们强调的三项跨实验室趋同点和两个最大的分歧点。

3. MATS的职业去向中约80%从事安全/安保工作。论证这种选择压力是适应性的（培养领域）还是偏向性的（过滤掉异见立场）。

4. Redwood和Apollo都从事控制/欺骗性策略工作，但风格不同。选择一个故障模式，描述各自会如何调查它。

5. Eleos AI是唯一专注于模型福利的机构。设计一个假设的第二机构，专注于另一个福利相关的问题（认知自由、机器人具身化等），并阐明其方法论。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| MATS | “那个指导项目” | 机器学习对齐与理论学者计划；自2021年以来527+名研究人员 |
| Redwood Research | “那个控制实验室” | 应用对齐；AI控制作者；英国AISI合作伙伴 |
| Apollo Research | “欺骗性策略评估” | 为前沿实验室提供部署前的欺骗性策略评估 |
| METR | “任务跨度评估” | 基于任务的能力评估；框架综合 |
| Eleos AI | “福利实验室” | 部署前的模型福利评估 |
| 人才输送管道 | “MATS -> 实验室” | MATS毕业生流向Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | “非实验室检查” | 评估不是由模型生产者进行；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 指导项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI控制论文
- [Apollo Research](https://www.apolloresearch.ai/) — 欺骗性策略评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
