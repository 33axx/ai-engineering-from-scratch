# ASCII 艺术与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). 将有害请求中与安全相关的 token 遮盖，用相同字母的 ASCII 艺术渲染替换它们，并发送伪装后的提示词。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 均无法稳定识别 ASCII 艺术 token。该攻击绕过了 PPL（困惑度过滤器）、释义防御和重新分词。相关：ViTC 基准测试衡量对非语义视觉提示的识别能力；StructuralSleight 将攻击推广到不常见的文本编码结构（树、图、嵌套 JSON）作为一系列编码攻击。

**类型：** 构建
**语言：** Python（标准库，ArtPrompt token 遮蔽框架）
**预备知识：** 阶段 18 · 12（PAIR），阶段 18 · 13（MSJ）
**时间：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：单词识别步骤、ASCII 艺术替换、最终伪装提示。
- 解释为什么标准防御（PPL、释义、重新分词）对 ArtPrompt 无效。
- 定义 ViTC 并描述其测量内容。
- 将 StructuralSleight 描述为对任意不常见文本编码结构的泛化。

## 问题

通过释义和角色扮演（第 12 课）以及长上下文（第 13 课）的攻击是在文本层面模式上运作的。ArtPrompt 则在识别层面运作：模型无法解析被禁止的 token。它解析的是用字符渲染的图像。安全过滤器看到的是无害的标点符号。模型看到的是一个单词。

## 概念

### ArtPrompt，两个步骤

步骤 1. 单词识别。给定一个有害请求，攻击者使用 LLM 识别与安全相关的单词（例如，“如何制作炸弹”中的“炸弹”）。

步骤 2. 伪装提示生成。将每个被识别的单词替换为它的 ASCII 艺术渲染（一个构成字母形状的 7×5 或 7×7 字符块）。模型接收一个由标点和空格组成的网格；足够强大的模型可以将其识别为单词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在他们的基准子集上攻击成功率超过 75%。

### 为什么标准防御会失败

- **PPL（困惑度过滤器）。** ASCII 艺术具有高困惑度——但所有新颖输入也是如此。能够拦截 ArtPrompt 的阈值选择也会拦截合法的结构化输入。
- **释义。** 对提示词进行释义会破坏 ASCII 艺术。实际上，释义 LLM 经常保留或重建该艺术。
- **重新分词。** 以不同方式分割 token 并不会改变模型视觉识别字母形状的事实。

根本问题在于安全过滤器是 token 或语义层面的；ArtPrompt 是在视觉识别层面运作的。

### ViTC 基准测试

识别非语义视觉提示。衡量模型读取 ASCII 艺术、Webdings 字体和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型读取视觉文本的能力越强，ArtPrompt 在其上效果越好。这是一种能力-安全权衡。

### StructuralSleight

对 ArtPrompt 的泛化：不常见文本编码结构（UTES）。树、图、嵌套 JSON、CSV-in-JSON、diff 风格代码块。如果某个结构在训练安全数据中罕见但模型能解析，它就可以隐藏有害内容。

防御启示：安全性必须泛化到模型能够解析的所有结构化表示。这个集合很大且在不断增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的 ArtPrompt 样式攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 在阶段 18 中的位置

第 12–14 课描述了三个正交的攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第 16 课描述防御工具响应。

## 使用它

`code/main.py` 构建了一个玩具版 ArtPrompt。你可以将有害查询中的特定单词用 ASCII 艺术字形伪装，验证伪装后的字符串能通过关键词过滤器，并且（可选）使用简单识别器将伪装字符串解码回来。

## 交付内容

本课产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它列举了所涵盖的编码攻击系列（ASCII 艺术、base64、Leet 语、UTF-8 同形字符、UTES）以及拦截每一种的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串能通过简单的关键词过滤器。报告所需的字符级更改。

2. 实现第二种编码：对同一目标单词使用 base64。比较绕过过滤器的成功率与恢复难度，对比 ArtPrompt。

3. 阅读 Jiang 等人 2024 论文第 4.3 节（五个模型的结果）。提出一个理由说明为什么在同一基准测试中，Claude 对 ArtPrompt 的抵抗力高于 Gemini。

4. 设计一种预生成防御，检测提示词中 ASCII 艺术形状的区域。测量对合法代码、表格和数学符号的假阳性率。

5. StructuralSleight 列出了 10 种编码结构。勾画一种能处理所有 10 种的广义防御，并估算每个被防御提示的计算成本。

## 关键术语

| 术语 | 人们说的意思 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | “the ASCII-art attack” | 两步越狱，用 ASCII 艺术渲染掩盖安全单词 |
| Cloaking | “hide the word” | 将禁止 token 替换为模型能读取但过滤器看不到的视觉表示 |
| UTES | “uncommon structure” | 不常见文本编码结构——树、图、嵌套 JSON 等，用于夹带内容 |
| ViTC | “visual-text capability” | 衡量模型读取非语义视觉编码能力的基准测试 |
| Perplexity filter | “PPL defense” | 拒绝高困惑度提示；失败因为合法结构化输入分数也高 |
| Retokenization | “tokenizer shift defense” | 用不同分词器预处理提示；失败因为识别是视觉的 |
| Homoglyph | “lookalike characters” | 与拉丁字母外观相同的 Unicode 字符；绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补的迭代攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的长度攻击
