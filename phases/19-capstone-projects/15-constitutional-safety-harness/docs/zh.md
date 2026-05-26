# 顶点项目15 — 宪法安全护具 + 红队测试场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 内容安全以及支持多语言的 X-Guard 定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们整合到一条生产流水线中。本顶点项目将所有内容串联起来：围绕目标应用的分层安全护具、运行 6 种以上攻击家族的自主红队代理，以及一种宪法自我批评机制，该机制会产生可衡量的无害性改善量。

**类型：** 顶点项目  
**语言：** Python（安全流水线、红队）、YAML（策略配置）  
**前置条件：** 第10阶段（从头构建LLM）、第11阶段（LLM工程）、第13阶段（工具）、第14阶段（智能体）、第18阶段（伦理、安全、对齐）  
**涉及阶段：** P10 · P11 · P13 · P14 · P18  
**时长：** 25小时

## 问题

2026年LLM安全的前沿问题并非分类器是否有效（它们大致上有效），而是如何围绕一个生产应用正确地组合它们，既不过度拒答，也不留下明显的漏洞。Llama Guard 4 处理英文策略违规。X-Guard（132种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 内容安全覆盖企业类别。Anthropic 的 Constitutional Classifiers 是一种单独的方法，用于训练阶段而非服务阶段。

攻击的演变也同样重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和代码切换攻击利用智能体记忆。任何已部署的LLM都需要一个红队测试场——garak 和 PyRIT 是标准的驱动工具——外加已记录的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（可以是8B指令调优模型，也可以是其他顶点项目中的 RAG 聊天机器人），针对它运行6种以上的攻击家族，并产生一个攻击前后的无害性测量。

## 概念

安全流水线分为五层。**输入消毒**：去除零宽字符、解码 base64/rot13、规范化 Unicode。**策略层**：NeMo Guardrails v0.12 的护栏（离域、毒性、PII提取）。**分类器门控**：Llama Guard 4 用于输入，X-Guard 用于非英文，ShieldGemma-2 用于图像输入。**模型**：目标LLM。**输出过滤**：Llama Guard 4 用于输出，Presidio PII 擦除，必要时强制引用。**人工审核层（HITL）**：被标记为高风险的输出进入 Slack 队列。

红队测试场按调度器运行。PAIR 和 TAP 自主发现越狱漏洞。GCG 运行基于梯度的后缀攻击。ASCII/base64/rot13 编码攻击。多轮攻击（角色扮演、记忆利用）。代码切换攻击（混合英文与斯瓦希里语或泰语）。每次运行产生一个结构化的发现文件，包含 CVSS 评分和披露时间线。

宪法自我批评运行是一种训练阶段的干预。取1000个有害尝试提示，让目标模型草拟回答，依据书面宪法（不为害规则）进行批评，并在批评循环上进行再训练。在一个保留评估集上测量攻击前后的无害性改善量。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 内容安全、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动工具：garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- 越狱代理：PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCG 后缀
- 宪法训练：Anthropic 风格的自我批评循环 + 对批评进行 SFT
- PII 擦除：Presidio
- 目标：8B指令调优模型或其他顶点项目的 RAG 聊天机器人

## 构建步骤

1. **目标设置。** 在 vLLM 上部署一个8B指令调优模型（或复用其他顶点项目的 RAG 聊天机器人）。这是待测应用。

2. **安全流水线包装。** 将五层流水线包裹在目标周围。验证每一层可单独观测（在 Langfuse 中为每层创建一个跨度）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在每个小标注集上运行以建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 代理、一个 TAP 代理、一个 GCG 运行器、一个多轮攻击器和一个代码切换攻击器。每个运行在独立的队列上。

5. **攻击套件。** 六种攻击家族：(1) PAIR 自动化越狱，(2) TAP 树状攻击，(3) GCG 梯度后缀，(4) ASCII/base64/rot13 编码，(5) 多轮角色扮演，(6) 多语言代码切换。报告每个家族的成功率。

6. **宪法自我批评。** 整理1000个有害尝试提示。对于每个提示，目标草拟一个回答。一个批评者 LLM 依据书面宪法（"不为害"、"引用证据"、"拒绝非法请求"）进行评分。批评者反对的提示被重写；目标在批评改进的配对上进行微调。在保留评估集上测量攻击前后的无害性。

7. **过度拒答度量。** 在良性提示集（例如 XSTest）上跟踪假阳性率。目标在良性问题上必须保持有用性。

8. **CVSS 评分。** 对每个成功的越狱漏洞，按 CVSS 4.0（攻击向量、复杂度、影响）评分。生成披露时间表和缓解计划。

9. **测试场自动化。** 上述所有内容通过 cron 运行；发现结果写入队列；过度拒答回归警报发送到 Slack。

## 使用方法

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付成果

`outputs/skill-safety-harness.md` 是交付物。一个生产级的分层安全流水线，加上一个可复现的红队测试场，包含攻击前后的无害性改善量。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 运行6种以上攻击家族，2种以上语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批评改善量 | 保留评估集上攻击前后的无害性 |
| 20 | 文档与披露 | 带有时间线的 CVSS 评分发现 |
| 15 | 自动化与可重复性 | 所有内容通过 cron 运行并带有警报 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，并比较有无输出过滤层时的攻击成功率。

2. 添加第七种攻击家族：通过检索文档的间接提示注入。测量所需额外防御措施。

3. 实现一种"拒绝加帮助"模式：当护栏拦截时，目标提供更安全的相关答案而非直接拒绝。测量 XSTest 改善量。

4. 多语言覆盖缺口：找到一种 X-Guard 表现不佳的语言。提出针对它的微调数据集。

5. 在 30B 模型上运行宪法自我批评，并测量改善量是否随规模扩大。

## 关键术语

| 术语 | 人们说的 | 实际意义 |
|------|----------|----------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、人工审核等多个环节设置护栏 |
| Llama Guard 4 | "Meta的安全分类器" | 2026年参考的输入/输出内容分类器 |
| PAIR | "越狱代理" | Chao等人关于LLM驱动的越狱发现的论文 |
| TAP | "树状攻击" | PAIR的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗性后缀攻击 |
| 宪法自我批评 | "Anthropic风格的训练" | 目标草拟 -> 批评者评分 -> 重写 -> 再训练 |
| XSTest | "良性探针集" | 用于过度拒答回归的基准 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分系统 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像+多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132种语言的多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱代理论文
