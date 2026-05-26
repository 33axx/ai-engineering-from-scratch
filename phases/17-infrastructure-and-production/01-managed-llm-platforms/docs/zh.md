# Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI

> 三大超大规模云平台，三种截然不同的策略。AWS Bedrock 是一个模型市场——Claude、Llama、Titan、Stability、Cohere 统一在一个 API 下。Azure OpenAI 是 OpenAI 的独家合作伙伴，加上预置吞吐单元 (PTU) 提供专用容量。Vertex AI 以 Gemini 为核心，拥有最佳的长上下文和多模态能力。到 2026 年，Artificial Analysis 测量显示 Azure OpenAI 中位数延迟约 50 ms，Bedrock 在等效 Llama 3.1 405B 上约 75 ms——PTU 解释了这一差距，因为专用容量优于共享按需。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps 表面匹配我的产品”。本课程教你基于已写明的权衡（而非直觉）进行选择。

**类型:** 学习  
**语言:** Python (stdlib, 简易成本与延迟比较器)  
**先修要求:** 阶段 11 (LLM 工程), 阶段 13 (工具与协议)  
**时间:** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场 vs 独家 vs 以 Gemini 为核心）并将每种策略匹配到相应的产品用例。
- 解释 Azure OpenAI 中的预置吞吐单元 (PTU) 能带来什么，以及为什么按需的 Bedrock 在 405B 规模下通常慢约 25 ms。
- 绘制每个平台的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex 每团队项目 vs Azure 作用域 + PTU 预留）。
- 写下“至少两个提供商”的策略，并解释为何单一供应商锁定在 2026 年是一个代价高昂的错误。

## 问题

你为你的产品选择了 Claude 3.7 Sonnet。现在需要提供该服务。你可以直接调用 Anthropic API，或通过 AWS Bedrock 调用，或通过网关调用。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关提供了故障转移、统一计费和跨提供商的速率限制。

更深层次的问题是目录。如果你在同一产品中需要 Claude、Llama 和 Gemini，你无法从一个地方购买所有这些，除非那个地方同时是 Bedrock、Vertex 和 Azure OpenAI。这些超大规模平台不可互换——它们各自对谁拥有模型层做出了不同的赌注。

本课程映射了这三个赌注、延迟差距、FinOps 差距以及锁定风险。

## 概念

### 三种策略

**AWS Bedrock** —— 市场。包含 Claude (Anthropic)、Llama (Meta)、Titan (AWS 第一方)、Stability (图像)、Cohere (嵌入)、Mistral，以及图像和嵌入子目录。统一 API，统一 IAM 面，统一 CloudWatch 导出。Bedrock 的赌注是客户更想要可选择性而非单一模型。

**Azure OpenAI** —— 独家合作。你获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心对 OpenAI 模型的微调。Azure OpenAI Service 目录中没有非 OpenAI 模型——这些模型属于 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 仍处于前沿地位，而客户希望在该特定关系上拥有企业级控制。

**Vertex AI** —— 以 Gemini 为核心，其余为辅。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，再加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文——百万 token 的 Gemini 上下文是差异化优势。

### 规模上的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位数首 token 延迟约为 50 ms；Bedrock 约为 75 ms。这一差距并非 AWS 的失败——而是容量模型的不同。Azure 销售 PTU（预置吞吐单元），为你的租户预留 GPU 容量。Bedrock 的等效产品（预置吞吐）确实存在，但每单元约 21 美元/小时起，且大多数客户仍使用共享按需。

共享按需容量需与所有其他客户的流量竞争。专用容量则不需要。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock 预置吞吐，要么接受默认的方差。

### 预置吞吐的经济学

Azure PTU：一个预留的推理计算块。对于可预测的工作负载，相比按需可节省高达约 70%。无论流量如何，按固定小时收费——即使空闲也要为预留付费。盈亏平衡点通常在 40-60% 的持续利用率。

Bedrock 预置吞吐：每模型、每区域 21-50 美元/小时。类似的计算——盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 的预置容量按 Gemini SKU 销售；定价因模型和区域而异，且不太公开。

### FinOps 表面——真正的差异化因素

**Bedrock Application Inference Profiles** 是市场上最清晰的归因方式。使用 `team`、`product`、`feature` 标签标记一个配置文件；所有模型调用都通过它路由；CloudWatch 无需后处理即可按配置文件分解成本。于 2025 年添加，仍然是超大规模云原生中最细粒度的。

**Vertex** 的归因方式是每团队项目加处处加标签。你将每个团队建模为一个 GCP 项目，在每个资源上放置标签，并使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许你对成本数据执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非请求，因此按请求归因需要 Application Insights 自定义指标或一个在头部打上标记的网关。

模式：Bedrock 的原生方式最清晰，Vertex 通过 BigQuery 最灵活，Azure 最不透明除非你进行检测。

### 锁定是 2026 年的风险

当一个模型占主导时，单一超大规模平台承诺是可以的。但 2026 年前沿每月变动——上一季度是 Claude 3.7，下一季度是 Gemini 2.5，再下一季度是 GPT-5。锁定在一个平台意味着被锁定在三分之二的前沿之外。

正在采用的团队的模式：对于任何产品关键的 LLM 调用，采用至少两个提供商。Bedrock 加 Azure OpenAI 是常见的组合——一个来源得到 Claude，另一个得到 GPT，两者之间故障转移，相同网关。成本增加可忽略不计，因为网关进行最优路由；在中断期间（如 2025 年 1 月的 Azure OpenAI 事件、AWS us-east-1 中断）的可用性提升是决定性的。

### 数据驻留、BAA 与受监管行业

Bedrock：大多数区域提供 BAA；VPC 端点；护栏。常见的金融科技默认选项。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选项。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规栈。

三者都满足基本要求。差异在于数据保留策略、日志处理方式以及滥用监控是否读取你的流量（大多数默认启用；企业可关闭）。

### 你应该记住的数字

- Azure OpenAI 在等效 Llama 3.1 405B 上的中位数 TTFT（使用 PTU）：约 50 ms。
- Bedrock 按需中位数 TTFT：约 75 ms。
- Bedrock 预置吞吐：每单元 21-50 美元/小时。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需节省：高达 70%。

## 使用它

`code/main.py` 在合成工作负载上比较三个平台——它模拟了按需与 PTU 的经济性、TTFT 方差以及成本归因保真度。运行它，看看 PTU 何时付出回报，以及市场的模型广度何时胜过 TTFT 差距。

## 发布它

本课程生成 `outputs/skill-managed-platform-picker.md`。给定工作负载描述（所需模型、TTFT SLA、每日量、合规要求），它将推荐一个主平台、一个备用平台以及一个 FinOps 检测计划。

## 练习

1. 运行 `code/main.py`。对于 70B 类模型，Azure PTU 在什么持续利用率下优于按需？计算盈亏平衡点并与宣传的 40-60% 范围进行比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双提供商部署——哪个模型放在哪个超大规模平台上，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAA、美东数据驻留以及低于 100ms 的 P99 TTFT。选择一个平台并用三个具体特性证明其合理性。
4. 你发现你的 Bedrock 账单本月上涨了 4 倍，但流量没有变化。如果没有 Application Inference Profiles，你如何找到原因？使用 Profiles，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于一个 1 亿 token/月的 Claude 工作负载，哪个更便宜——直接 Anthropic API、Bedrock 按需还是 Bedrock 预置吞吐？

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|----------------|------------------------|
| Bedrock | “AWS LLM 服务” | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 的 ChatGPT” | 在 Azure 数据中心内的独家 OpenAI 模型，带有企业控制功能 |
| Vertex AI | “Google 的 LLM” | 以 Gemini 为核心的平台，附带 Model Garden 提供第三方模型 |
| PTU | “专用容量” | 预置吞吐单元——预留的推理 GPU，按小时定价 |
| Application Inference Profile | “Bedrock 标签” | 带标签的按产品/按用途成本和使用配置文件，CloudWatch 原生支持 |
| Model Garden | “Vertex 目录” | Vertex AI 的第三方模型部分，与 Gemini 分开 |
| 双提供商最低要求 | “LLM 冗余” | 策略：每个关键 LLM 路径至少跨越两个超大规模平台 |
| BAA | “HIPAA 文件” | 业务伙伴协议；处理受保护健康信息所需；三个平台都提供 |
| 滥用监控 | “日志监视器” | 平台侧对提示/输出的安全扫描；企业可选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) —— 官方费率卡和预置吞吐定价。
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) —— PTU 经济学和费率卡。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) —— Gemini 层级和 Model Garden 附加费用。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) —— 跨提供商的持续延迟和吞吐量基准。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) —— 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) —— 归因机制并排比较。
