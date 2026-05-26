# 推理平台经济学——Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026年的推理市场不再是GPU时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU平台（Baseten、Together、Fireworks、Modal）和API优先市场（Replicate、DeepInfra）。Fireworks于2026年5月1日将每GPU每小时价格上调1美元，而$4B估值、每日处理10T+ token的事实说明，基于规模的模式是有效的。Baseten于2026年1月以$5B估值完成$300M E轮融资。竞争定位规则很简单：Fireworks优化延迟，Together优化目录广度，Baseten优化企业级调优，Modal优化Python原生开发者体验，Replicate优化多模态覆盖，Anyscale优化分布式Python。本课程为你提供一份可直接交给创始人的矩阵。

**类型：** 学习  
**语言：** Python（stdlib，玩具级每次调用经济学比较器）  
**前置知识：** Phase 17 · 01（托管LLM平台），Phase 17 · 04（vLLM服务内部机制）  
**时长：** ~60分钟

## 学习目标

- 说出三个市场细分（定制芯片、GPU平台、API优先），并将每个供应商映射到一个细分。
- 解释为什么“按token计费”的API定价模型压缩到服务引擎的成本曲线上，而不是硬件上。
- 计算至少三个供应商的每次请求有效成本，并说明按分钟计费（Baseten、Modal）何时优于按token计费。
- 针对给定的工作负载（无服务器突发型、稳定高吞吐、微调变体、多模态），确定哪个平台是正确的默认选择。

## 问题

你评估了托管超大规模平台。你决定需要一个更窄、更快的供应商——Fireworks（低延迟）、Together（广度）、Baseten（微调定制模型）。现在你有六个真实选择，但定价页面并不对齐。Fireworks显示$/M token；Baseten显示$/分钟；Modal显示$/秒；Replicate显示$/次预测。如果不建模工作负载，你无法直接比较。

更糟的是，每个定价页面背后的商业模型都不同。Fireworks在共享GPU上运行自己的定制引擎（FireAttention）；按token费率反映了其利用率曲线。Baseten提供Truss + 专用GPU；按分钟计费反映了独占性。Modal是真正的Python无服务器——按秒计费，亚秒级冷启动。相同输出（LLM响应），三种不同的成本函数。

本课程对六个供应商进行建模，并告诉你各自何时胜出。

## 概念

### 三个细分

**定制芯片** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在相同模型上，解码速度通常比基于GPU的集群快5-10倍。按token价格更高（Groq在2025年底对Llama-70B约为$0.99/M），但在延迟敏感型用例中无可匹敌。Groq是语音代理和实时翻译的生产选择。

**GPU平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在NVIDIA（H100、H200、B200，截至2026年）或有时是AMD之上。经济层位于“原始GPU租赁”（RunPod、Lambda）和“超大规模托管服务”（Bedrock）之间。

**API优先市场** — Replicate、DeepInfra、OpenRouter、Fal。目录广泛，按预测或按秒付费，强调首次调用时间。

### Fireworks——延迟优化的GPU平台

- FireAttention引擎（自研）；营销称在同等配置下延迟比vLLM低4倍。
- 批量层价格约为无服务器价格的50%，适用于非交互式工作负载。
- 微调模型以与基础模型相同的速率提供服务——这与那些对LoRA收取额外费用的供应商相比是一个真正的差异化优势。
- 2026年中：自2026年5月1日起，按需GPU租赁有效提价每小时1美元。大规模使用时价格可协商。
- 财务信号：$4B估值，每日处理10T+ token。

### Together——广度优化的平台

- 200多个模型，包括上游发布后数天内即开源的模型。
- 在同等LLM模型上比Replicate便宜50-70%——“AI原生云”的定位是规模和目录。
- 推理 + 微调 + 训练，统一API。

### Baseten——企业级调优优化的平台

- Truss框架：模型打包，包括依赖项、密钥、服务配置，全部在一个清单中。
- GPU范围从T4到B200。按分钟计费，具备合理的冷启动缓解措施。
- SOC 2 Type II，HIPAA就绪。常见的金融科技和医疗选择。
- $5B估值，2026年1月E轮融资（CapitalG、IVP、NVIDIA领投$300M）。

### Modal——Python原生优化的平台

- 纯Python的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一个命令即可部署。
- 按秒计费。冷启动时间2-4秒（预预热）；小模型<1秒。
- $87M B轮融资，$1.1B估值（2025年）。在独立调查中开发者体验评分最高。

### Replicate——多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS插件）。
- 在LLM按token费率上竞争力较弱，但在多模态多样性方面胜出。

### Anyscale——Ray原生平台

- 基于Ray；RayTurbo是Anyscale的专有推理引擎（与vLLM竞争）。
- 最适合推理步骤是更大图中的一个节点的分布式Python工作负载。
- 托管Ray集群；与Ray AIR和Ray Serve紧密集成。

### 按token计费与按分钟计费——各自何时胜出

按token计费适用于工作负载对延迟不敏感且具有突发性——你只需为实际使用付费。按分钟计费适用于利用率高且可预测——一旦GPU饱和，你就能胜过按token计费。

粗略规则：对于专用GPU持续利用率约30%以上的工作负载，按分钟计费（Baseten、Modal）开始优于按token计费（Fireworks、Together）。低于此值，按token计费胜出，因为你避免了为空闲付费。

### 定制引擎才是真正的护城河

每个基于vLLM和SGLang的平台都声称拥有定制引擎。FireAttention、RayTurbo、Baseten的推理栈。定制引擎的说法带有营销色彩——诚实的表述是vLLM + SGLang大约占生产环境开源推理的80%，而平台层的差异化在于开发者体验、归因和服务水平协议。

### 你应该记住的数字

- Fireworks GPU租赁：2026年5月1日起有效涨价$1/小时。
- Fireworks声称：同等配置下延迟比vLLM低4倍。
- Together：在LLM上比Replicate便宜50-70%。
- Baseten估值：$5B（2026年1月E轮，$300M round）。
- Modal估值：$1.1B（2025年B轮）。
- 持续利用率约30%以上时，按分钟计费优于按token计费。

## 使用它

`code/main.py` 在一个合成工作负载上比较六个供应商的定价模型。报告 $/天 和有效 $/M token。运行它以找出按token计费和按分钟计费之间的盈亏平衡点。

## 交付

本课程产出 `outputs/skill-inference-platform-picker.md`。给定工作负载概况、SLA和预算，选择主要的推理平台并提名备选。

## 练习

1. 运行 `code/main.py`。对于单个H100上的70B模型，Baseten（按分钟计费）在多少持续利用率下会优于Fireworks（按token计费）？自己推导交叉点并与经验法则比较。
2. 你的产品提供图像生成+聊天+语音转文本。为每种模态选择平台，并说出统一它们的网关模式。
3. Fireworks将你主要模型的价格提高了$1/小时。如果40%的流量转移到批量层（五折），建模混合成本影响。
4. 一个受监管客户要求SOC 2 Type II + HIPAA + 专用GPU。哪三个平台可行，哪个在FinOps上胜出？
5. 比较Llama 3.1 70B在Fireworks无服务器、Together按需、Baseten专用和Replicate API上的每1000次预测成本。在每天10次预测时哪个最便宜？在每天10000次时呢？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| 定制芯片 | "非GPU芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——专为解码优化 |
| FireAttention | "Fireworks引擎" | 自定义注意力内核；营销称延迟比vLLM低4倍 |
| Truss | "Baseten的格式" | 模型打包清单；依赖项+密钥+服务配置 |
| 按token计费 | "API定价" | 按消耗的token收费；不为空闲付费 |
| 按分钟计费 | "专用定价" | 按挂钟GPU时间收费；高利用率时胜出 |
| 按预测计费 | "Replicate定价" | 按模型调用收费；常见于图像/视频 |
| RayTurbo | "Anyscale引擎" | Ray上的专有推理；与Ray集群上的vLLM竞争 |
| 批量层 | "五折" | 以较低费率的非交互式队列；常见于Fireworks、OpenAI |
| 微调按基础费率 | "Fireworks LoRA" | 以基础模型费率对LoRA服务请求收费（差异化优势） |

## 扩展阅读

- [Fireworks定价](https://fireworks.ai/pricing) — 按token费率、批量层、GPU租赁。
- [Baseten定价](https://www.baseten.co/pricing/) — 按分钟费率、预留容量、企业层。
- [Modal定价](https://modal.com/pricing) — 按秒GPU费率和免费层。
- [Together AI定价](https://www.together.ai/pricing) — 模型目录和按token费率。
- [Anyscale定价](https://www.anyscale.com/pricing) — RayTurbo和托管Ray定价。
- [Northflank — Fireworks AI替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — 2026年AI推理API供应商对比](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。
