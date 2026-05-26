# Capstone 07 — 端到端微调流水线（数据→SFT→DPO→服务）

> 一个在你自己的数据上训练的8B模型，根据你自己的偏好进行DPO对齐，经过量化、推测性解码，并以可度量的$/百万token提供服务。2026年的开源栈是Axolotl v0.8、TRL 0.15、用于迭代的Unsloth、用于量化的GPTQ/AWQ/GGUF、以及配备EAGLE-3的vLLM 0.7。这个capstone的目标是让整个流水线可复现——YAML输入，服务端点输出——并在2026年模型开放框架下发布一个模型卡片。

**类型:** Capstone  
**语言:** Python（流水线）、YAML（配置）、Bash（脚本）  
**前提条件:** 阶段2（ML）、阶段3（DL）、阶段7（transformers）、阶段10（从头训练LLM）、阶段11（LLM工程）、阶段17（基础设施）、阶段18（安全）  
**涉及的阶段:** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**时间:** 35小时

## 问题

到2026年，每个严肃的AI团队都会保留一条随时可用的微调流水线。不是因为他们会发布一个前沿基础模型，而是因为下游适应——领域SFT、针对标记偏好的DPO、用于推测性解码的蒸馏草稿、使用EAGLE-3提供服务——才是可衡量的胜利所在。Axolotl v0.8处理多GPU SFT配置。TRL 0.15处理DPO和GRPO。Unsloth让你快速进行单GPU迭代。配备EAGLE-3的vLLM 0.7在不损失质量的情况下将解码吞吐量提升2-3倍。工具已经就绪；技艺体现在YAML文件、数据卫生和评估纪律中。

你将在特定任务的数据上运行一个8B基础模型（Llama 3.3、Qwen3或Gemma 3），依次进行SFT和DPO，量化以供服务，并使用lm-evaluation-harness、RewardBench-2、MT-Bench-v2和MMLU-Pro来测量增益。你将根据2026年模型开放框架制作一个模型卡片。关键在于可复现性——一条命令就能端到端地重新运行整个流水线。

## 概念

该流水线包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC风格的分类器）、PII清洗、针对公开基准测试污染的分割卫生检查。**SFT**：Axolotl YAML、8×H100上的ZeRO-3、余弦调度、打包序列、2-3个epoch。**DPO或GRPO**：TRL配置、1个epoch、偏好对（人工标注或模型评判）、beta调优。**量化**：GPTQ + AWQ + GGUF以获得部署灵活性。**服务**：配备EAGLE-3推测性head的vLLM 0.7（或配备SpecForge的SGLang）、Kubernetes部署、基于队列等待的HPA。

可交付物是消融实验：在三个特定任务基准上比较SFT-only vs SFT+DPO vs SFT+GRPO。服务指标：批次1/8/32下的token/s、EAGLE-3接受率、$/百万token。安全评估：Llama Guard 4通过率。模型卡片：偏差评估、可复现种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 堆栈

- 数据：用于去重的Datatrove、用于质量的Nemotron-CC分类器、用于PII的Presidio
- 基础模型：Llama 3.3 8B、Qwen3 14B或Gemma 3 12B
- SFT：配备ZeRO-3、Flash Attention 3、打包序列的Axolotl v0.8
- 偏好调优：用于DPO或GRPO的TRL 0.15；用于单GPU迭代的Unsloth
- 量化：GPTQ（Marlin）、AWQ、通过llama.cpp的GGUF
- 服务：配备EAGLE-3推测性解码的vLLM 0.7（或SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA设备插件、基于队列等待指标的HPA
- 可观测性：用于训练的W&B、用于推理的Langfuse

## 构建它

1. **数据流水线。** 在原始语料库上运行Datatrove去重。应用Nemotron-CC风格的质量分类器。Presidio清洗PII。使用明确种子写入训练/验证分割。

2. **污染检查。** 对于每个验证分割，计算与MMLU-Pro、MT-Bench-v2、RewardBench-2测试集的MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML配置ZeRO-3、FA3、序列打包。在8×H100上训练2-3个epoch。记录到W&B。

4. **TRL DPO / GRPO。** 使用SFT检查点，在偏好对（或使用可验证奖励的数学/代码GRPO）上运行一个epoch的DPO。搜索beta。

5. **量化。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、用于llama.cpp的GGUF-Q4_K_M。记录大小和名义吞吐量。

6. **使用推测性解码提供服务。** vLLM 0.7配置，搭配通过Red Hat Speculators训练的EAGLE-3草稿head。在批次1/8/32下测量接受率和尾部延迟。报告与Anthropic / OpenAI在相同评估上的$/百万token。

7. **评估矩阵。** 在基础模型、SFT-only、SFT+DPO、SFT+GRPO上运行lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成一个表格。

8. **安全评估。** 在开发集上的Llama Guard 4通过率。ShieldGemma-2输出过滤器。

9. **模型卡片。** MOF 2026模板：数据、训练、评估、安全、许可、可复现部分（包含YAML和commit SHA）。

## 使用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付它

`outputs/skill-finetuning-pipeline.md` 描述可交付物。一条命令运行数据→SFT→DPO→量化→服务→评估，并输出一个模型卡片以及已服务的端点。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 相对于基础模型的评估增益 | 在目标任务（MMLU-Pro、MT-Bench-v2、特定任务）上测量的增益 |
| 20 | 流水线可复现性 | 一条命令以相同种子端到端重新运行 |
| 15 | 数据卫生 | 去重率、PII清洗覆盖率、污染检查绿色 |
| 20 | 服务效率 | bs=1/8/32下的token/s、EAGLE-3接受率、$/百万token |
| 15 | 模型卡片 + 安全评估 | 2026 MOF完整性 + Llama Guard 4通过率 |
| **100** | | |

## 练习

1. 在同一个特定任务基准上运行SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及差距是多少。

2. 将Llama 3.3 8B替换为Qwen3 14B。在匹配质量下测量$/百万token。

3. 测量EAGLE-3在领域数据与通用ShareGPT上的接受率。报告差异及其对延迟预算的意义。

4. 注入1%的污染（将MMLU-Pro答案泄露到训练数据中）并重新运行评估。观察MMLU-Pro准确率不切实际地提升。构建一个污染检查CI门控来捕获这种情况。

5. 添加LoRA SFT作为全微调的替代方案。在10倍更低内存下测量质量差距。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Axolotl | “SFT训练器” | 统一的YAML驱动的训练器，支持SFT、DPO和蒸馏 |
| TRL | “偏好调优器” | Hugging Face库，用于LLM的DPO、GRPO、PPO |
| GRPO | “组相对策略优化” | DeepSeek R1的带有可验证奖励的RL方法 |
| EAGLE-3 | “推测性解码草稿” | 预测N个后续token的草稿head；vLLM用目标模型验证 |
| MOF | “模型开放框架” | 2026年标准，用于根据数据、代码、许可对模型发布进行分级 |
| 污染检查 | “分割卫生” | 基于MinHash的测试集泄露到训练集的检测 |
| 接受率 | “EAGLE / MTP指标” | 目标模型接受的草稿token占比 |

## 进一步阅读

- [Axolotl文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考SFT / DPO训练器
- [TRL文档](https://huggingface.co/docs/trl) — DPO和GRPO参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单GPU迭代参考
- [DeepSeek R1论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPO方法论
- [vLLM + EAGLE-3文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选推测性解码训练器
- [模型开放框架2026](https://isocpp.org/) — 开放发布分级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器
