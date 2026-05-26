# 生产环境量化 — AWQ、GPTQ、GGUF K-量化、FP8、MXFP4/NVFP4

> 量化格式并非通用选择——它取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘设备，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中胜出，当你在同一基座上需要多 LoRA 时。AWQ 配合 Marlin-AWQ 内核，在 7B 类模型上可实现约 741 tok/s，并在 INT4 下拥有最佳 Pass@1——这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间立场——近乎无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）较为激进，需要逐块验证。两个陷阱常让团队受害：校准数据集必须匹配部署领域，而且 KV 缓存与权重量化是分开的——AWQ 的教训是“我的模型现在只有 4 GB”忽略了在生产批次规模下 10-30 GB 的 KV 缓存。

**类型：** 学习  
**语言：** Python（标准库，跨格式的玩具级内存和吞吐量对比）  
**前置知识：** 第 10 阶段 · 13（量化基础），第 17 阶段 · 04（vLLM 推理内部原理）  
**时间：** ~75 分钟

## 学习目标

- 列举 2026 年六种生产量化格式及其最佳适用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、推理引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规对话、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存以及未触及的 KV 缓存。
- 指出会降低量化模型在领域流量上性能的校准数据集陷阱。

## 问题

量化减少了内存和 HBM 带宽，这正是解码阶段所需要的。FP16 70B 模型权重为 140 GB。将权重量化到 INT4（AWQ 或 GPTQ）后，模型变为 35 GB——可装入一块 H100 并留出 KV 缓存空间，这一点很重要，因为在 128 并发序列、上下文长度 2k 的情况下，仅 KV 缓存就需要 20-30 GB。

但量化并非免费。激进的量化会降低质量，尤其是在推理密集型任务上。不同格式适用于不同推理引擎。不同硬件原生支持不同的精度。2026 年的格式动物园是真实存在的，你不能复制别人的选择——你必须基于自己的技术栈做出选择。

## 概念

### 六种格式

| 格式 | 比特数 | 最佳适用场景 | 推理引擎 |
|------|--------|-------------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF——CPU/边缘设备默认选择

GGUF 是一种文件格式，而非严格意义上的量化方案——它在一个容器中捆绑了 K-量化变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）。Q4_K_M 和 Q5_K_M 是生产环境默认选择——在 4-5 比特下达到接近 BF16 的质量。当部署目标是 CPU 或边缘设备时，这是最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

在 vLLM 中的吞吐量代价：7B 模型约 93 tok/s——该格式未针对 GPU 内核进行优化。当部署目标是 CPU/边缘设备时使用 GGUF。否则不要用。

### GPTQ——vLLM 中的多 LoRA

GPTQ 是一种后训练量化算法，带有一个校准步骤。Marlin 内核使其在 GPU 上快速运行（相对非 Marlin GPTQ 加速 2.6 倍）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你要部署一个基座模型加上 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ——数据中心 GPU 默认选择

激活感知权重量化。在量化过程中保护约 1% 的最显著权重。Marlin-AWQ 内核：相比朴素实现加速 10.9 倍。7B 模型约 741 tok/s，INT4 格式中 Pass@1 最佳。

对于新的 GPU 推理，选择 AWQ，除非你需要多 LoRA（此时用 GPTQ）或激进的 Blackwell FP4（此时用 NVFP4）。

### FP8——可靠的折中方案

8-bit 浮点数。近乎无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。当质量不可妥协（推理、医疗、代码生成）时，FP8 是 2026 年安全默认选择。内存节省只有 INT4 的一半，但质量风险远低。

### MXFP4 / NVFP4——Blackwell 激进方案

微缩放 FP4。每个权重块有自己的缩放因子。激进但由 Blackwell Tensor Core 硬件加速。相比 FP8 每 Token 字节数减半——这是第 17 阶段 · 07 中的经济优势。

注意事项：
- 还未支持 LoRA（2026 年初）。
- 在推理密集型工作负载上质量下降明显。
- 需要针对每个模型在评估集上验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），使用通用网页文本进行校准会使算法在保护哪些权重方面做出错误决策。HumanEval 上的 Pass@1 可能下降数个百分点。

解决方法：使用领域内数据进行校准。通常几百个领域样本就足够了。在部署前在评估集上测试。

### KV 缓存陷阱

AWQ 将权重缩小到 4 比特。KV 缓存是独立的，保持在 FP16/FP8。以 70B 模型配合 AWQ 为例：

- 权重：约 35 GB（从 140 GB INT4）。
- 128 并发 × 2k 上下文下的 KV 缓存：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB——可装入 H100 80GB。

天真地认为“我将模型量化到了 4 GB”忽略了另外 30-50 GB。需要整体规划 HBM。

另外，KV 缓存量化（FP8 KV 或 INT8 KV）是一个不同的选择，有其自身的权衡——它直接影响注意力精度，并非免费收益。

### AWQ INT4 对推理任务有风险

思维链、数学、长上下文代码生成——这些任务会明显受到激进量化的影响。AWQ INT4 在 MATH 上损失约 3-5 分。对于推理密集型工作负载，选择 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘推理：GGUF Q4_K_M。完成。
- GPU 推理，常规对话，无 LoRA：AWQ。
- GPU 推理，多 LoRA：GPTQ 配合 Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心，质量已验证：NVFP4 + FP8 KV。
- 不确定：在每种候选格式上运行 1000 样本评估。

## 使用它

`code/main.py` 计算六种格式在多种模型规模下的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV 缓存何时占主导，权重压缩何时有效，以及 FP8 何时是安全选择。

## 交付它

本节课程生成 `outputs/skill-quantization-picker.md`。根据硬件、模型规模、工作负载类型和质量容限，选择一种格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于一个 70B 模型在 128 并发、上下文长度 2k 的情况下，计算每种格式所需的总 HBM。哪种格式可以装入一块 H100 80GB？
2. 你有一个 7B 代码模型。选择一种格式并说明理由。如果你对质量容限判断错误，恢复路径是什么？
3. 计算校准一个医疗领域模型的 AWQ 所需的校准数据集大小。为什么数据更多并不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上能达到 741 tok/s，而原始 GPTQ 约为 712 tok/s。
5. 什么情况下将 AWQ 权重与 FP8 KV 缓存结合是有意义的，而保持 KV 为 BF16 呢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GGUF | "llama.cpp 格式" | 捆绑 K-量化变体的文件格式；CPU/边缘设备默认 |
| Q4_K_M | "Q4 K M" | 4-bit K-量化中等；生产环境 GGUF 默认 |
| GPTQ | "G P T Q" | 后训练 INT4 量化，带校准；vLLM 中支持 LoRA |
| AWQ | "A W Q" | 激活感知 INT4 量化；Marlin 内核；INT4 下最佳 Pass@1 |
| Marlin 内核 | "快速 INT4 内核" | 面向 Hopper 的定制 CUDA INT4 内核；加速 10 倍 |
| FP8 | "八位浮点数" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四" | Blackwell 4-bit FP，带逐块缩放因子 |
| 校准数据集 | "校准数据" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV 缓存量化 | "KV INT8" | 与权重独立的选择；影响注意力精度 |

## 扩展阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) —— 对比基准。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) —— 按格式的吞吐量数字。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) —— 逐格式选择。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) —— 支持的格式和标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) —— 原始 AWQ 公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) —— 原始 GPTQ 公式。
