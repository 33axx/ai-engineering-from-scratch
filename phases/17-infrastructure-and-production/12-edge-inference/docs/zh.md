# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 核心的边缘约束是内存带宽，而非算力。移动端 DRAM 带宽为 50-90 GB/s；数据中心 HBM3 则达到 2-3 TB/s — 相差 30-50 倍。解码是内存受限的，因此这一差距是决定性的。到 2026 年，局面将分为四条赛道。Apple M4/A18 Neural Engine 峰值达 38 TOPS，统一内存（无需 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以 ~41 tok/s 运行 Llama 3.1 8B (Q4)（约为原生性能的 70-80%）；GitHub 星标 17.6k，兼容 OpenAI API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super (8GB) 可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 可达 ~40 tok/s；Jetson T4000 (JetPack 7.1) 性能是 AGX Orin 的两倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill — 已在 CES 2026 由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 学习  
**语言：** Python (stdlib, 玩具级带宽受限解码模拟器)  
**先修知识：** 阶段 17 · 04 (vLLM Serving 内部原理)、阶段 17 · 09 (生产环境量化)  
**时间：** ~60 分钟

## 学习目标

- 解释为何移动端 LLM 推理是内存带宽受限的，而算力是次要的。
- 列举四种边缘目标平台 (Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson) 并为每种平台匹配一个用例。
- 说出 2026 年 WebGPU 覆盖缺口 (Firefox Android 追赶中) 以及 Safari iOS 26 的落地情况。
- 针对每种目标平台选择量化格式 (ANE 用 Core ML INT4 + FP16、Hexagon 用 QNN INT8/INT4、浏览器用 WebGPU Q4、Jetson Thor 用 NVFP4)。

## 问题

客户需要一个设备端聊天机器人：语音优先、默认私密、离线可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行速度约为 ~55 tok/s — 没问题。在 iPhone 16 Pro 上，同样的模型只有 3 tok/s — 不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 设备上，7 tok/s。在浏览器中通过 WebGPU 在 Chrome Android v121+ 上，4-8 tok/s（取决于设备）。

吞吐量的差异并非移植问题。而是带宽差距乘以量化格式，再加上 NPU 是否可从用户空间访问的结果。2026 年的边缘推理是四个不同的问题，需要四种不同的解决方案。

## 概念

### 带宽是真正的天花板

解码每个 token 都需要读取完整的权重集。一个 7B 模型在 Q4 下大小为 3.5 GB。以 50 GB/s 的速度读取 3.5 GB 需要 70 毫秒 — 理论天花板约为 ~14 tok/s。在 90 GB/s（高端移动 DRAM）下，天花板升至 ~25 tok/s。低于这个数字，再多的算力也无济于事。

数据中心 HBM3 在 3 TB/s 下读取同样的 3.5 GB 只需 1.2 毫秒 — 天花板为 830 tok/s。同一个模型，同样的权重。不同的内存子系统。

### Apple Neural Engine (M4 / A18)

- 最高 38 TOPS。统一内存（CPU 与 ANE 共享同一内存池）— 无拷贝开销。
- 通过 Core ML + 编译的 `.mlmodel` 模型访问，或通过 PyTorch 的 Metal Performance Shaders (MPS) 访问。
- Llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：使用 INT4 权重 + FP16 激活的 Core ML。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 最高 45 TOPS。与 CPU 和 GPU 集成在 SoC 中，但拥有独立的内存域。
- QNN (Qualcomm Neural Network) SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- 聊天模板、Llama 3.2、Phi-3 均作为一流工件在 AI Hub 上提供。

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。软件支持落后于 Apple/Qualcomm；OpenVINO 正在改进但仍属小众。
- 最适合 Windows ARM Copilot 应用；在 AMD/Intel 桌面端用于本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- 在 M3 Max 上 Llama 3.1 8B Q4 约 ~41 tok/s — 约为同一后端原生性能的 70-80%。
- WebLLM 在 GitHub 上有 17.6k 星标；提供兼容 OpenAI 的 JS API；采用 Apache 2.0 许可证。
- 2026 年覆盖情况：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。整体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super (8GB)：可运行 Llama 3.2 3B、Phi-3，tok/s 表现良好。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 ~40 tok/s。
- Thor / T4000 (JetPack 7.1)：性能是 AGX Orin 的两倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM (2026) 支持 EAGLE-3 推测解码、NVFP4 权重、chunked prefill — 将数据中心优化移植到边缘。

### 按目标平台选择量化格式

| 目标平台 | 格式 | 备注 |
|--------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘环境中的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心功能。在 8 GB 内存的手机上，4 GB 模型 + 2 GB KV 缓存（用于 32K tokens）+ 操作系统开销 = 内存溢出。边缘部署通常将上下文保持在 4K-8K，除非接受激进的 KV 量化 (Q4 KV)。

### 语音是杀手级应用

语音智能体对延迟敏感（首 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文字（Whisper Turbo 变体可在边缘运行），边缘推理便成为生产级语音回路。

### 你应该记住的数字

- Apple M4 / A18 ANE: 38 TOPS。
- Qualcomm Hexagon SD X Elite: 45 TOPS。
- WebLLM on M3 Max: Llama 3.1 8B Q4 约 ~41 tok/s。
- AGX Orin: 通过 vLLM 运行 gpt-oss-20b 约 ~40 tok/s。
- 数据中心与边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 滞后）。

## 使用

`code/main.py` 根据带宽受限的数学计算，计算各边缘目标平台的理论解码吞吐量天花板。并与观察到的基准测试进行比较，突出显示带宽而非算力才是瓶颈。

## 交付

本次课程产出 `outputs/skill-edge-target-picker.md`。给定平台 (iOS/Android/browser/Jetson)、模型、延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于搭载 Snapdragon 8 Gen 3（带宽约 77 GB/s）的 7B 模型（Q4），计算解码天花板。与观察到的 6-8 tok/s 相比 — 运行时效率高吗？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计一个降级方案 — 通过同一兼容 OpenAI 的 API 切换到服务端。
3. 你的 iOS 应用需要 4K 上下文的流式处理。在 iPhone 16 上，哪些模型/格式组合能让活跃内存保持在 4 GB 以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能容纳 3B 模型。如果你的产品同时面向两者，如何统一推理栈？
5. 论证“WebLLM 在 2026 年已达到生产级”这一观点。引用覆盖率、性能以及 Firefox Android 的差距。

## 关键术语

| 术语 | 人们常说的意思 | 实际含义 |
|------|----------------|----------|
| ANE | "Apple 神经网络引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；兼容 OpenAI 的 JS 接口 |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 的边缘移植版；支持 EAGLE-3 + NVFP4 |
| Unified memory | "共享内存池" | CPU 和 NPU 看见同一块 RAM；无拷贝开销 |
| Bandwidth-bound | "内存受限" | 解码受限于读取权重的字节/秒 |
| Core ML | "Apple 转换" | Apple 框架，用于 ANE 原生的模型 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

## 扩展阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 全景分析与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 边缘端移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生模型转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 已转换的 Hexagon 模型。
