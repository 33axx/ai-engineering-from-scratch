# 专题项目 12 — 视频理解流程（场景、问答、搜索）

> Twelve Labs 已将 Marengo + Pegasus 产品化。VideoDB 提供了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开放 VLM 检查点。Gemini 长上下文原生处理数小时的视频。TimeLens-100K 定义了大规模时序定位。2026 年的流程已定型：场景分割、逐场景描述与嵌入、文本对齐、多向量索引，以及一个能返回（起始、结束）时间戳和帧预览的查询。该专题项目将摄入 100 小时的视频，在公开基准上进行测试，并测量在计数和动作类问题上的幻觉率。

**类型：** 专题项目
**语言：** Python（流程）、TypeScript（UI）
**前置要求：** 阶段 4（计算机视觉）、阶段 6（语音）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

长视频问答是 2026 年规模下带宽消耗最大的多模态问题。Gemini 2.5 Pro 原生可读取 2 小时的视频，但将 100 小时的视频摄入到可查询的语料库中，仍然需要一个场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、基于 VLM 的逐场景描述（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、文本对齐（带单词时间戳的 Whisper-v3-turbo）以及一个多向量索引，该索引同时存储描述、帧嵌入和文本。查询流程会返回（起始、结束）时间戳和帧预览。

基准评测包括公开数据集（ActivityNet-QA、NeXT-GQA）以及你自建的 100 条查询集。计数和动作类问题上的幻觉是已知的困难失败模式；该专题项目将明确测量它。

## 概念

摄入时三个流程并行运行。**场景分割**将视频切割成场景。**VLM 描述生成**为每个场景生成一条描述以及来自关键帧的帧嵌入。**ASR 对齐**生成词级时间戳。三条流通过（scene_id, time range）连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：描述嵌入、关键帧嵌入、文本嵌入。

查询时，自然语言问题对三个向量分别进行密集检索；使用倒数排名融合（RRF）合并结果；一个时序定位适配器（类似 TimeLens）在最佳场景内细化（起始、结束）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询、最佳场景及其裁剪帧和文本，然后返回带有引用时间戳和帧预览的答案。

幻觉测量非常重要。计数类（“多少人进入房间？”）和动作类（“厨师是先倒再搅拌吗？”）问题通常不可靠。请分别报告其准确率，而不是与描述类问题混在一起。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-2026 年最先进）或 PySceneDetect
- ASR：通过 faster-whisper 使用带词级时间戳的 Whisper-v3-turbo
- VLM 描述生成与回答：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时序定位：基于 TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持多向量的 Qdrant（描述/帧/文本）
- UI：Next.js 15，含 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA，以及自建 100 条手工标注问题集
- 幻觉基准：计数和动作类型的子集，带手工标注

## 构建步骤

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。如有需要，降至 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 场景。

3. **ASR 流程。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；按场景切分为文本片段。

4. **VLM 描述生成。** 对每个场景，使用关键帧和简短描述模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成描述和帧嵌入。

5. **多向量索引。** Qdrant 集合，包含三个命名向量。负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题对三个向量进行密集检索；使用倒数排名融合（RRF）合并；取 top-k=5 个场景。

7. **时序定位。** 对最佳场景运行 TimeLens 风格适配器，在该场景内细化（起始、结束）窗口。

8. **VLM 合成。** 用查询 + 最佳 3 个场景（以图像或短视频片段形式）加上文本调用 Gemini 2.5 Pro。要求返回 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 条自定义查询集。报告总体准确率以及按类别（计数、动作、描述）分解的准确率。

## 使用方式

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付物

`outputs/skill-video-qa.md` 是交付产物。给定一个 YouTube URL 或上传的视频，流程索引场景并回答带有时间戳引用的问题。

| 权重 | 标准 | 测量方式 |
|:---:|------|----------|
| 25  | 时序定位 IoU | 在保留的定位集上的交并比 |
| 20  | QA 准确率 | NeXT-GQA 和自定义 100 条查询 |
| 20  | 摄入吞吐量 | 每美元对应的视频小时数 |
| 20  | UI 与引用体验 | 时间戳链接、缩略图条、跳转至帧 |
| 15  | 幻觉率 | 分别统计计数类和动作类准确率 |
|**100**| | |

## 练习

1. 将描述生成流程中的 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评分的 50 场景样本上报告描述质量差异。

2. 将每场景的帧嵌入降为一个合并向量，而不是多向量。测量检索性能的退化。

3. 构建一个“严格计数”模式：合成器为每个被计数的实例提取时间戳，用户点击验证。测量用户验证是否降低幻觉率。

4. 基准测试摄入成本：在三种 VLM 选择间比较每美元对应的视频小时数。找出性价比最佳点。

5. 添加说话人分离文本：在音频上运行 pyannote 说话人分离，并为每个说话人嵌入文本。演示“爱丽丝关于 X 说了什么？”这类查询。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 场景分割 | “镜头检测” | 在镜头边界处将视频切割成场景 |
| 多向量索引 | “描述 + 帧 + 文本” | 每个表示对应一个命名向量的 Qdrant 集合 |
| 时序定位 | “具体什么时候发生的” | 为查询答案细化（起始、结束）窗口 |
| 帧嵌入 | “视觉表示” | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | “倒数排名融合” | 跨多个排序列表的合并策略；经典的混合检索技巧 |
| 计数幻觉 | “数错” | VLM 在“有多少个 X”问题上的已知失败模式 |
| ActivityNet-QA | “视频问答基准” | 长视频问答准确率基准 |

## 扩展阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时序定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管参考实现
- [VideoDB](https://videodb.io) — 面向视频的 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
