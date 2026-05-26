# Capstone 04 — 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年的文档问答前沿已从 OCR-然后-文本转向视觉优先的延迟交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每一页 PDF 视为一张图像，用多向量延迟交互进行嵌入，并让查询直接关注图像块。在金融 10-K、科学论文和手写笔记上，这种模式大幅击败了 OCR 优先的方法。在 1 万页数据上端到端构建该管道，并发布与 OCR-然后-文本的对比结果。

**类型：** Capstone
**语言：** Python（管道），TypeScript（查看器 UI）
**前置条件：** 阶段 4（计算机视觉）、阶段 5（自然语言处理）、阶段 7（Transformer）、阶段 11（大语言模型工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

企业面对大量 PDF，而 OCR 管道会破坏它们：包含旋转表格的扫描版 10-K、密集公式的科学论文、只有作为图像才有意义的图表、手写注释。将这些内容视为文本优先意味着丢失一半信号。2026 年的答案是在原始页面图像上使用延迟交互的多向量检索。ColPali（Illuin Tech）引入该方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确性。在 ViDoRe v3 上，视觉优先检索的得分显著高于 OCR-然后-文本——并且在图表、表格和手写内容上差距更大。

其代价是存储和延迟。ColQwen 的嵌入约为每页 2048 个补丁向量，而不是单个 1024 维向量。原始存储急剧膨胀。DocPruner（2026 年）可在无明显精度损失的情况下实现 50% 的剪枝。你将索引 1 万页，测量 ViDoRe v3 的 nDCG@5，在 2 秒内提供答案，并直接与 OCR-然后-文本基线进行对比。

## 概念

延迟交互意味着每个查询标记对每个补丁标记进行评分，并求和每个查询标记的最高得分。你获得细粒度的匹配，而不需要单个池化向量。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个补丁的嵌入，并在检索时运行 MaxSim。

答案生成器是一个视觉语言模型，它接收查询以及作为图像的 top-k 检索页面，并写出带证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年前沿选择。对于公式和科学符号，可拼接一个 OCR 回退（Nougat、dots.ocr）作为可选的文本通道。

评估是一个二维矩阵。一个维度：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）。另一个维度：检索方法（视觉优先的延迟交互 vs OCR-然后-文本 vs 混合）。每个单元格获得 nDCG@5 和答案准确率。报告即交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF (fitz) 在 180 DPI，纵向归一化
- 延迟交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：带多向量字段的 Vespa，或 Qdrant 多向量，或带 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差补丁，在 < 0.5% 精度损失下实现 50% 压缩）
- OCR 回退（公式/密集表格）：dots.ocr 或 Nougat
- VLM 答案生成器：自托管 Qwen3-VL-30B 或托管的 Gemini 2.5 Pro；InternVL3 作为回退
- 评估：ViDoRe v3 基准测试，用于多页推理的 M3DocVQA
- 查看器 UI：Next.js 15，带证据区域的画布叠加

## 构建步骤

1. **导入。** 遍历包含 10-K、科学论文和扫描文档的 1 万页 PDF 语料库。将每页渲染为 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每页图像运行 ColQwen2.5-v0.2。输出形状为约 2048 个维度为 128 的补丁嵌入。应用 DocPruner 保留最高信号的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对每个传入查询，用查询塔（标记级嵌入）进行嵌入。对索引运行 MaxSim：对每个查询标记，取页面补丁嵌入的最大点积，求和。返回 top-k 页面。

4. **综合。** 调用 Qwen3-VL-30B，输入查询和 top-5 页面图像。提示词：“仅使用提供的页面回答。对每个主张引用 (doc_id, page)，并指出区域（图表、表格、段落）。”

5. **证据区域。** 后处理答案以提取引用的区域。如果 VLM 输出了边界框（Qwen3-VL 支持），则在查看器中将其渲染为叠加层。

6. **OCR 回退。** 对于识别为公式密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一起传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。同时在相同语料库和相同综合器上运行 OCR-然后-文本管道。生成内容类型 × 方法矩阵。

8. **UI。** 先基于 Streamlit 原型；然后使用 Next.js 15 生产环境查看器，带逐页证据区域叠加。

## 使用方法

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付要求

`outputs/skill-doc-qa.md` 描述交付物：一个针对特定语料库调优的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与 OCR-然后-文本基线进行对比评估。

| 权重 | 标准 | 衡量方法 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR-文本基线及已公布排行榜对比的基准数字 |
| 20 | 证据区域定位 | 被引用的区域中实际包含答案跨度的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、答案 p95 |
| 20 | 多页推理 | 在手动标注的 100 题多页集上的准确率 |
| 15 | 源检查 UX | 查看器清晰度、叠加保真度、并排比较工具 |
| **100** | | |

## 练习

1. 在相同语料库上比较 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面是一个正确而另一个错误？向索引添加“内容类别”标签，以便按类型路由。

2. 激进地剪枝嵌入（75%，90%）。找到压缩悬崖：当 ViDoRe nDCG@5 下降到低于 OCR 基线时的点。

3. 构建混合系统：并行运行 OCR-然后-文本和 ColQwen，用 RRF 融合，用交叉编码器重排序。混合系统是否优于任一单独系统？在哪些方面帮助最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量准确率-成本曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索。与手写 OCR 管道进行对比。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|------------------------|
| 延迟交互 | "ColPali 风格检索" | 查询标记独立对页面补丁评分；MaxSim 聚合 |
| 多向量 | "每个补丁的嵌入" | 每个文档有多个向量，而不是一个池化向量 |
| MaxSim | "延迟交互评分" | 对每个查询标记，取文档向量上的最大相似度；求和 |
| DocPruner | "补丁压缩" | 2026 年剪枝方法，保留 50% 的补丁，精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年测量视觉文档检索的标准 |
| 证据区域 | "被引用的边界框" | 源页上定位答案跨度的边界框 |
| OCR 回退 | "公式通道" | 与视觉并行使用的文本管道，用于公式或表格密集页面 |

## 进一步阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 参考的延迟交互文档检索
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪的检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 替代索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 替代托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退
