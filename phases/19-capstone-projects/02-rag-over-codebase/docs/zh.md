# 项目实战 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，每个严肃的工程组织都会运行一个理解含义（而不仅仅是字符串）的内部代码搜索。Sourcegraph Amp、Cursor 的代码库答案、Augment 的企业知识图谱、Aider 的仓库地图、Pinterest 的内部 MCP——套路相同。摄取多个仓库，用 tree-sitter 解析，嵌入函数和类级别的代码块，混合搜索，重排序，引用回答。本项目实战要求你构建一个能处理跨 10 个仓库的 200 万行代码，并能应对每次 git 推送时的增量重索引。

**类型：** 项目实战  
**语言：** Python（摄取），TypeScript（API + UI）  
**先决条件：** 阶段 5（NLP 基础）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 17（基础设施）  
**涉及的阶段：** P5 · P7 · P11 · P13 · P17  
**时长：** 30 小时

## 问题

到 2026 年，每个前沿编码智能体都配备了一个代码库检索层，因为仅凭上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文有所帮助，但不能替代排序检索。对原始代码块进行简单的余弦搜索会在生成的代码、单仓库重复以及很少使用的符号的长尾分布上毒化结果。生产级别的答案是经过混合（稠密 + BM25）搜索，构建在 AST 感知的代码块之上，再加一个重排序器，背后是一个符号引用图。

你要通过索引一个真实的代码库（而不是一个教程仓库）来学习，并测量 MRR@10、引用的忠实度以及增量新鲜度。失败模式是基础设施层面的：一个包含 10 万个文件的单仓库、一次改动一半文件的推送、一个需要跨越四个仓库才能正确回答的查询。

## 概念

一个感知 AST 的摄取管道使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处切块，而不是固定的 token 窗口。每个代码块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一个简短的自然语言摘要。该摘要添加了第三种可检索的模态——用户问“X 是如何被授权的”，摘要会提到“authz”，即使代码中只有 `check_permission`。

检索是混合的。一个查询同时触发稠密搜索和 BM25 搜索，合并 top-k 结果，将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表进入一个长上下文合成器（带提示缓存的 Claude Sonnet 4.7，或自托管的 Llama 3.3 70B），指示它对每个声明按文件和行范围进行引用。没有引用的答案会被后过滤器拒绝。

增量新鲜度是基础设施问题。Git 推送触发差异：哪些文件改变，哪些符号改变。只有受影响的代码块重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致性，而不需要每次提交都重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），后备为 bge-code-v1
- 稀疏索引：Tantivy（Rust），BM25 字段加权，符号名权重高于主体
- 向量数据库：Qdrant 1.12（混合搜索），或 pgvector + pgvectorscale（适用于 5000 万以下向量的团队）
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，带提示缓存
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows 用于摄取，LangGraph 用于查询智能体
- 合成器：Claude Sonnet 4.7（100 万上下文）带提示缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式），用于导入和方法调用边
- 可观测性：Langfuse 在每个检索和合成步骤添加监测

## 构建步骤

1. **摄取遍历器。** 在每个推送钩子上遍历 git 历史。收集变更的文件。对于每个文件，使用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 批量处理代码块，通过带提示缓存的 Haiku 4.5 调用（系统前缀）。提示：“用一句话概括此函数，说明其公共契约和副作用。”将摘要与代码块一同存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3，批量 128）和摘要（相同模型，但作用于摘要字符串）。将向量写入 Qdrant，附带载荷 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4、符号体权重 1、摘要权重 2。支持“查找名为 X 的函数”查询和“查找执行 X 的函数”查询。

5. **符号图。** 对每个代码块，记录边：导入（此文件使用仓库 Z 中的符号 Y）、调用（此函数调用类 C 的方法 M）、继承。存储在 kuzu 中。查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph 包含三个节点。`retrieve` 并行触发稠密搜索和 BM25，按 `(repo, path, symbol)` 去重。`rerank` 对 top-50 运行交叉编码器并保留 top-10。`synth` 调用 Claude Sonnet 4.7，将重排序后的代码块放入上下文，缓存系统提示，要求文件:行引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明被标记为重新询问或丢弃。只向用户返回带引用的答案。

8. **增量重索引。** 在每个 webhook 上，计算符号级别的差异。仅重新嵌入文本发生变化的代码块。对导入发生变化的代码块重新计算符号边。衡量标准：在一个 200 万行代码的代码库中，50 个文件的推送在 60 秒内完成重索引。

9. **评估。** 标记 100 个跨仓库问题，附带黄金答案的文件:行答案。测量 MRR@10、nDCG@10、引用忠实度（具有可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用它

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付它

可交付技能 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它搭建起摄取管道、混合索引和查询智能体，并针对任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 衡量方式 |
|:---:|---|---|
| 25 | 检索质量 | 在 100 个问题的保留集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 具有可验证文件:行锚点的答案声明比例 |
| 20 | 延迟与规模 | 在已索引语料库大小下，10000 QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git 推送到可搜索的时间（50 文件提交） |
| 15 | 用户体验与答案格式 | 引用的可点击性、片段预览、后续追问支持 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的差异。报告启用重排序后差距是否缩小。

2. 向语料库注入 20% 的生成代码（LLM 产生的样板代码）并重新评估。观察检索中毒。向载荷中添加“生成”标志并降低这些结果的权重。

3. 在您的语料库规模上，基准测试 Qdrant 混合搜索与 pgvector + pgvectorscale 的性能。报告批量大小为 1 时的 p99。

4. 添加基于抽样的漂移检查：每周重新运行 100 个问题的评估。如果 MRR@10 下降超过 5% 则发出警报。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用 Go 服务。使用符号图将它们链接起来。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| AST 感知切块 | “函数级拆分” | 在 tree-sitter 节点边界处切割代码，而不是固定 token 窗口 |
| 混合搜索 | “稠密 + 稀疏” | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | “第二阶段排序” | 对每个（查询，候选）配对一起打分的模型，比余弦更准确 |
| 提示缓存 | “缓存的系统提示” | 2026 年 Claude / OpenAI 功能，对重复前缀 token 提供高达 90% 的折扣 |
| 符号图 | “代码图” | 跨文件和仓库的导入、调用、继承边 |
| 引用忠实度 | “接地答案率” | 用户可通过点击锚点并阅读引用范围来验证的声明比例 |
| 增量重索引 | “从推送到可搜索的时间” | 从 git 推送到变更符号可查询的挂钟时间 |

## 进一步阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本项目实战的参考深度探讨
- [Aider 仓库地图](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
