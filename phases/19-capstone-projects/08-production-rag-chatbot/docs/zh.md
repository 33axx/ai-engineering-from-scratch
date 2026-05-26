# 顶点项目 08 — 受监管垂直领域的生产级 RAG 聊天机器人

> 2026 年，Harvey、Glean、Mendable 和 LlamaCloud 都运行着相同的生产架构。使用 docling 或 Unstructured 以及 ColPali 处理视觉内容进行摄取。混合搜索。使用 bge-reranker-v2-gemma 进行重排序。使用 Claude Sonnet 4.7 进行综合，并利用提示缓存实现 60-80% 的命中率。使用 Llama Guard 4 和 NeMo Guardrails 进行防护。使用 Langfuse 和 Phoenix 进行监控。使用 RAGAS 在 200 道题的黄金集上进行评分。在受监管领域（法律、临床、保险）构建一个这样的系统，顶点项目需要通过在黄金集、红队测试和漂移仪表板上的评估。

**类型：** 顶点项目
**语言：** Python（管道 + API）、TypeScript（聊天 UI）
**前置要求：** 阶段 5（NLP）、阶段 7（transformers）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**用时：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年部署最多的生产架构，因为投资回报率显而易见，且风险具体。Harvey（Allen & Overy）为法律领域构建了它。Mendable 提供开发者文档版本。Glean 覆盖企业搜索。模式是：高保真摄取、混合检索加重排序、附带引用强制和提示缓存的综合、多层安全防护、以及持续漂移监控。

难点不在于模型。而在于司法管辖区感知的合规性（HIPAA、GDPR、SOC2）、引用级别的可审计性、成本控制（当命中率高时，提示缓存可节省 60-90% 费用）、通过 RAGAS 忠实度进行的幻觉检测，以及当源文档更新而索引未同步时的漂移检测。本顶点项目要求你在一套包含 200 道题的黄金集和一个红队测试套件上完成所有这些功能。

## 概念

管道有两个端。**摄取**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富文档；生成的块附带摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（5000 万以下向量）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮对话；每次查询执行混合检索，使用 bge-reranker-v2-gemma-2b 重排序，使用 Claude Sonnet 4.7（提示缓存）综合，输出通过 Llama Guard 4 和 NeMo Guardrails，并生成带引用锚点的响应。

评估栈分为四层。**黄金集**（200 个带引用标注的问答对）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全性。**RAGAS** 用于每轮自动评估忠实度/答案相关性/上下文精确度。**漂移仪表板**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示 + 检索到的上下文。在 60-80% 的命中率下，每次查询成本下降 3-5 倍。管道必须设计为具有稳定前缀（首先是系统提示 + 重排序后的上下文）以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄取：结构化文档使用 Unstructured.io 或 docling；视觉丰富 PDF 使用 ColPali
- 向量数据库：5000 万以下向量使用 pgvector + pgvectorscale；否则使用 Qdrant Cloud
- 稀疏：Tantivy BM25 带字段权重
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM：Claude Sonnet 4.7 带提示缓存；后备使用自托管 Llama 3.3 70B
- 评估：RAGAS 0.2 在线评估，DeepEval 用于幻觉和越狱套件
- 可观测性：Langfuse 自托管带标注队列；Arize Phoenix 用于漂移
- 防护：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 擦除
- 合规性：块上的基于角色的访问标签；用于 GDPR/HIPAA 的司法管辖区标签

## 构建步骤

1. **摄取。** 使用 Unstructured 或 docling 解析你的语料库（认真构建需要 1000-10000 个文档）。对于扫描件/视觉密集页面，转给 ColPali。生成带有摘要、角色标签、司法管辖区标签的块。

2. **索引。** 将稠密嵌入（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 构建 BM25 侧索引。角色和司法管辖区过滤作为载荷。

3. **混合检索。** 首先按角色+司法管辖区过滤；然后并行进行稠密 + BM25 搜索；使用倒数排名融合合并；取前 20 给重排序器；取前 5 给综合。

4. **使用提示缓存进行综合。** 系统提示 + 静态策略放入缓存头；重排序后的上下文作为缓存扩展；用户问题作为未缓存后缀。目标在稳态下达到 60-80% 缓存命中率。

5. **防护。** Llama Guard 4 用于输入；NeMo Guardrails 规则阻止领域外问题或策略禁止话题；Presidio 擦除输出中的意外 PII；引用强制执行后置过滤。

6. **黄金集。** 200 个由领域专家标注的问答对，含（答案、引用）。根据精确引用匹配、答案正确性、忠实度（RAGAS）对智能体评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 提取尝试、领域外问题、跨司法管辖区泄露。通过通过/失败和严重性评分。

8. **漂移仪表板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。在 5% 下降时发出警报。

9. **成本报告。** Langfuse：提示缓存命中率、每次查询令牌数、按阶段划分的 $/查询。

## 使用方法

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付标准

`outputs/skill-production-rag.md` 描述交付物。一个已部署的受监管领域聊天机器人，带有合规标签，通过了评分标准，并配有实时漂移监控。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金集（200 个问答）上的在线分数 |
| 20 | 引用正确性 | 答案中可验证源锚点的比例 |
| 20 | 防护覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、P95 延迟、$/查询 |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板，显示每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同司法管辖区下构建第二个语料库切片（例如 HIPAA 与 GDPR 并存）。通过一个包含 20 个问题的跨司法管辖区探测，展示角色+司法管辖区过滤能够防止交叉泄露。

2. 在生产流量下测量一周的提示缓存命中率。识别哪些查询会破坏缓存前缀。进行重构。

3. 添加带 10k 令牌摘要缓冲区的多轮记忆。测量随着对话增长，忠实度是否下降。

4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $/查询和忠实度的差异。

5. 添加“不确定”模式：如果最高重排序分数低于某个阈值，智能体说“我没有自信的引用”而非回答。测量错误自信的减少情况。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| 提示缓存 | “缓存系统提示+上下文” | Claude/OpenAI 功能：命中的缓存前缀令牌折扣 60-90% |
| RAGAS | “RAG 评估器” | 自动评分忠实度、答案相关性、上下文精确度 |
| 黄金集 | “标注评估集” | 200+ 专家标注的问答对及引用；真实基准 |
| 司法管辖区标签 | “合规标签” | 附加到块上的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制实施 |
| 引用忠实度 | “有根据的答案率” | 由可检索源片段支持的声明的比例 |
| 漂移 | “检索质量衰减” | nDCG 或引用分数的每周变化；警报阈值 5% |
| 红队 | “对抗性评估” | 发布前的越狱、PII 提取、领域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 参考
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄取
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略规则框架
