# Capstone 06 — Kubernetes 的 DevOps 故障排查 Agent

> AWS 的 DevOps Agent 已正式上线，Resolve AI 发布了其 K8s playbook，NeuBird 演示了语义监控，Metoro 将 AI SRE 与按服务划分的 SLO 关联起来。生产形态已经确定：告警 webhook 触发，Agent 读取遥测数据，遍历 K8s 对象的图谱，对根因假设进行排序，并在 Slack 上发布带有批准按钮的简报。默认只读。所有补救措施均需人工批准。本 Capstone 即为该 Agent，将在 20 个合成事件上进行评估，并与 AWS Agent 在三个共享案例上进行对比。

**类型：** Capstone
**语言：** Python（Agent），TypeScript（Slack 集成）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具与 MCP）、阶段 14（Agent）、阶段 15（自主系统）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**预计时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变为：「AI Agent 对事件进行分类，人类批准补救措施。」AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 均已将此形态投入生产。Agent 读取 Prometheus 指标、Loki 日志、Tempo 链路、kube-state-metrics 以及 K8s 对象的知识图谱。它在五分钟内生成带有遥测引用证据的排序后根因假设。在没有通过 Slack 获得明确人工批准的情况下，它永远不会执行破坏性命令。

大部分困难工作在于范围界定和安全，而非推理。Agent 需要一个默认只读的 RBAC 表面、一个加固的 MCP 工具服务器，以及针对每个考虑执行与实际执行命令的审计日志。它需要知道何时超出自身能力范围并上报。此外，它必须运行得足够廉价，以免 OOM-kill 级联导致产生 5000 美元的 Agent 账单。

## 概念

Agent 在知识图谱上运行。节点为 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测源（Prometheus 序列、Loki 流、Tempo 链路）。边编码了所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）以及观测关系（Pod -> Prometheus 序列）。该图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，Agent 从受影响的对象开始根因分析。它遍历边，拉取相关遥测切片（最近 15 分钟），并起草假设。假设按证据排序：支持它的遥测引用数量、引用时效性、引用特异性。前三名假设连同图路径可视化以及用于补救操作的批准按钮一起发送到 Slack。

补救措施受门控限制。允许的默认操作为只读。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 批准；ArgoCD 回滚钩子需要一个 Agent 从未持有的认证令牌。审计日志记录 Agent 曾经*考虑过*的每条命令——而不仅仅是执行的命令——以便审查过程能捕获近失误。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），包含 K8s 对象 + 遥测边
- Agent：LangGraph，带有每个工具的允许列表，默认只读
- 工具传输：FastMCP 基于 StreamableHTTP；破坏性工具位于单独的服务器中，位于批准门控之后
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 补救：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 批准卡片
- 审计：仅追加的结构化日志（已考虑、已执行、已批准、结果）
- 部署：K8s 部署，拥有自身狭窄的 RBAC 角色；独立的命名空间

## 构建

1. **图谱摄入**。每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（Pod 被 Prometheus 序列观测）。

2. **告警接收器**。FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违规。

3. **只读工具表面**。通过 FastMCP 包装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有狭窄的 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有 "delete"、"exec"、"scale"。

4. **根因 Agent**。LangGraph 包含三个节点：`sample` 拉取过去 15 分钟的遥测切片，`walk` 查询图谱以获取相邻对象，`hypothesize` 起草带有遥测引用的排序后根因候选。

5. **证据评分**。每个假设的分数 = 时效性 * 特异性 * 图谱路径长度倒数 * 引用数量。返回前三名。

6. **Slack 简报**。发布一个附件，包含假设、图谱路径可视化（服务端渲染的子图图像）以及最多一个补救操作的批准按钮。

7. **补救门控**。破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，位于批准令牌之后。Agent 只有在 Slack 卡片被人工批准后才能调用它们。

8. **审计日志**。仅追加的 JSONL：对于每个候选命令，记录是否被考虑、是否被执行、由谁批准。每天传输到 S3。

9. **合成事件套件**。构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 填满、嘈杂邻居、故障 Sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取回退等。根据根因准确性和假设生成时间对 Agent 评分。

## 使用

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付

`outputs/skill-devops-agent.md` 是可交付物。给定一个 K8s 集群和告警源，Agent 生成排序后的根因假设以及一个受 Slack 门控的补救流程。

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | 场景套件 RCA 准确性 | 在 20 个合成事件中，≥80% 正确根因 |
| 20 | 安全性 | 审计日志中，破坏性操作守卫从未在无 Slack 批准的情况下触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的相同三个事件上运行你的 Agent。发布并排对比结果。报告你的 Agent 在哪些地方存在分歧。

2. 添加一个「近失误」审计，标记 Agent *曾经考虑过*但在未批准情况下本会具有破坏性的所有命令。在一周内测量近失误率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确率的变化以及每事件的美元成本。

4. 构建一个因果过滤器：区分相关的遥测峰值与真正的根因。在 20 个场景标签上训练一个小型分类器。

5. 添加回滚试运行：对具有相同清单的预演集群执行 ArgoCD 回滚。在 Slack 批准按钮之前，在生产集群中验证回滚计划。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| K8s 知识图谱 | "集群图谱" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| 默认只读 | "作用域 RBAC" | Agent 的服务账户仅拥有 get/list/describe 动词；破坏性动词位于另一个服务器中，需要批准 |
| 审计日志 | "考虑与执行" | 每个候选命令的仅追加记录，包括是否执行、由谁批准 |
| 假设排序 | "证据分数" | 时效性 × 特异性 × 图谱路径长度倒数 × 引用数量 |
| Slack 批准卡片 | "人在环门控" | 带有补救按钮的交互式 Slack 消息；Agent 必须在有人点击后才能继续 |
| 遥测引用 | "证据指针" | 支持某个论断的 Prometheus 查询、Loki 选择器或 Tempo 链路 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的挂钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年标准参考文献
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考文献
- [NeuBird semantic monitoring](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — 以 SLO 为先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考 Agent 编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控补救目标
