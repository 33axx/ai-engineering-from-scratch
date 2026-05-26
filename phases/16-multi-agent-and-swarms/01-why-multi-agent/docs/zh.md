# 为什么选择多智能体？

> 单个智能体会碰壁。明智的做法不是造一个更大的智能体——而是造更多智能体。

**类型：** 学习  
**语言：** TypeScript  
**前置知识：** 第 14 阶段（智能体工程）  
**时间：** ~60 分钟  

## 学习目标

- 识别单智能体的天花板（上下文溢出、混合专长、顺序瓶颈），并解释何时拆分为多个智能体是正确的选择  
- 比较编排模式（流水线、并行扇出、监督者、层级结构），并根据给定任务结构选择合适的一种  
- 设计一个具有明确角色边界、共享状态和通信契约的多智能体系统  
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性之间的权衡  

## 问题

你在第 14 阶段构建了一个单智能体。它能工作。它能读取文件、运行命令、调用 API 并对结果进行推理。然后你把它指向一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及要求在编写代码之前研究外部 API。  

智能体卡住了。不是因为 LLM 笨，而是因为任务超出了单个智能体循环的处理能力。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用之前读过的内容。它试图同时扮演研究员、程序员和评审者，结果三者都做得很差。  

这就是单智能体的天花板。每当任务需要以下条件时，你都会遇到它：  

- **需要超过一个窗口容量的上下文** – 读取 50 个文件会超过 200k token  
- **不同阶段需要不同专长** – 研究需要与代码生成不同的提示词  
- **可以并行完成的工作** – 既然可以同时读取三个文件，为什么还要按顺序读？  

## 概念  

### 单智能体的天花板  

一个单智能体就是一个循环、一个上下文窗口、一个系统提示词。想象它：  

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```  

有三件事会出问题：  

1. **上下文饱和** – 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 150k token 的文件内容、命令输出和先前的推理。第 5 轮中的关键细节丢失了。  

2. **角色混淆** – 系统提示词说“你是研究员、程序员、评审者和测试员”，结果产生了一个半研究、半编码、从未完成评审的智能体。  

3. **顺序瓶颈** – 智能体读取文件 A，然后文件 B，然后文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行。  

### 多智能体解决方案  

拆分工作。每个智能体负责一项工作、一个上下文窗口和一个针对该工作调优的系统提示词：  

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```  

每个智能体拥有：  
- 一个专注的系统提示词（“你是代码评审者。你唯一的工作是发现 bug。”）  
- 自己独立的上下文窗口（不被其他智能体的工作污染）  
- 清晰的输入/输出契约（接收研究笔记，输出代码）  

### 实际采用此方案的系统  

**Claude Code 子智能体** – 当 Claude Code 使用 `Task` 生成子智能体时，它会创建一个具有限定任务的子智能体。父智能体保持自身上下文干净。子智能体完成聚焦工作并返回摘要。  

**Devin** – 运行一个规划智能体、一个编码智能体和一个浏览器智能体。规划智能体将工作分解为步骤。编码智能体编写代码。浏览器智能体研究文档。每个都有独立的上下文。  

**多智能体编码团队（SWE-bench）** – SWE-bench 上表现最好的系统使用一个研究员来读取代码库，一个规划器来设计修复方案，一个编码器来实现。单智能体系统得分较低。  

**ChatGPT Deep Research** – 并行生成多个搜索智能体，每个探索不同的角度，然后综合结果。  

### 频谱  

多智能体不是二元的。它是一个频谱：  

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```  

**单智能体** – 一个循环，一个提示词。适合简单任务。  

**子智能体** – 父智能体为聚焦的子任务生成子智能体。父智能体维护计划。子智能体报告结果。这就是 Claude Code 的做法。  

**流水线** – 智能体按序列运行。智能体 A 的输出成为智能体 B 的输入。适合阶段式工作流：研究 -> 编码 -> 评审 -> 测试。  

**团队** – 智能体并行运行，共享消息总线。每个都有一个角色。编排者进行协调。当同时需要不同技能时效果很好。  

**群体** – 许多相同或几乎相同的智能体，共享状态。没有固定的编排者。智能体从队列中取任务。适合高吞吐量的并行任务。  

### 四种多智能体模式  

#### 模式 1：流水线  

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```  

每个智能体转换数据并向前传递。易于推理。一个阶段的失败会阻塞后续所有阶段。  

#### 模式 2：扇出 / 扇入  

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```  

将工作拆分到并行智能体，然后合并结果。适合可以分解为独立子任务的任务。  

#### 模式 3：编排者 - 工作者  

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```  

一个智能的编排者决定做什么，委派给工作者，并综合结果。编排者本身就是一个拥有生成工作者工具的智能体。  

#### 模式 4：对等群体  

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```  

没有中央编排者。智能体点对点通信。决策从交互中涌现。更难调试，但可以扩展到大量智能体。  

### 何时不应使用多智能体  

多智能体增加了复杂性。智能体之间的每条消息都是一个潜在的失败点。调试从“读取一个对话”变成“在五个智能体之间追踪消息”。  

**保持单智能体的情况：**  
- 任务适合一个上下文窗口（工作数据低于 ~100k token）  
- 不同阶段不需要不同的系统提示词  
- 顺序执行足够快  
- 任务简单到拆分带来的开销大于价值  

**复杂性的代价：**  
- 每个智能体边界都是一个有损压缩步骤：智能体 A 的完整上下文被总结为一条发送给智能体 B 的消息  
- 协调逻辑（谁做什么、何时做、按什么顺序）本身就是一个 bug 源  
- 延迟增加：N 个智能体至少需要 N 次串行 LLM 调用，如果需要来回沟通则更多  
- 成本倍增：每个智能体独立消耗 token  

经验法则：如果一个任务需要的工具调用少于 20 次且适合 100k token，就保持单智能体。  

## 构建它  

### 第 1 步：过载的单智能体  

这是一个试图做所有事情的单智能体。它有一个庞大的系统提示词和一个同时包含研究、代码和评审的上下文窗口：  

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```  

这种方法的问题：  
- 上下文窗口随每个阶段增长。到评审步骤时，它已经包含了研究笔记、代码和先前的推理。  
- 系统提示词是通用的。不能为每个阶段调优。  
- 没有任何东西可以并行运行。  

### 第 2 步：专业智能体  

现在拆分它。每个智能体负责一项工作：  

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```  

每个专家都有一个专注的提示词。每个都获得一个干净的上下文窗口，只包含它需要的输入。  

### 第 3 步：通过消息进行协调  

用显式消息传递将专家智能体连接起来：  

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```  

每个智能体只接收发送给它的消息。没有上下文污染。研究员的 50k token 文档阅读内容永远不会进入评审者的上下文。  

### 第 4 步：比较  

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```  

多智能体版本使用更多总 token（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文保持干净。每个阶段的质量因为系统提示词专门化而提高。  

## 使用它  

本课产生一个可重用的提示词，用于决定何时采用多智能体。请参见 `outputs/prompt-multi-agent-decision.md`。  

## 练习  

1. 添加第四个专家：一个“测试者”智能体，它从编码者那里接收代码，从评审者那里接收评审反馈，然后编写测试。  
2. 修改流水线，使评审者可以将反馈发回给编码者进行修订循环（最多 2 轮）。  
3. 将顺序流水线转换为扇出：并行运行研究员和一个“需求分析器”智能体，然后在传递给编码者之前合并它们的输出。  

## 关键术语  

| 术语 | 人们通常说的 | 实际含义 |
|------|----------------|----------------------|
| 群体 | “AI 智能体的蜂群思维” | 一组对等智能体，共享状态，没有固定领导者。行为从局部交互中涌现。 |
| 编排者 | “老板智能体” | 一个智能体，其工具包括生成和管理其他智能体。它制定计划并委派任务，但可能不实际执行工作。 |
| 协调者 | “交通协管员” | 一个非智能体组件（通常是代码，而不是 LLM），根据规则在智能体之间路由消息。 |
| 共识 | “智能体们达成一致” | 一种协议，要求多个智能体必须达成一致才能继续。用于需要解决冲突输出的场景。 |
| 涌现行为 | “智能体们自己搞定了” | 系统层面从智能体交互中产生的模式，并非显式编程得到。可能有益也可能有害。 |
| 扇出 / 扇入 | “智能体的 Map-Reduce” | 将任务分配到并行的智能体（扇出），然后合并它们的结果（扇入）。 |
| 消息传递 | “智能体之间互相交谈” | 智能体之间的通信机制：从一个智能体发送到另一个智能体的结构化数据，替代共享的上下文窗口。 |

## 延伸阅读  

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) – 多智能体模式综述  
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) – 微软的多智能体对话框架  
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) – Claude Code 如何使用 Task 进行委派  
- [CrewAI documentation](https://docs.crewai.com/) – 基于角色的多智能体框架
