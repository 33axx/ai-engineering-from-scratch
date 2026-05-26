# Roots 与征询输入 —— 范围界定与工具调用中途的用户输入

> 硬编码路径一旦用户打开不同项目就会失效。预填的工具参数一旦用户输入不足就会失效。Roots 将服务器限定在用户控制的一组 URI 内；征询输入则能在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。两种客户端原语，分别修复 MCP 的两种常见失效模式。SEP-1036（URL 模式征询，2025-11-25）在 2026 年上半年仍处于实验阶段 —— 使用前请检查 SDK 版本。

**类型：** 构建  
**语言：** Python（stdlib，roots + 征询演示）  
**前置条件：** 阶段 13 · 07（MCP 服务器）  
**时长：** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在已声明的 root 集合内的 URI。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在表单模式与 URL 模式征询之间做出选择（后者为实验性；注意漂移风险）。

## 问题

一个笔记 MCP 服务器在生产环境遇到的两个具体失效。

**路径假设错误。** 服务器针对 `~/notes` 编写。用户在其他机器上，笔记位于 `~/Documents/Notes`，工具调用静默失败（找不到文件），更糟的情况是写到了错误位置。

**用户本应知道的参数缺失。** 用户要求“删除旧的 TPS 报告笔记”。模型调用了 `notes_delete(title: "TPS report")`，但存在三个匹配的笔记，分别来自 2023、2024 和 2025。工具无法猜测。返回“不明确”很烦人；对三个全部执行则是灾难性的。

Roots 解决第一个问题：客户端在 `initialize` 时声明服务器可以操作的 URI 集合。征询输入解决第二个问题：服务器暂停工具调用，发送 `elicitation/create` 询问用户选择哪一个。

## 概念

### Roots

客户端在 `initialize` 时声明一个 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：任何对 root 集合之外的文件的读写操作都应被拒绝。这不由客户端强制执行（服务器仍然是用户信任的代码），但符合规范的服务器会遵守。

当用户添加或移除 root 时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它们代表了用户的同意模型。用户对 Claude Desktop 说“授予这台笔记服务器访问这两个目录的权限”。服务器不能自行扩大该范围。

### 征询输入：表单模式（默认）

`elicitation/create` 接收一个表单模式（form schema）加上一个自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户回答，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户已填写）、`decline`（用户关闭）、`cancel`（用户中止了整个工具调用）。

表单模式是扁平的 —— v1 不支持嵌套对象。SDK 通常拒绝任何比单层更复杂的结构。

### 征询输入：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器发送一个 URL 而不是模式：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待完成，用户返回后返回结果。适用于表单不足以处理的场景，如 OAuth 流程、支付授权和文档签名。

漂移风险说明：SEP-1036 的响应形状仍在稳定中；一些 SDK 返回回调 URL，另一些返回完成令牌。在生产环境使用 URL 模式前，请阅读所用 SDK 的发布说明。

### 何时使用征询输入

- 破坏性操作前的用户确认（破坏性提示 + 征询）。
- 消歧（从 N 个匹配项中选一个）。
- 首次运行设置（API 密钥、目录、偏好设置）。
- OAuth 风格流程（URL 模式）。

### 何时不应使用征询输入

- 填充工具必需的参数，但模型本可用自然语言要求用户提供。应使用正常重新提示，而非征询对话框。
- 高频调用。征询会打断对话；不要在循环内触发。
- 服务器可以在事后验证的任何内容。验证，返回错误，让模型以文字方式询问用户。

### 人机交互桥接

征询输入与采样（sampling）共同构成 MCP 的“人机交互”模型。服务器的代理循环可以为用户输入（征询）或模型推理（采样）而暂停。阶段 13 · 11 介绍了采样；本课介绍征询。将两者结合即可实现完整的循环中控制。

## 使用

`code/main.py` 对笔记服务器进行了如下扩展：

- 响应 `roots/list`，并在收到 root 列表更改通知后重新查询。
- `notes_delete` 工具，使用 `elicitation/create` 在多个笔记匹配时进行消歧。
- `notes_setup` 工具，使用 URL 模式征询打开首次运行配置页面（模拟）。
- 边界检查，拒绝在已声明 roots 之外的 URI 上的操作。

演示运行三个场景：正常路径（一个匹配项）、消歧（三个匹配项，触发征询）、写入 root 之外（被拒绝）。

## 交付

本节课生成 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该技能将设计征询表单模式和消息模板。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户回答被路由回工具。
2. 新增一个工具 `notes_archive`，每次都需要征询确认（破坏性提示）。检查用户体验：与模型以文字方式重新询问相比如何？
3. 实现 URL 模式征询，用于首次运行的 OAuth 流程。注意漂移风险并添加 SDK 版本防护。
4. 扩展 `roots/list` 处理：收到通知时，服务器应以原子方式重新读取并重新扫描可能已超出范围的打开文件句柄。
5. 阅读 GitHub 上 SEP-1036 问题的讨论线程。找出一个影响服务器应如何处理 URL 模式回调的未决问题。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Root | “同意边界” | 客户端允许服务器操作的 URI |
| `roots/list` | “服务器询问范围” | 客户端返回当前的 root 集合 |
| `notifications/roots/list_changed` | “用户更改了范围” | 客户端通知 root 集合已改变 |
| Elicitation | “在调用中途询问用户” | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | “该方法” | 用于征询请求的 JSON-RPC 方法 |
| 表单模式 | “模式驱动的表单” | 扁平 JSON Schema，在客户端 UI 中渲染为表单 |
| URL 模式 | “浏览器重定向” | SEP-1036 试验性；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器需处理的三种分支 |
| Disambiguation | “选一个” | 当工具有 N 个候选时常见的征询用例 |
| 扁平表单 | “仅顶层属性” | 征询模式不允许嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) —— roots 规范权威参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) —— elicitation 规范权威参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) —— 2025-11-25 新增内容详解
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) —— URL 模式征询提案（实验性，有漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) —— UX 教程
