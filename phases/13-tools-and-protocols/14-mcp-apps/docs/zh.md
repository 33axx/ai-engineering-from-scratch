# MCP Apps — 通过 `ui://` 提供交互式 UI 资源

> 纯文本的工具输出限制了智能体所能展示的内容。MCP Apps（SEP-1724，2026年1月26日正式发布）允许工具返回沙箱化的交互式 HTML，这些 HTML 会内联渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。仪表盘、表单、地图、3D 场景，全部通过这一个扩展实现。本课将讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、ifram-sandbox 的 postMessage 协议，以及让服务器渲染 HTML 所伴随的安全攻防面。

**类型：** 实操
**语言：** Python（标准库，UI 资源发射器）、HTML（示例应用）
**前置知识：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源）
**时长：** ~75 分钟

## 学习目标

- 从工具调用返回一个 `ui://` 资源，并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具对应的 UI。
- 实现用于 UI 与宿主通信的 iframe sandbox postMessage JSON-RPC。
- 应用能够防御源自 UI 的攻击的 CSP 和 permissions-policy 默认值。

## 问题

一个2025年式的 `visualize_timeline` 工具可以返回“以下是按时间顺序排列的14条笔记：……”。这只是一个段落。用户真正想要的是交互式时间线。在 MCP Apps 之前，可选的方案要么是客户端特定的 Widget API（Claude artifacts、OpenAI Custom GPT HTML），要么根本没有 UI。

MCP Apps（SEP-1724，2026年1月26日发布）标准化了约定。工具结果包含一个 `resource`，其 URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。宿主会在一个带有有限 CSP 且默认无网络访问权限（除非明确授权）的沙箱化 iframe 中渲染它。iframe 内部的 UI 通过一个微型的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以相同的方式渲染同一个 `ui://` 资源。一个服务器，一份 HTML 束，通用的 UI。

## 概念

### `ui://` 资源方案

一个工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后宿主会调用 `resources/read` 读取 `ui://notes/timeline` URI，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主持有 HTML 并将其渲染在一个沙箱化的 `<iframe>` 中，该 iframe 带有：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格的设置）
- 通过响应头应用服务器声明的 CSP。
- 没有来自宿主源的 cookies 或 localStorage。
- 网络访问限制在 CSP 的 `connectSrc` 中。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。采用微型的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对端的精确源，并且在接收端对 `event.origin` 进行白名单验证后再处理任何负载。永远不要在信道的任何一侧使用 `"*"` —— 消息体携带工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可以调用的宿主端方法：

- `host.callTool(name, arguments)` —— 调用服务器工具。
- `host.readResource(uri)` —— 读取 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示模板。
- `host.close()` —— 关闭 UI。

每次调用仍然通过 MCP 协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外的能力：

- `camera` —— 访问用户的摄像头（用于扫描文档等 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 位置信息。
- `network:*` —— 比 `connectSrc` 单独允许的更宽松的网络访问。

每个权限在 UI 渲染之前都会作为提示显示给用户。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 进行提示注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本，从而欺骗用户。宿主渲染时应明显区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 进行数据泄露。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任何地方。默认应严格限制。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操纵并强制执行透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 将作为 MCP 安全的一部分深入介绍这些内容；本课仅作引入。

### `ui/initialize` 握手

iframe 加载完成后，它会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应能力列表和一个会话令牌（session token）。UI 在后续每次宿主调用中都使用该会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露了两个便捷原语：

- `AppRenderer`（服务端）—— 包装一个 React / Vue / Solid 组件，并发射出一个带有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源，挂载 iframe，并中介 postMessage。

你可以使用这些原语，也可以手工编写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起全面支持。
- **ChatGPT。** 通过 Apps SDK（底层使用相同的 MCP Apps 协议）全面支持。
- **Cursor。** Beta 版；需在设置中启用。
- **VS Code。** 仅限 Insider 构建版本。
- **Goose。** 全面支持。
- **Zed、Windsurf。** 已列入路线图。

生产环境中的服务器：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 使用它

`code/main.py` 扩展了笔记服务器，增加了一个 `visualize_timeline` 工具，该工具返回一个 `ui://notes/timeline` 资源，并增加了一个针对该 URI 的 `resources/read` 处理器，处理器返回一个微小但完整的带有 SVG 时间线的 HTML 包。HTML 是通过标准库模板生成的——无需构建系统。postMessage 在 JS 注释中以草图形式呈现，因为标准库无法驱动浏览器。

值得关注的点：

- 工具响应上的 `_meta.ui` 携带了 resourceUri、CSP、权限。
- HTML 渲染时没有网络访问；所有数据都是内联的。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此 stdlib 演示中已文档化但未实际运行）。

## 交付

本课生成 `outputs/skill-mcp-apps-spec.md`。针对一个可从交互式 UI 获益的工具，该技能将产出完整的 MCP Apps 约定：`ui://` URI、CSP、权限、postMessage 入口点以及安全清单。

## 练习

1. 运行 `code/main.py` 并检查生成的 HTML。直接在浏览器中打开该 HTML，验证 SVG 是否正确渲染。然后画出 UI 用来调用 `host.callTool("notes_update", ...)` 的 postMessage 约定草图。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中需要做哪些更改？

3. 添加第二个 UI 资源 `ui://notes/editor`，带有一个用于原地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪里注入内容？iframe 沙箱防御了什么，又没有防御什么？

5. 阅读 SEP-1724 规范，找出该玩具实现未使用的 MCP Apps SDK 中的一项能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP Apps | "交互式 UI 资源" | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | "App URI 方案" | UI 束的资源方案 |
| `text/html;profile=mcp-app` | "MIME" | MCP App HTML 的内容类型 |
| Iframe sandbox | "渲染容器" | 通过 CSP 和权限对 UI 进行浏览器沙箱化 |
| postMessage JSON-RPC | "UI 到宿主的连线" | 用于宿主调用的迷你 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务端 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 挂载 iframe 并中介 postMessage 的助手 |
| `ui/initialize` | "握手" | UI 发送给宿主的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) —— 参考实现和 SDK
- [MCP Apps 规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) —— 正式规范文档
- [MCP — Apps 扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) —— 高级文档
- [MCP 博客 — MCP Apps 发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) —— 2026 年 1 月发布博文
- [MCP Apps API 参考](https://apps.extensions.modelcontextprotocol.io/api/) —— JSDoc 风格的 SDK 参考
