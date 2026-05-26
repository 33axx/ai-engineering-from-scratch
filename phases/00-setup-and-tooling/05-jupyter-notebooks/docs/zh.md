# Jupyter Notebooks

> Notebook 是 AI 工程的工作台。你在原型中探索，然后把可用的部分迁移到生产环境。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 0，第 01 课  
**时长：** 约 30 分钟

## 学习目标

- 安装并启动 JupyterLab、Jupyter Notebook 或带有 Jupyter 扩展的 VS Code
- 使用魔术命令（`%timeit`、`%%time`、`%matplotlib inline`）进行基准测试和内联可视化
- 区分何时使用 notebook 与何时使用脚本，并应用“在 notebook 中探索，在脚本中交付”的工作流
- 识别并避免常见的 notebook 陷阱：乱序执行、隐藏状态和内存泄漏

## 问题所在

每篇 AI 论文、教程和 Kaggle 竞赛都在使用 Jupyter notebook。它们允许你分片运行代码，内联查看输出，将代码与解释混在一起，并快速迭代。如果你试图在不使用 notebook 的情况下学习 AI，那就相当于做数学作业没有草稿纸。

但是 notebook 也有真正的陷阱。人们用它们来做任何事情，包括那些它们根本不擅长的事情。知道何时使用 notebook 以及何时使用脚本，会免去你日后调试的噩梦。

## 概念

notebook 是一个由单元格组成的列表。每个单元格要么是代码，要么是文本。

```mermaid
graph TD
    A["**Markdown Cell**\n# My Experiment\nTesting learning rate 0.01"] --> B["**Code Cell** ► Run\nmodel.fit(X, y, lr=0.01)\n---\nOutput: loss = 0.342"]
    B --> C["**Code Cell** ► Run\nplt.plot(losses)\n---\nOutput: inline plot"]
```

内核是一个在后台运行的 Python 进程。当你运行一个单元格时，它会将代码发送给内核，内核执行代码并返回结果。所有单元格共享同一个内核，因此变量在单元格之间持久存在。

```mermaid
graph LR
    A[Notebook UI] <--> B[Kernel\nPython process]
    B --> C[Keeps variables in memory]
    B --> D[Runs cells in whatever order you click]
    B --> E[Dies when you restart it]
```

那个“你点击的任何顺序”的部分既是超能力，也是伤脚的利器。

## 动手构建

### 第 1 步：选择你的界面

三种选项，一种格式：

| 界面 | 安装方式 | 最适合 |
|-----------|---------|----------|
| JupyterLab | `pip install jupyterlab` 然后 `jupyter lab` | 完整的 IDE 体验，多标签页，文件浏览器，终端 |
| Jupyter Notebook | `pip install notebook` 然后 `jupyter notebook` | 简单轻量，一次处理一个 notebook |
| VS Code | 安装 "Jupyter" 扩展 | 已经在你编辑器中，Git 集成，调试 |

三者读取和写入相同的 `.ipynb` 文件。选择你喜欢的任意一个。在 AI 工作中 JupyterLab 最为常见。

```bash
pip install jupyterlab
jupyter lab
```

### 第 2 步：重要的键盘快捷键

你在两种模式下操作。按 `Escape` 进入命令模式（左侧蓝色边框），按 `Enter` 进入编辑模式（绿色边框）。

**命令模式（最常用）：**

| 按键 | 操作 |
|-----|--------|
| `Shift+Enter` | 运行单元格，移到下一个 |
| `A` | 在上方插入单元格 |
| `B` | 在下方插入单元格 |
| `DD` | 删除单元格 |
| `M` | 转换为 markdown |
| `Y` | 转换为代码 |
| `Z` | 撤销单元格操作 |
| `Ctrl+Shift+H` | 显示所有快捷键 |

**编辑模式：**

| 按键 | 操作 |
|-----|--------|
| `Tab` | 自动补全 |
| `Shift+Tab` | 显示函数签名 |
| `Ctrl+/` | 切换注释 |

`Shift+Enter` 是你每天会使用上千次的快捷键。先学会它。

### 第 3 步：单元格类型

**代码单元格** 运行 Python 并显示输出：

```python
import numpy as np
data = np.random.randn(1000)
data.mean(), data.std()
```

输出：`(0.0032, 0.9987)`

**Markdown 单元格** 渲染格式化文本。用它们来记录你正在做什么以及为什么。支持标题、粗体、斜体、LaTeX 数学公式（`$E = mc^2$`）、表格和图片。

### 第 4 步：魔术命令

这些不是 Python。它们是 Jupyter 特有的命令，以 `%`（行魔术）或 `%%`（单元格魔术）开头。

**计时你的代码：**

```python
%timeit np.random.randn(10000)
```

输出：`45.2 us +/- 1.3 us per loop`

```python
%%time
model.fit(X_train, y_train, epochs=10)
```

输出：`Wall time: 2.34 s`

`%timeit` 多次运行代码并取平均值。`%%time` 只运行一次。对于微基准测试使用 `%timeit`，对于训练运行使用 `%%time`。

**启用内联绘图：**

```python
%matplotlib inline
```

每个 `plt.plot()` 或 `plt.show()` 都会直接渲染在 notebook 中。

**在不离开 notebook 的情况下安装包：**

```python
!pip install scikit-learn
```

`!` 前缀可以运行任何 shell 命令。

**检查环境变量：**

```python
%env CUDA_VISIBLE_DEVICES
```

### 第 5 步：内联显示富文本输出

Notebook 会自动显示单元格中的最后一个表达式。但你也可以控制它：

```python
import pandas as pd

df = pd.DataFrame({
    "model": ["Linear", "Random Forest", "Neural Net"],
    "accuracy": [0.72, 0.89, 0.94],
    "training_time": [0.1, 2.3, 45.6]
})
df
```

这会渲染一个格式化的 HTML 表格，而不是文本转储。绘图也同样如此：

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Inline Plot")
plt.show()
```

绘图会直接出现在单元格下方。这就是 notebook 在 AI 工作中占主导地位的原因。你看数据、看图表、看代码，三者在一起。

对于图片：

```python
from IPython.display import Image, display
display(Image(filename="architecture.png"))
```

### 第 6 步：Google Colab

Colab 是云端的免费 Jupyter notebook。它提供 GPU、预装库和 Google Drive 集成。无需任何设置。

1. 前往 [colab.research.google.com](https://colab.research.google.com)
2. 上传本课程中的任意 `.ipynb` 文件
3. 选择 运行时 > 更改运行时类型 > T4 GPU（免费）

Colab 与本地 Jupyter 的区别：
- 文件在会话之间不会持久保存（保存到 Drive 或下载）
- 预装了：numpy、pandas、matplotlib、torch、tensorflow、sklearn
- 使用 `from google.colab import files` 上传/下载文件
- 使用 `from google.colab import drive; drive.mount('/content/drive')` 实现持久存储
- 空闲 90 分钟后会话超时（免费版）

## 使用它

### Notebook 与脚本：何时使用哪个

| 使用 notebook 的场景 | 使用脚本的场景 |
|-------------------|-----------------|
| 探索数据集 | 训练管道 |
| 原型制作模型 | 可复用的工具函数 |
| 可视化结果 | 任何包含 `if __name__` 的代码 |
| 解释你的工作 | 定时运行的代码 |
| 快速实验 | 生产代码 |
| 课程练习 | 包和库 |

规则：**在 notebook 中探索，在脚本中交付**。

AI 中的常见工作流：
1. 在 notebook 中探索数据
2. 在 notebook 中原型化模型
3. 一旦可行，将代码移到 `.py` 文件中
4. 将这些 `.py` 文件导入回 notebook 以进行进一步实验

### 常见陷阱

**乱序执行。** 你运行了单元格 5，然后单元格 2，然后单元格 7。Notebook 在你机器上工作正常，但当其他人从顶部到底部运行时却出错了。解决方法：在分享之前，使用 内核 > 重启并全部运行。

**隐藏状态。** 你删除了一个单元格，但它创建的变量仍然在内存中。Notebook 看起来整洁，但依赖一个幽灵单元格。解决方法：定期重启内核。

**内存泄漏。** 加载一个 4GB 数据集，训练一个模型，再加载另一个数据集。没有任何东西被释放。解决方法：使用 `del variable_name` 和 `gc.collect()`，或者重启内核。

## 交付

本节课产出：
- `outputs/prompt-notebook-helper.md` 用于调试 notebook 问题

## 练习

1. 打开 JupyterLab，创建一个 notebook，使用 `%timeit` 比较列表推导式与 numpy 在创建包含 100,000 个随机数的数组时的性能
2. 创建一个同时包含 markdown 和代码单元格的 notebook，加载一个 CSV 文件，显示一个数据框，并绘制一张图表。然后运行 内核 > 重启并全部运行，验证从头到尾能正常工作
3. 从 `code/notebook_tips.py` 中取出代码，粘贴到一个 Colab notebook 中，并使用免费 GPU 运行它

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| 内核 | “运行我代码的东西” | 一个独立的 Python 进程，执行单元格并将变量保留在内存中 |
| 单元格 | “一个代码块” | notebook 中一个可独立运行的单元，可以是代码或 markdown |
| 魔术命令 | “Jupyter 小技巧” | 以 `%` 或 `%%` 为前缀的特殊命令，用于控制 notebook 环境 |
| `.ipynb` | “Notebook 文件” | 一个 JSON 文件，包含单元格、输出和元数据。代表 IPython Notebook |

## 进一步阅读

- [JupyterLab 文档](https://jupyterlab.readthedocs.io/) 了解完整功能集
- [Google Colab 常见问题](https://research.google.com/colaboratory/faq.html) 了解 Colab 特定的限制和功能
- [28 个 Jupyter Notebook 技巧](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) 了解高级用户快捷键
