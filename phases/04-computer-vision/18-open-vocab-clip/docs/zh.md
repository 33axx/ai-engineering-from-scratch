# 开放词汇视觉 — CLIP

> 将图像编码器和文本编码器一起训练，使得匹配的（图像，标题）对在共享空间中的同一位置落地。这就是全部诀窍。

**类型：** 构建 + 使用  
**语言：** Python  
**前置知识：** 第四阶段第14课（ViT），第四阶段第17课（自监督）  
**时间：** 约45分钟

## 学习目标

- 解释CLIP的双塔架构和对比训练目标
- 使用预训练的CLIP（或SigLIP）进行零样本分类，无需任何任务特定训练
- 从头实现零样本分类：编码类别提示，计算余弦相似度，取argmax
- 区分CLIP、SigLIP、OpenCLIP和LLaVA/LLaMA-vision模型——它们在2026年的各自用途

## 问题

传统分类器是封闭词汇的：一个1000类的ImageNet模型只能预测1000个标签。每增加一个新类别都需要标注数据和重新训练分类头。

CLIP（Radford等人，OpenAI 2021）展示了在从网络抓取的4亿（图像，标题）对上训练，可以产生一个在推理时能对任意类别集进行分类的模型，这些类别完全用自然语言描述。你通过写一个句子来给出一个新类别。

这种能力——零样本迁移——正是为什么每个现代视觉系统都从一个CLIP系列检查点开始。检测（Grounding DINO, OWL-ViT）、分割（CLIPSeg, SAM）、检索、内容审核、VLM和文本到图像生成都建立在CLIP风格的联合嵌入之上。

## 概念

### 双塔

```mermaid
flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```

两个编码器都以线性投影结束，投影到相同的嵌入维度（CLIP-B/32为512维，CLIP-L/14为1024维）。L2归一化后计算余弦相似度。

### 目标函数

给定一个批次的N个（图像，标题）对，构建一个NxN相似度矩阵。训练两个编码器，使得对角线（匹配对）具有高相似度，非对角线（不匹配对）具有低相似度。

```
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```

对称是因为图像到文本和文本到图像的检索都应该有效。`tau`（温度）通常作为学习参数初始化，初始值为0.07。

### SigLIP：更好的损失函数

SigLIP（Zhai等人，2023）用逐对sigmoid取代了softmax：

```
loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise
```

逐对损失去除了CLIP所需的批次级归一化。SigLIP在小批量大小下训练得更好，并且在同等数据量下达到或超过CLIP。

### 零样本分类

给定一个训练好的CLIP：

1. 对于每个类别，编写一个提示："a photo of a {class}"。
2. 用文本编码器编码所有类别的提示 -> `T` 形状 (C, d)。
3. 编码测试图像 -> `I` 形状 (1, d)。
4. 相似度 = `I @ T.T` 形状 (1, C)。
5. Argmax -> 预测类别。

提示工程很重要。OpenAI发布了80个ImageNet的提示模板（"a photo of a {}", "a blurry photo of a {}", "a sketch of a {}", ...）。对每个类别所有模板的嵌入取平均，可以获得额外1-3%的top-1准确率。

### 2026年CLIP风格模型的应用场景

- **零样本分类** — 直接使用。
- **图像检索** — 一次编码所有图像，推理时嵌入查询。
- **文本条件检测** — Grounding DINO、OWL-ViT将一个CLIP文本塔包裹在检测器周围。
- **文本条件分割** — CLIPSeg；SAM通过CLIP使用文本提示输入。
- **VLM** — LLaVA、Qwen-VL、InternVL将CLIP系列视觉编码器接入LLM。
- **文本到图像生成** — Stable Diffusion、DALL-E 3以CLIP文本嵌入为条件。

一旦你拥有了共享嵌入空间，每个视觉+语言任务都变成了距离计算。

## 构建它

### 步骤1：微型双塔模型

真正的CLIP是ViT +  transformer。本课中，塔是小MLP，在预提取特征上运行，以便在CPU上看到训练信号。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```

两个投影，共享维度的输出，学习温度。与真实CLIP API形状相同。

### 步骤2：对比损失

```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```

对称。logit_scale越高 = softmax越尖锐 = 越自信但存在不稳定风险。

### 步骤3：零样本分类器

```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```

每一步一行代码。这正是使用生产级CLIP检查点时的零样本流程。

### 步骤4：合理性检查

```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```

随机初始化的模型损失应接近 `log(N) = log(8) = 2.08` —— 当尚未学到任何结构时的对称交叉熵目标。

## 使用它

OpenCLIP是2026年的社区默认选择：

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```

SigLIP更新，在小规模下训练效果更好，新工作更倾向于使用它：`google/siglip-base-patch16-224`。Hugging Face同时提供两者。

## 交付它

本课产出：

- `outputs/prompt-zero-shot-class-picker.md` — 一个提示，用于为给定类别列表和领域的零样本CLIP设计类别模板。
- `outputs/skill-image-text-retriever.md` — 一个技能，使用任意CLIP检查点构建图像嵌入索引，支持按文本查询和按图像查询。

## 练习

1. **(简单)** 使用预训练的OpenCLIP ViT-B/32，在CIFAR-10上使用80模板提示集进行零样本分类。报告top-1准确率，应约为85-90%。
2. **(中等)** 在相同CIFAR-10任务上，比较单模板（"a photo of a {}"）与80模板平均嵌入。量化差距并解释为什么模板有帮助。
3. **(困难)** 构建一个零样本图像检索索引：用CLIP嵌入1000张图像，构建FAISS索引，用自然语言描述进行查询。为你手写的20个保留查询报告检索recall@5。

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|------|-----------|---------|
| 双塔 | "双编码器" | 独立的图像和文本编码器，以共享维度的投影头结束 |
| 零样本 | "无需任务特定训练" | 仅通过推理时用文本描述的类别进行分类；未接触任何标签 |
| 温度 / logit_scale | "tau" | 可学习的标量，在softmax之前缩放相似度矩阵 |
| 提示模板 | "A photo of a {}" | 围绕类别名称的自然语言包装器；取多个模板的平均值可提升零样本准确率 |
| CLIP | "图像+文本模型" | 2021年OpenAI模型；2026年该领域的通用词汇 |
| SigLIP | "Sigmoid CLIP" | 将softmax替换为逐对sigmoid；在小批量下训练效果更好 |
| OpenCLIP | "开源复现" | 社区在LAION上训练的CLIP变体；开源管道的生产默认选择 |
| VLM | "视觉语言模型" | 一个CLIP系列编码器加上一个LLM，训练用来回答关于图像的问题 |

## 延伸阅读

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language-Image Pre-Training (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — 社区代码库
- [DINOv2 vs CLIP vs MAE: a features comparison](https://huggingface.co/blog/dinov2) — HF指南，包含并排使用案例
