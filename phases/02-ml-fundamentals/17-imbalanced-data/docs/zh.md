# 处理不平衡数据

> 当 99% 的数据都是“正常”时，准确率就是一个谎言。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 2，课程 01-09（尤其是评估指标）  
**时间：** 约 90 分钟  

## 学习目标

- 从零实现 SMOTE，并解释合成过采样与随机复制的区别
- 使用 F1、AUPRC 和马修斯相关系数而非准确率评估不平衡分类器
- 比较类别加权、阈值调整和重采样策略，针对给定的不平衡比率选择合适的方法
- 构建一个完整的不平衡数据处理流程，结合 SMOTE、类别权重和阈值优化

## 问题

你构建了一个欺诈检测模型。它达到了 99.9% 的准确率。你庆祝了一番。然后你发现它对每笔交易都预测为“非欺诈”。

这不是一个 bug。当只有 0.1% 的交易是欺诈时，这样做是理性的。模型学会了总是预测多数类可以最小化整体误差。这在技术上正确，但完全无用。

在真正的分类场景中，这种情况比比皆是。疾病诊断：阳性率 1%。网络入侵：攻击占比 0.01%。制造缺陷：次品率 0.5%。垃圾邮件过滤：垃圾邮件占比 20%。客户流失预测：流失率 5%。越重要的少数类，往往越稀有。

准确率之所以失效，是因为它平等对待所有正确的预测。正确标记一笔合法交易和正确捕获一笔欺诈都算作一个准确率点。但捕获欺诈才是模型存在的全部理由。我们需要能强制模型关注稀有但重要类别的指标、技术和训练策略。

## 核心概念

### 为什么准确率会失效

考虑一个包含 1000 个样本的数据集：990 个负类，10 个正类。一个始终预测负类的模型：

|                 | 预测为正类 | 预测为负类 |
|-----------------|------------|------------|
| 实际为正类      | 0 (TP)     | 10 (FN)    |
| 实际为负类      | 0 (FP)     | 990 (TN)   |

准确率 = (0 + 990) / 1000 = 99.0%

模型没有捕获任何欺诈、任何疾病、任何缺陷。但准确率却显示 99%。这就是为什么在不平衡问题上使用准确率是危险的。

### 更好的指标

**精确率** = TP / (TP + FP)。在所有被标记为正类的样本中，有多少是真正的正类？高精确率意味着误报少。

**召回率** = TP / (TP + FN)。在所有真正的正类样本中，我们捕获了多少？高召回率意味着漏报少。

**F1 分数** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。调和平均值。比算术平均更能惩罚精确率和召回率之间的极端不平衡。

**F-beta 分数** = (1 + beta^2) * 精确率 * 召回率 / (beta^2 * 精确率 + 召回率)。当 beta > 1 时，召回率更重要；当 beta < 1 时，精确率更重要。在欺诈检测中，F2 很常用（漏掉欺诈比误报更严重）。

**AUPRC**（精确率-召回率曲线下面积）。类似 AUC-ROC，但对不平衡数据更有信息量。随机分类器的 AUPRC 等于正类率（不是像 ROC 那样的 0.5），这使得改进更易观察。

**马修斯相关系数** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。取值范围 -1 到 +1。只有当模型在两个类别上都表现良好时才会给出高分。即使类别数量差异很大，该指标也是平衡的。

对于上述“始终预测负类”的模型：精确率 = 0/0（通常视为 0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地将模型识别为无用。

### 不平衡数据处理流程

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE：合成少数类过采样技术

随机过采样会复制已有的少数类样本。这有效但存在过拟合风险，因为模型反复看到相同的点。

SMOTE 创建新的合成少数类样本，这些样本是合理的但不是复制品。算法如下：

1. 对于每个少数类样本 x，找到其 k 个最近的少数类邻居
2. 随机选择一个邻居
3. 在 x 与该邻居之间的线段上创建一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这会插值真实的少数类点之间的区域，在不单纯复制现有数据的情况下，在特征空间的同一区域创建样本。

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 采样策略对比

**随机过采样**：复制少数类样本使其数量与多数类相等。
- 优点：简单，无信息丢失
- 缺点：完全重复导致过拟合，增加训练时间

**随机欠采样**：移除多数类样本使其数量与少数类相等。
- 优点：训练快，简单
- 缺点：丢弃了可能有用的多数类数据，方差更大

**SMOTE**：通过插值创建合成少数类样本。
- 优点：生成新数据点，与随机过采样相比减少过拟合
- 缺点：可能在决策边界附近产生噪声样本，不考虑多数类分布

| 策略       | 数据变化             | 风险         | 何时使用                                     |
|------------|----------------------|--------------|----------------------------------------------|
| 过采样     | 少数类被复制         | 过拟合       | 小数据集，中等不平衡                         |
| 欠采样     | 多数类被移除         | 信息丢失     | 大数据集，追求快训练                         |
| SMOTE      | 添加合成少数类样本   | 边界噪声     | 中等不平衡，有足够少数类样本进行 k-NN       |

### 类别权重

不改变数据，而是改变模型处理错误的方式。对少数类的错误分类赋予更高权重。

对于一个包含 950 个负类和 50 个正类的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得了 19 倍的权重。错误分类一个正类样本的代价等于错误分类 19 个负类样本。模型被迫关注少数类。

在逻辑回归中，这会修改损失函数：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

从期望上看，类别权重在数学上与过采样等价，但不会创建新的数据点。这使得它更快，并避免了复制样本导致的过拟合风险。

### 阈值调整

大多数分类器输出的是概率。默认阈值为 0.5：如果 P(正类) >= 0.5，则预测为正类。但 0.5 是任意的。当类别不平衡时，最优阈值通常要低得多。

流程：
1. 训练模型
2. 在验证集上获取预测概率
3. 在 0.0 到 1.0 之间扫描阈值
4. 计算每个阈值下的 F1（或你选择的指标）
5. 选择使指标最大化的阈值

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

一个模型可能会对欺诈交易输出 P(欺诈) = 0.15。在阈值 0.5 下，这被分类为非欺诈；在阈值 0.10 下，它被正确捕获。概率校准不如排名重要——只要欺诈交易的概率高于非欺诈交易，就存在一个阈值能将它们分开。

### 代价敏感学习

类别权重的泛化。不是采用统一代价，而是指定具体的误分类代价：

|                 | 预测为正类 | 预测为负类 |
|-----------------|------------|------------|
| 实际为正类      | 0（正确）  | C_FN = 100 |
| 实际为负类      | C_FP = 1   | 0（正确）  |

漏掉一笔欺诈交易（FN）的代价是误报（FP）的 100 倍。模型优化的是总代价，而不是总错误数量。

当你能估计现实世界的代价时，这是最原则性的方法。漏诊癌症与引起额外活检的误报有着截然不同的代价。明确这些代价能迫使模型做出正确的权衡。

### 决策流程图

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

## 动手构建

### 步骤 1：生成不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 步骤 2：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 步骤 3：随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 步骤 4：带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 步骤 5：阈值调整

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 步骤 6：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 步骤 7：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件在一个脚本中运行所有这些步骤并打印结果。

## 使用它

借助 scikit-learn 和 imbalanced-learn，这些技术可以一行代码实现：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现的代码精确展示了每种技术实际做了什么。SMOTE 就是少数类上的 k-NN 插值。类别权重就是乘以损失。阈值调整就是遍历截断值的 for 循环。没有魔法。

## 交付成果

本课程产生：
- `outputs/skill-imbalanced-data.md` —— 一份用于处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，使其只对靠近决策边界的少数类点（其 k 近邻中包含多数类样本）生成合成样本。在类别重叠的数据集上与标准 SMOTE 比较结果。

2. **代价矩阵优化**：实现代价敏感学习，其中代价矩阵作为参数。创建一个函数，该函数接收代价矩阵并返回使期望代价最小化的最优预测。使用不同的代价比（1:10, 1:100, 1:1000）进行测试，并绘制精确率-召回率权衡如何变化。

3. **阈值校准**：实现 Platt 缩放（在模型的原始输出上拟合逻辑回归以产生校准概率）。比较校准前后的精确率-召回率曲线。展示校准不会改变排序（AUC 保持不变），但使概率更有意义。

4. **带平衡 Bagging 的集成**：训练多个模型，每个模型在平衡的自助采样（所有少数类样本 + 随机选择的多数类子集）上训练。平均它们的预测。将此方法与单个模型加 SMOTE 进行比较。同时测量性能和跨运行的方差。

5. **不平衡比率实验**：取一个平衡数据集，逐步增加不平衡比率（50/50，70/30，90/10，95/5，99/1）。对于每个比率，分别使用和不使用 SMOTE 进行训练。绘制两种方法下 F1 分数与不平衡比率的关系图。SMOTE 在哪个比率开始产生有意义的影响？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 类别不平衡 | “一个类别的样本远多于另一个” | 数据集中类别的分布严重偏斜，导致模型偏向多数类 |
| SMOTE | “合成过采样” | 通过在现有少数类样本及其 k 个最近少数类邻居之间插值来创建新的少数类样本 |
| 类别权重 | “让对稀有类别的错误更加昂贵” | 使用类别特定的权重乘以损失函数，使模型更严厉地惩罚少数类的误分类 |
| 阈值调整 | “移动决策边界” | 将分类的概率截止值从默认的 0.5 更改为优化特定指标的值 |
| 精确率-召回率权衡 | “鱼和熊掌不可兼得” | 降低阈值可捕获更多正类（召回率提高），但也会标记更多假正类（精确率下降），反之亦然 |
| AUPRC | “PR 曲线下的面积” | 将精确率-召回率曲线总结为一个数字；当类别严重不平衡时，比 AUC-ROC 更有信息量 |
| 马修斯相关系数 | “均衡指标” | 预测标签与真实标签之间的相关系数，只有当模型在两个类别上都表现良好时才会产生高分 |
| 代价敏感学习 | “不同的错误代价不同” | 将现实世界中的误分类代价纳入训练目标，使模型优化总代价而非错误数量 |
| 随机过采样 | “复制少数类” | 重复少数类样本以平衡类别数量；简单但存在复制的点导致过拟合的风险 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) —— SMOTE 原始论文，仍是不平衡学习领域被引用最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) —— 全面综述，涵盖抽样、代价敏感和算法方法
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) —— 包含 SMOTE 变体、欠采样策略和流水线集成的 Python 库
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) —— 何时以及为什么在不平衡问题中应优先使用 PR 曲线而不是 ROC 曲线
