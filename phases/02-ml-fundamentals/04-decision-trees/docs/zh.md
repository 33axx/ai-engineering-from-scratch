# 决策树与随机森林

> 决策树只是一张流程图。但一片森林却是机器学习中最强大的工具之一。

**类型：** 构建  
**语言：** Python  
**前置条件：** 第一阶段（第09课 信息论，第06课 概率）  
**时间：** ~90 分钟  

## 学习目标

- 实现基尼不纯度、熵和信息增益的计算，以找到最优的决策树分裂点
- 从零开始构建一个带有预剪枝控制（最大深度、最小样本数）的决策树分类器
- 使用自助采样和特征随机化构建随机森林，并解释它为何能降低方差
- 比较平均不纯度减少（MDI）特征重要性与排列重要性，并识别MDI何时存在偏差

## 问题

你有一份表格数据。行是样本，列是特征，还有一列你想预测的目标。你可以用神经网络来处理。但对于表格数据，基于树的模型（决策树、随机森林、梯度提升树）始终优于深度学习。Kaggle在结构化数据上的竞赛被XGBoost和LightGBM主导，而不是Transformer。

为什么？树能处理混合特征类型（数值型和类别型），无需预处理。它能处理非线性关系，无需特征工程。它是可解释的：你可以查看树的结构，准确看到预测是如何做出的。而随机森林，通过平均多棵树，在中型数据集上具有很强的抗过拟合能力。

本课程通过递归分裂从零开始构建决策树，然后在此基础上构建随机森林。你将实现分裂准则背后的数学（基尼不纯度、熵、信息增益），并理解为什么弱学习器的集成能成为强学习器。

## 概念

### 决策树的作用

决策树通过提出一系列是/否问题，将特征空间划分为矩形区域。

```mermaid
graph TD
    A["Age < 30?"] -->|Yes| B["Income > 50k?"]
    A -->|No| C["Credit Score > 700?"]
    B -->|Yes| D["Approve"]
    B -->|No| E["Deny"]
    C -->|Yes| F["Approve"]
    C -->|No| G["Deny"]
```

每个内部节点测试一个特征与一个阈值。每个叶子节点做出一个预测。要对一个新的数据点进行分类，从根节点开始，沿着分支走，直到到达一个叶子。

树的构建是自上而下的：在每个节点，选择最能区分数据的特征和阈值。“最好”由分裂准则定义。

### 分裂准则：衡量不纯度

在每个节点，我们有一组样本。我们希望将它们分裂，使得子节点尽可能“纯净”，即每个子节点主要包含一个类别。

**基尼不纯度**衡量的是：如果根据该节点的类别分布对样本随机标注，该样本被错误分类的概率。

```
Gini(S) = 1 - sum(p_k^2)

where p_k is the proportion of class k in set S.
```

对于一个纯节点（全是同一类），基尼 = 0。对于二分类50/50的情况，基尼 = 0.5。越低越好。

```
Example: 6 cats, 4 dogs

Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48
```

**熵**衡量节点中的信息含量（无序度）。在第一阶段第09课中已介绍。

```
Entropy(S) = -sum(p_k * log2(p_k))
```

对于一个纯节点，熵 = 0。对于50/50的二分类，熵 = 1.0。越低越好。

```
Example: 6 cats, 4 dogs

Entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
        = -(0.6 * -0.737 + 0.4 * -1.322)
        = 0.442 + 0.529
        = 0.971 bits
```

**信息增益**是分裂后不纯度（熵或基尼）的减少量。

```
IG(S, feature, threshold) = Impurity(S) - weighted_avg(Impurity(S_left), Impurity(S_right))

where the weights are the proportions of samples in each child.
```

在每个节点上的贪心算法：尝试每个特征和每个可能的阈值。选择使信息增益最大化的（特征，阈值）对。

### 分裂如何工作

对于当前节点上有n个特征和m个样本的数据集：

1. 对每个特征j（j = 1到n）：
   - 按特征j对样本排序
   - 尝试每个连续不同值之间的中点作为阈值
   - 计算每个阈值的信息增益
2. 选择信息增益最高的特征和阈值
3. 将数据分为左支（特征 <= 阈值）和右支（特征 > 阈值）
4. 对每个子节点递归

这种贪心方法不能保证全局最优树。寻找最优树是NP难的。但贪心分裂在实践中效果很好。

### 停止条件

如果没有停止条件，树会一直生长直到每个叶子都是纯的（每个叶子一个样本）。这会完美记忆训练数据，但泛化能力极差。

**预剪枝**在树完全生长之前停止：
- 最大深度：当树达到设定深度时停止分裂
- 叶子最小样本数：如果节点样本数少于k则停止
- 最小信息增益：如果最佳分裂对不纯度的改善小于阈值则停止
- 最大叶子节点数：限制叶子总数

**后剪枝**先生长完整树，再修剪：
- 代价复杂度剪枝（scikit-learn使用）：添加与叶子数量成比例的惩罚项。增大惩罚项以获得更小的树
- 减少误差剪枝：如果去除一个子树后验证误差不增加，则移除该子树

预剪枝更简单、更快。后剪枝通常能产生更好的树，因为它不会过早停止那些可能带来有用后续分裂的分裂。

### 用于回归的决策树

对于回归，叶子的预测是该叶子中目标值的均值。分裂准则也会变化：

**方差减少**取代信息增益：

```
VR(S, feature, threshold) = Var(S) - weighted_avg(Var(S_left), Var(S_right))
```

选择使方差减少最多的分裂。树将输入空间划分为多个区域，并在每个区域预测一个常数（均值）。

### 随机森林：集成的力量

单棵决策树具有高方差。数据微小的变化可能导致完全不同的树。随机森林通过平均多棵树来解决这个问题。

```mermaid
graph TD
    D["Training Data"] --> B1["Bootstrap Sample 1"]
    D --> B2["Bootstrap Sample 2"]
    D --> B3["Bootstrap Sample 3"]
    D --> BN["Bootstrap Sample N"]
    B1 --> T1["Tree 1<br>(random feature subset)"]
    B2 --> T2["Tree 2<br>(random feature subset)"]
    B3 --> T3["Tree 3<br>(random feature subset)"]
    BN --> TN["Tree N<br>(random feature subset)"]
    T1 --> V["Aggregate Predictions<br>(majority vote or average)"]
    T2 --> V
    T3 --> V
    TN --> V
```

两种随机性使树变得多样化：

**装袋（Bootstrap Aggregating）：** 每棵树都在一个自助样本（从训练数据中有放回地随机抽取的样本）上训练。每个自助样本包含大约63%的原始样本（其余的是袋外样本，可用于验证）。

**特征随机化：** 在每个分裂点，只考虑一个随机子集的特征。对于分类，默认是 sqrt(特征数)。对于回归，是 特征数/3。这防止了所有树都在同一个主导特征上分裂。

关键洞察：平均许多去相关的树可以降低方差，而不增加偏差。每棵单独的树可能平庸。但集成是强大的。

### 特征重要性

随机森林自然提供了特征重要性评分。最常见的方法：

**平均不纯度减少（MDI）：** 对于每个特征，将该特征在所有树的所有节点中导致的不纯度减少量相加。在更早分裂中产生更大不纯度减少的特征更重要。

```
importance(feature_j) = sum over all nodes where feature_j is used:
    (n_samples_at_node / n_total_samples) * impurity_decrease
```

这种方法很快（在训练期间计算），但对高基数特征和具有许多可能分裂点的特征有偏差。

**排列重要性**是另一种方法：随机打乱一个特征的值，衡量模型准确率下降多少。更可靠，但更慢。

### 树何时优于神经网络

在表格数据上，树和森林优于神经网络。原因如下：

| 因素 | 树 | 神经网络 |
|--------|-------|----------------|
| 混合类型（数值 + 类别） | 原生支持 | 需要编码 |
| 小数据集（< 10k 行） | 效果很好 | 容易过拟合 |
| 特征交互 | 通过分裂发现 | 需要架构设计 |
| 可解释性 | 完全透明 | 黑箱 |
| 训练时间 | 几分钟 | 几小时 |
| 超参数敏感度 | 低 | 高 |

当数据具有空间或序列结构（图像、文本、音频）时，神经网络胜出。对于扁平的特征表格，树的默认选择。

## 动手构建

### 第1步：基尼不纯度和熵

从零开始构建两个分裂准则，并验证它们在哪些分裂较好上达成一致。

```python
import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )
```

### 第2步：找到最佳分裂

尝试每个特征和每个阈值。返回信息增益最高的那个。

```python
def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity
```

### 第3步：构建DecisionTree类

递归分裂、预测以及特征重要性追踪。

```python
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
```

### 第4步：构建RandomForest类

自助采样、特征随机化以及多数投票。

```python
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for preds in all_preds:
                v = preds[i]
                votes[v] = votes.get(v, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions
```

完整实现及所有辅助方法请参见 `code/trees.py`。

## 使用它

使用scikit-learn，训练随机森林只需三行代码：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Feature importances: {rf.feature_importances_}")
```

在实践中，梯度提升树（XGBoost、LightGBM、CatBoost）通常比随机森林更强，因为它们顺序构建树，每棵树纠正前一棵树的错误。但随机森林更难配置出错，且几乎不需要超参数调优。

## 交付物

本课程产生 `outputs/prompt-tree-interpreter.md` —— 一个用于向业务利益相关者解释决策树分裂的提示。将训练好的树的结构（深度、特征、分裂阈值、准确率）输入其中，它会将模型转化为自然语言的规则，对特征重要性进行排序，标记过拟合或数据泄露，并推荐下一步。每当需要向不阅读代码的人解释基于树的模型时，都可以使用它。

## 练习

1. 在具有3个类别的2D数据集上训练一棵单决策树。手动追踪分裂并绘制矩形决策边界。比较 max_depth=2 和 max_depth=10 时的边界。
2. 为回归树实现方差减少分裂。生成 y = sin(x) + 噪声（200个点），拟合你的回归树。绘制树的分段常数预测曲线与真实曲线对比。
3. 构建包含1、5、10、50、200棵树的随机森林。绘制训练准确率和测试准确率随树数量的变化图。观察到测试准确率趋于平稳但不下降（随机森林对过拟合有抵抗力）。
4. 在5个不同的数据集上比较基尼不纯度和熵作为分裂准则。测量准确率和树深度。大多数情况下结果几乎相同。解释原因。
5. 实现排列重要性。在一个特征为随机噪声但具有高基数的数据集上，将其与MDI重要性进行比较。MDI会将噪声特征排名很高，而排列重要性不会。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 决策树 | “一个预测流程图” | 通过学习一系列if/else分裂，将特征空间划分为矩形区域的模型 |
| 基尼不纯度 | “节点有多混杂” | 在一个节点随机抽样错误分类的概率。0=纯净，对于二分类0.5=最大不纯度 |
| 熵 | “节点中的无序度” | 节点中的信息含量。0=纯净，对于二分类1.0=最大不确定性。来自信息论 |
| 信息增益 | “分裂有多好” | 分裂后不纯度的减少量。选择分裂的贪心准则 |
| 预剪枝 | “提前停止树生长” | 通过设置最大深度、最小样本数或最小增益阈值提前停止树生长 |
| 后剪枝 | “先生长再修剪” | 先生成完整树，然后移除不提升验证性能的子树 |
| 装袋 | “在随机子集上训练” | 自助聚合。每棵树在训练数据的不同有放回随机样本上训练 |
| 随机森林 | “一堆树” | 决策树的集成，每棵树在自助样本上训练，并在每个分裂点使用随机特征子集 |
| 特征重要性（MDI） | “哪些特征重要” | 每个特征贡献的总不纯度减少量，跨所有树和节点求和 |
| 排列重要性 | “打乱再检查” | 随机打乱一个特征的值后准确率下降的值。对于噪声特征比MDI更可靠 |
| 方差减少 | “回归版本的信息增益” | 回归树中信息增益的类比。选择使目标方差减少最多的分裂 |
| 自助样本 | “重复的随机样本” | 从原始数据集中有放回抽取的随机样本。大小相同，但包含重复 |

## 延伸阅读

- [Breiman: Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324) - 原始随机森林论文
- [Grinsztajn et al.: Why do tree-based models still outperform deep learning on tabular data? (2022)](https://arxiv.org/abs/2207.08815) - 关于表格任务中树与神经网络的严格比较
- [scikit-learn Decision Trees documentation](https://scikit-learn.org/stable/modules/tree.html) - 带有可视化工具的实用指南
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754) - 主导Kaggle的梯度提升论文
