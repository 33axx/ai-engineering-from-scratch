# K近邻与距离

> 存储所有数据，通过查看邻居进行预测。最简单却实际有效的算法。

**类型：** 动手实践  
**语言：** Python  
**前置要求：** 第一阶段（第14课 范数与距离）  
**时长：** 约90分钟  

## 学习目标

- 从零实现K近邻分类与回归，支持可配置的K值和距离加权投票  
- 比较L1、L2、余弦以及闵可夫斯基距离度量，并能根据数据类型选择适当的度量  
- 解释维度灾难，并论证为什么KNN在高维空间中效果下降  
- 构建KD树实现高效的最近邻搜索，并分析其优于暴力搜索的条件  

## 问题描述

你有一个数据集。一个新的数据点到来。你需要对它进行分类或预测其值。与线性回归或支持向量机等从数据中学习参数的方法不同，你只需找到离新点最近的K个训练点，让它们投票。

这就是K近邻。没有训练阶段，没有需要学习的参数，没有要最小化的损失函数。你存储整个训练集，并在预测时计算距离。

听起来简单得不像能工作。但KNN在许多问题上出人意料地有竞争力，尤其是在中小型数据集上。深入理解它能揭示基本概念：距离度量的选择（与第一阶段第14课衔接）、维度灾难，以及惰性学习与急切学习的区别。

此外，KNN在现代AI中无处不在，只是名称不同。向量数据库对嵌入进行KNN搜索。检索增强生成（RAG）找到最近的K个文档块。推荐系统找到相似的用户或物品。算法相同，只是规模和数据结构不同。

## 概念

### KNN的工作原理

给定一个带标签的数据集和一个新的查询点：

1. 计算查询点到数据集中每个点的距离  
2. 按距离排序  
3. 取最近的K个点  
4. 对于分类：K个邻居进行多数投票  
5. 对于回归：对K个邻居的值求平均（或加权平均）  

```python
import numpy as np

def knn_predict(X_train, y_train, query, k=3, distance_fn=None):
    distances = [distance_fn(query, x) for x in X_train]
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return np.bincount(nearest_labels).argmax()
```

这就是整个算法。没有拟合，没有梯度下降，没有迭代轮次。

### 选择K

K是唯一的超参数，控制偏差-方差权衡：

| K | 行为 |
|---|------|
| K=1 | 决策边界跟随每个点，训练误差为零，方差高，过拟合 |
| 小K（3-5） | 对局部结构敏感，能捕捉复杂边界 |
| 大K | 边界更平滑，对噪声更鲁棒，可能欠拟合 |
| K=N | 对所有点预测多数类，偏差最大 |

常见起点是对N个点的数据集取K=√N。对二分类使用奇数K以避免平局。

```python
def choose_k(n_samples):
    return int(np.sqrt(n_samples)) | 1  # 确保为奇数
```

### 距离度量

距离函数定义了“近”的含义。不同的度量会产生不同的邻居、不同的预测。

**L2（欧几里得距离）** 是默认选择。直线距离。

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

对特征尺度敏感。在KNN中使用L2之前，务必标准化特征。

**L1（曼哈顿距离）** 对绝对差求和。由于没有平方差，对异常值比L2更鲁棒。

```python
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))
```

**余弦距离** 测量向量之间的角度，忽略幅值。对文本和嵌入数据至关重要。

```python
def cosine_distance(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - cos_sim
```

**闵可夫斯基距离** 通过参数p泛化L1和L2。

```python
def minkowski_distance(a, b, p):
    return np.sum(np.abs(a - b) ** p) ** (1/p)
```

使用哪种度量取决于数据：

| 数据类型 | 最佳度量 | 原因 |
|---------|---------|------|
| 数值特征，尺度相近 | L2（欧几里得） | 默认，适用于空间数据 |
| 数值特征，有异常值 | L1（曼哈顿） | 鲁棒，不会放大大的差异 |
| 文本嵌入 | 余弦 | 幅值是噪声，方向是意义 |
| 高维稀疏数据 | 余弦或L1 | L2受维度灾难影响大 |
| 混合类型 | 自定义距离 | 按特征类型组合度量 |

### 加权KNN

标准KNN对所有K个邻居给予相同权重。但距离0.1的邻居应该比距离5.0的邻居更重要。

**距离加权KNN** 按距离的倒数对每个邻居加权：

```python
def weighted_knn_predict(X_train, y_train, query, k=3, distance_fn=None):
    distances = [distance_fn(query, x) for x in X_train]
    nearest_indices = np.argsort(distances)[:k]
    nearest_distances = [distances[i] for i in nearest_indices]
    nearest_labels = y_train[nearest_indices]
    weights = 1 / (np.array(nearest_distances) + 1e-8)
    return np.average(nearest_labels, weights=weights).round()
```

epsilon防止查询点与训练点完全匹配时分母为零。

加权KNN对K的选择不那么敏感，因为远处的邻居无论如何贡献都很小。

### 维度灾难

KNN的性能在高维中下降。这不是模糊的担忧，而是数学事实。

**问题1：距离收敛。** 随着维度增加，最大距离与最小距离之比趋近于1。所有点与查询点都“同样远”。

```python
def curse_demo(max_dim=500):
    n_points = 1000
    ratios = []
    for d in range(1, max_dim + 1):
        points = np.random.uniform(0, 1, (n_points, d))
        dists = np.linalg.norm(points[0] - points[1:], axis=1)
        ratios.append(np.max(dists) / np.min(dists))
    return ratios
```

**问题2：体积爆炸。** 要在一个固定比例的数据内捕获K个邻居，你需要扩展搜索半径，覆盖特征空间中更大的比例。高维中的“邻域”实际上包含了大部分空间。

**问题3：角落主导。** 在d维单位超立方体中，大部分体积集中在角落附近，而不是中心。立方体内接球所含的体积随着d增长而趋近于零。

实际后果：KNN在特征数约20-50以下时效果良好。超过这个范围，需要在应用KNN之前进行降维（PCA、UMAP、t-SNE），或者使用利用数据固有低维性的树形搜索结构。

### KD树：快速最近邻搜索

暴力KNN计算查询点到每个训练点的距离，每次查询复杂度O(n·d)。对大型数据集来说太慢。

KD树沿特征轴递归划分空间。每一层在某一维度的中位数处进行分割。

```python
class KDTreeNode:
    def __init__(self, points, labels, depth=0):
        if len(points) == 0:
            self.is_leaf = True
            self.points = np.array([])
            self.labels = np.array([])
            return
        
        n_features = points.shape[1]
        axis = depth % n_features
        sorted_idx = np.argsort(points[:, axis])
        points = points[sorted_idx]
        labels = labels[sorted_idx]
        median_idx = len(points) // 2
        
        self.is_leaf = False
        self.point = points[median_idx]
        self.label = labels[median_idx]
        self.axis = axis
        self.left = KDTreeNode(points[:median_idx], labels[:median_idx], depth + 1)
        self.right = KDTreeNode(points[median_idx + 1:], labels[median_idx + 1:], depth + 1)
```

要找到最近邻，遍历树到包含查询点的叶子节点，然后回溯，只检查可能包含更近点的相邻分区。

平均查询时间：低维时为O(log n)。但在高维（d>20）下KD树退化为O(n)，因为回溯时能排除的分支越来越少。

### 球树：更适合中等维度

球树将数据划分为嵌套的超球体，而不是轴对齐的盒子。每个节点定义一个球（中心+半径），包含该子树的所有点。

相对于KD树的优势：
- 在中等维度（最高约50）下效果更好
- 能处理非轴对齐结构
- 更紧密的包围体积使搜索时能剪去更多分支

KD树和球树都是精确算法。对于真正的大规模搜索（数百万点，数百维），则使用近似最近邻方法（HNSW、IVF、乘积量化），这些在第一阶段第14课中涉及。

### 惰性学习 vs 急切学习

KNN是惰性学习者：训练时不执行任何工作，所有工作在预测时完成。大多数其他算法（线性回归、SVM、神经网络）是急切学习者：在训练时进行大量计算以构建紧凑模型，预测则很快。

| 方面 | 惰性（KNN） | 急切（SVM、神经网络） |
|------|-------------|----------------------|
| 训练时间 | O(1)，仅存储数据 | O(n·轮次) |
| 预测时间 | 每次查询O(n·d) | O(d) 或 O(参数) |
| 预测时内存占用 | 存储整个训练集 | 仅存储模型参数 |
| 适应新数据 | 即时添加点 | 重新训练模型 |
| 决策边界 | 隐式，实时计算 | 显式，训练后固定 |

惰性学习在以下情况理想：
- 数据集频繁变化（增删点无需重新训练）
- 只需对非常少的查询进行预测
- 想要零训练时间
- 数据集足够小，暴力搜索很快

### KNN用于回归

KNN回归不是多数投票，而是对K个邻居的目标值求平均。

```python
def knn_regression(X_train, y_train, query, k=3, distance_fn=None):
    distances = [distance_fn(query, x) for x in X_train]
    nearest_indices = np.argsort(distances)[:k]
    return np.mean(y_train[nearest_indices])
```

KNN回归产生分段常数（加权时则为分段平滑）的预测。它不能外推到训练数据范围之外。如果训练目标都在0到100之间，KNN永远不会预测200。

## 动手实现

### 第一步：距离函数

实现L1、L2、余弦和闵可夫斯基距离。这些直接与第一阶段第14课衔接。

```python
# L1 距离
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# L2 距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 余弦距离
def cosine_distance(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1 - cos_sim

# 闵可夫斯基距离（p为超参数）
def minkowski_distance(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)
```

### 第二步：KNN分类器和回归器

构建完整的KNN，支持可配置的K、距离度量以及可选的距离加权。

```python
class KNN:
    def __init__(self, k=5, distance_fn='euclidean', weighted=False):
        self.k = k
        self.weighted = weighted
        if distance_fn == 'euclidean':
            self.distance = euclidean_distance
        elif distance_fn == 'manhattan':
            self.distance = manhattan_distance
        elif distance_fn == 'cosine':
            self.distance = cosine_distance
        else:
            raise ValueError("Unsupported distance function")
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _predict_one(self, x):
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        if self.weighted:
            nearest_distances = np.array([distances[i] for i in nearest_indices])
            weights = 1 / (nearest_distances + 1e-8)
            return np.average(nearest_labels, weights=weights).round()
        else:
            return np.bincount(nearest_labels).argmax()
    
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

class KNNRegressor(KNN):
    def _predict_one(self, x):
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_values = self.y_train[nearest_indices]
        if self.weighted:
            nearest_distances = np.array([distances[i] for i in nearest_indices])
            weights = 1 / (nearest_distances + 1e-8)
            return np.average(nearest_values, weights=weights)
        else:
            return np.mean(nearest_values)
```

### 第三步：KD树实现高效搜索

从零构建KD树，沿每一维的中位数递归分割。

```python
class KDTree:
    def __init__(self, points, labels=None, depth=0):
        self.points = np.array(points)
        self.labels = np.array(labels) if labels is not None else None
        self.depth = depth
        self._build()
    
    def _build(self):
        n = len(self.points)
        if n == 0:
            self.is_leaf = True
            return
        if n == 1:
            self.is_leaf = True
            return
        
        self.is_leaf = False
        self.axis = self.depth % self.points.shape[1]
        sorted_idx = np.argsort(self.points[:, self.axis])
        self.points = self.points[sorted_idx]
        if self.labels is not None:
            self.labels = self.labels[sorted_idx]
        median = n // 2
        self.median_point = self.points[median]
        self.median_label = self.labels[median] if self.labels is not None else None
        
        left_points = self.points[:median]
        left_labels = self.labels[:median] if self.labels is not None else None
        right_points = self.points[median + 1:]
        right_labels = self.labels[median + 1:] if self.labels is not None else None
        
        self.left = KDTree(left_points, left_labels, self.depth + 1)
        self.right = KDTree(right_points, right_labels, self.depth + 1)
    
    def query(self, query_point, k=1):
        best_dist = float('inf')
        best_points = []
        self._search(query_point, k, best_dist, best_points)
        return best_points
    
    def _search(self, query, k, best_dist, best_points):
        if self.is_leaf:
            dist = np.linalg.norm(query - self.points)
            if dist < best_dist:
                best_dist = dist
                best_points = self.points
            return best_dist, best_points
        
        axis = self.axis
        diff = query[axis] - self.median_point[axis]
        if diff <= 0:
            best_dist, best_points = self.left._search(query, k, best_dist, best_points)
            if abs(diff) < best_dist:
                best_dist, best_points = self.right._search(query, k, best_dist, best_points)
        else:
            best_dist, best_points = self.right._search(query, k, best_dist, best_points)
            if abs(diff) < best_dist:
                best_dist, best_points = self.left._search(query, k, best_dist, best_points)
        return best_dist, best_points
```

完整实现（包括所有辅助方法和演示）请参见 `code/knn.py`。

### 第四步：特征缩放

KNN需要特征缩放，因为距离对特征幅值敏感。取值范围0到1000的特征会主导取值范围0到1的特征。

```python
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def min_max_scale(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)
```

## 使用

使用 scikit-learn：

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 缩放特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 分类
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='distance')
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# 回归
knn_reg = KNeighborsRegressor(n_neighbors=5, metric='manhattan')
knn_reg.fit(X_train_scaled, y_train)
y_pred = knn_reg.predict(X_test_scaled)
```

Scikit-learn 在数据集足够大且维度足够低时自动使用KD树或球树。对于高维数据，它退化为暴力搜索。可以通过 `algorithm` 参数控制。

对于大规模最近邻搜索（数百万向量），使用 FAISS、Annoy 或向量数据库：

```python
import faiss
dimension = 128
index = faiss.IndexFlatL2(dimension)
index.add(X_train)  # X_train形状为 (n, dim)
distances, indices = index.search(query, k=5)
```

## 练习

1. 在包含3个类别的2D数据集上实现KNN分类。绘制K=1、K=5、K=15和K=N时的决策边界。观察从过拟合到欠拟合的转变。

2. 在2、5、10、50、100和500维上各生成1000个随机点。对每个维度，计算最大成对距离与最小成对距离之比。绘制比例与维度的关系图，以可视化维度灾难。

3. 在文本分类问题（使用TF-IDF向量）上比较L1、L2和余弦距离用于KNN的效果。哪种度量准确率最高？为什么余弦距离通常对文本效果最佳？

4. 实现KD树，并在2D、10D和50D下分别对1k、10k和100k点的数据集测量查询时间与暴力搜索的比较。在什么维度下KD树不再比暴力搜索快？

5. 构建加权KNN回归器，目标为 y = sin(x) + 噪声。与未加权KNN在K=3、10、30时进行比较。证明加权能产生更平滑的预测，尤其在大K时。

## 关键术语

| 术语 | 实际含义 |
|------|---------|
| K近邻 | 非参数算法，通过找到离查询点最近的K个训练点来进行预测 |
| 惰性学习 | 训练时不进行计算，所有工作在预测时完成。KNN是典型例子 |
| 急切学习 | 训练时进行大量计算以构建紧凑模型。大多数机器学习算法是急切学习 |
| 维度灾难 | 在高维中距离收敛，邻域扩展覆盖大部分空间，导致KNN失效 |
| KD树 | 沿特征轴递归划分空间的二叉树。低维下查询时间为O(log n) |
| 球树 | 嵌套超球体构成的树。在中等维度（最高约50）下比KD树效果更好 |
| 加权KNN | 邻居按距离倒数加权。更近的邻居对预测影响更大 |
| 特征缩放 | 将特征归一化到可比范围。对基于距离的方法（如KNN）是必需的 |
| 多数投票 | 通过统计K个邻居中哪个类别出现最多进行分类 |
| 暴力搜索 | 计算查询点到每个训练点的距离，每次查询O(n·d)，精确但大n时慢 |
| 近似最近邻 | 比精确搜索快得多的算法（HNSW、LSH、IVF），寻找近似最近的点 |
| 沃罗诺伊图 | 将空间划分为每个区域包含离一个训练点比任何其他点更近的所有点。K=1的KNN产生沃罗诺伊边界 |

## 进一步阅读

- [Cover & Hart: Nearest Neighbor Pattern Classification (1967)](https://ieeexplore.ieee.org/document/1053964) —— KNN奠基论文，证明其误差率至多为贝叶斯最优的两倍  
- [Friedman, Bentley, Finkel: An Algorithm for Finding Best Matches in Logarithmic Expected Time (1977)](https://dl.acm.org/doi/10.1145/355744.355745) —— KD树原始论文  
- [Beyer et al.: When Is "Nearest Neighbor" Meaningful? (1999)](https://link.springer.com/chapter/10.1007/3-540-49257-7_15) —— 对最近邻维度灾难的形式化分析  
- [scikit-learn Nearest Neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html) —— 带算法选择的实用指南  
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) —— Meta的十亿级近似最近邻搜索库
