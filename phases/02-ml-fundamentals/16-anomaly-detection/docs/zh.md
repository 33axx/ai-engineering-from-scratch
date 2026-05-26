# 异常检测

> 正常容易定义。异常就是任何不符合正常的事物。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段，第01-09课
**时间：** 约75分钟

## 学习目标

- 从头实现Z-score、IQR和孤立森林异常检测方法
- 区分点异常、情境异常和集体异常，并为每种异常选择合适的检测方法
- 解释为什么异常检测被定义为对正常数据建模，而不是对异常进行分类
- 比较无监督异常检测与有监督分类，评估新颖异常覆盖率和精确率之间的权衡

## 问题

一张信用卡下午2点在纽约使用，然后下午2:05在东京使用。一个工厂传感器读数为150度，而正常范围是80-120度。一台服务器每秒发送50,000个请求，而日均值为200。

这些都是异常。发现它们很重要。欺诈造成数十亿美元的损失。设备故障导致停机时间。网络入侵导致数据泄露。

挑战在于：你很少有标记的异常样本。欺诈只占交易的0.1%。设备故障一年才发生几次。你无法训练一个标准的分类器，因为“异常”类别几乎没有什么可以学习。即使你有一些标签，你见过的异常也不是你将遇到的唯一类型。明天的欺诈手法与今天的不同。

异常检测反转了问题。不是学习什么是异常，而是学习什么是正常。任何偏离正常的事物都是可疑的。这种方法不需要标签，能适应新型异常，并且可以扩展到海量数据集。

## 概念

### 异常的类型

并非所有异常都是一样的：

- **点异常。** 单个数据点，无论上下文如何都是不寻常的。温度读数为500度。一笔50,000美元的交易，而账户通常只花50美元。
- **情境异常。** 一个数据点在其上下文中是不寻常的。温度90度在夏天是正常的，在冬天则是异常的。同样的值，不同的上下文。
- **集体异常。** 一组数据点作为一个整体是不寻常的，即使每个单独的点可能是正常的。五次登录失败是正常的。连续五十次就是暴力破解攻击。

大多数方法检测点异常。情境异常需要时间或位置特征。集体异常需要序列感知方法。

```python
# 异常类型示例
import numpy as np

# 点异常：单个极值
temp_readings = np.array([80, 82, 81, 79, 500, 83, 78, 82, 81, 80])
# 500是点异常

# 情境异常：正常值，在错误的时间
hourly_temps = np.array([30, 28, 25, 20, 15, 10, 8, 12, 18, 22, 28, 35])
# 35度在午夜是异常的（情境），但在中午正常

# 集体异常：行为序列
login_attempts = np.array([1, 0, 2, 1, 0, 0, 1, 55, 60, 45, 50, 1, 0])
# [55, 60, 45, 50]是集体异常——连续大量尝试
```

### 无监督框架

在标准分类中，你有两个类别的标签。在异常检测中，你通常面临三种情况之一：

1. **完全无监督。** 完全没有标签。你在所有数据上拟合检测器，并希望异常足够稀少，不会污染“正常”模型。
2. **半监督。** 你有一个干净的仅包含正常数据的数据集。你在这个干净集上拟合，并对其他所有数据进行评分。如果可能，这是最强有力的设置。
3. **弱监督。** 你有少量标记的异常。将它们用于评估，而不是训练。进行无监督训练，然后在标记的子集上测量精确率/召回率。

关键见解：异常检测从根本上不同于分类。你在对正常数据的分布建模，而不是两个类别之间的决策边界。

### 有监督与无监督：权衡

如果你确实有标记的异常，是应该将它们用于训练（有监督分类）还是仅用于评估（无监督检测）？

**有监督（当作分类处理）：**
- 捕获你之前见过的确切异常类型
- 在已知异常类型上精确率更高
- 完全遗漏新型异常
- 当新型异常出现时需要重新训练
- 需要足够的异常样本（通常太少）

**无监督（对正常建模，标记偏差）：**
- 捕获任何偏离正常的情况，包括新型异常
- 不需要标记的异常
- 误报率较高（并非所有不寻常的事情都是坏的）
- 对分布漂移更鲁棒

在实践中，最好的系统结合两者：无监督检测用于广泛覆盖，有监督模型用于已知的高优先级异常类型，以及人工审查用于模糊情况。

### Z-Score方法

最简单的方法。计算每个特征的均值和标准差。标记任何距离均值超过k个标准差的点。

```python
def zscore_detect(X, threshold=3.0):
    """基于Z-score标记异常点"""
    # 按特征标准化
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    # 如果任何特征的|z-score| > 阈值，则为异常
    return (np.abs(X_std) > threshold).any(axis=1)
```

默认阈值是3.0（在正态分布下，99.7%的正常数据落在3个标准差内）。

**优势：** 简单。快速。可解释（“这个值距正常值4.5个标准差”）。

**劣势：** 假设数据服从正态分布。对训练数据中的异常值敏感（异常值会偏移均值和膨胀标准差，使它们更难被检测）。在多峰分布上失效。

**适用场景：** 单特征监控，数据大致呈钟形分布。服务器响应时间、制造公差、具有稳定基线的传感器读数。

**失效场景：** 多簇数据（两个办公地点具有不同的基线温度）、偏态数据（交易金额，其中1000美元很少见但并非异常）、训练集中包含异常值的数据。

### IQR方法

比Z-score更鲁棒。使用四分位距而不是均值和标准差。

```python
def iqr_detect(X, factor=1.5):
    """基于IQR标记异常点"""
    # 按特征计算四分位数
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    # 定义界限
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    # 标记任何超出界限的点
    return ((X < lower) | (X > upper)).any(axis=1)
```

默认因子为1.5。

**优势：** 对异常值鲁棒（百分位数不受极端值影响）。适用于偏态分布。无正态性假设。

**劣势：** 仅适用于单变量（每个特征独立应用）。无法检测仅在特征联合考虑时才异常的样本（一个点可能在每个特征单独看都正常，但在联合空间中是异常的）。

**实用说明：** IQR中的1.5因子对应箱线图中的须。须外的点是潜在的异常值。使用3.0而不是1.5会使检测器更保守（更少标记，更少误报）。正确的因子取决于你对误报的容忍度。

### 孤立森林

关键见解：异常数量少且与众不同。在数据的随机划分中，异常更容易被孤立——它们需要更少的随机分割就能与其余数据分开。

```python
class IsolationTree:
    """单一无监督树，通过随机分割空间来隔离点"""
    def __init__(self, height_limit=None):
        self.height_limit = height_limit
        self.root = None
    
    def fit(self, X):
        self.root = self._build_tree(X, current_height=0)
    
    def path_length(self, x):
        return self._path_length(x, node=self.root, current_height=0)
```

**工作原理：**
1. 构建许多随机树（一个孤立森林）
2. 在每个节点，随机选择一个特征以及该特征最小值和最大值之间的一个随机分割值
3. 持续分割直到每个点都被孤立（在自己的叶子中）
4. 异常值在所有树中的平均路径长度更短

**为什么有效：** 正常点位于密集区域。需要多次随机分割才能将一个点与其邻居隔离。异常点位于稀疏区域。一次或两次随机分割就足以隔离它们。

异常分数基于所有树中的平均路径长度，通过随机二叉搜索树的期望路径长度进行归一化：

```python
def anomaly_score(avg_path_length, n_samples):
    """将平均路径长度转换为[0.5, 1]范围内的异常分数，越高越异常"""
    c = expected_path_length(n_samples)  # 归一化常数
    return 2 ** (-avg_path_length / c)
```

其中`c(n)`是n个样本的期望路径长度。接近1的分数表示异常。接近0.5的分数表示正常。接近0的分数表示非常正常（位于密集簇的深处）。

**优势：** 无分布假设。适用于高维数据。可扩展性好（因为每棵树使用子样本，所以复杂度与样本量成次线性关系）。处理混合特征类型。

**劣势：** 在密集区域中的异常（掩盖效应）上表现不佳。当许多特征不相关时，随机分割效果较差。

**关键超参数：**
- `n_estimators`：树的数量。通常100就够了。更多的树提供更稳定的分数，但计算更慢。
- `max_samples`：每棵树的样本数。原始论文中默认值为256。较小的值使单棵树精度降低但增加多样性。子采样是孤立森林快速的原因——每棵树只看到一小部分数据。
- `contamination`：预期的异常比例。仅用于设置阈值。不影响分数本身。

### 局部异常因子（LOF）

LOF比较点周围的局部密度与其邻居周围的密度。位于稀疏区域但被密集区域包围的点是异常的。

**工作原理：**
1. 对于每个点，找到它的k个最近邻
2. 计算局部可达密度（邻域有多密集）
3. 将每个点的密度与其邻居的密度进行比较
4. 如果一个点的密度远低于其邻居，则它是异常值

**LOF分数：**
- LOF接近1.0表示与邻居密度相似（正常）
- LOF大于1.0表示密度低于邻居（潜在异常）
- LOF远大于1.0（例如2.0以上）表示密度显著较低（很可能是异常）

“局部”部分至关重要。考虑一个包含两个簇的数据集：一个密集簇有1000个点，一个稀疏簇有50个点。稀疏簇边缘上的一个点在全局上并不异常——它有50个邻居。但如果它的直接邻居比它更密集，则在局部是异常的。LOF捕捉到了全局方法忽略的这种细微差别。

**优势：** 检测局部异常（在其邻域中异常的点，即使它们在全局上并非异常）。适用于不同密度的簇。

**劣势：** 在大数据集上速度慢（朴素实现为O(n^2)）。对k的选择敏感。在非常高维的数据上效果不佳（维数灾难影响距离计算）。

### 比较

| 方法 | 假设 | 速度 | 处理高维 | 检测局部异常 |
|--------|------------|-------|-------------------|------------------------|
| Z-score | 正态分布 | 非常快 | 是（按特征） | 否 |
| IQR | 无（按特征） | 非常快 | 是（按特征） | 否 |
| 孤立森林 | 无 | 快 | 是 | 部分 |
| LOF | 距离有意义 | 慢 | 不佳 | 是 |

### 评估挑战

评估异常检测器比评估分类器更难：

- **极端类别不平衡。** 如果异常只占0.1%，那么对所有样本都预测“正常”可以得到99.9%的准确率。准确率毫无用处。
- **AUROC具有误导性。** 在严重不平衡的情况下，AUROC可能看起来不错，即使模型在实际阈值下漏掉了大多数异常。
- **更好的指标：** Precision@k（在排名最高的k个标记项中，有多少是真正的异常）、AUPRC（精确率-召回率曲线下面积）以及在固定误报率下的召回率。

```python
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score

def evaluate_detector(y_true, y_pred, y_scores=None, k=100):
    """评估异常检测器，关注不平衡设置"""
    results = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision_at_k': precision_at_k(y_true, y_scores, k)
    }
    if y_scores is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        results['auprc'] = auc(recall, precision)
    return results
```

### 异常检测流程

在实践中，异常检测遵循以下工作流程：

1. **收集基线数据。** 理想情况下，是一段你确信没有（或极少）异常的时间。
2. **特征工程。** 原始特征加上派生特征（滚动统计量、时间特征、比率）。
3. **训练检测器。** 在基线数据上拟合。模型学习“正常”是什么样的。
4. **对新数据评分。** 每个新观察值获得一个异常分数。
5. **选择阈值。** 选择分数截止点。这是一个业务决策：阈值越高，误报越少，但漏报的异常越多。
6. **警报和调查。** 标记的点提交人工审查或自动响应。
7. **收集反馈。** 记录标记项是真正的异常还是误报。使用这些数据评估检测器并随时间调整阈值。

流程永远不会“完成”。数据分布会漂移，新型异常会出现，阈值需要调整。将异常检测视为一个活的系统，而不是一次性的模型。

## 构建它

`code/anomaly_detection.py`中的代码从头实现了Z-score、IQR和孤立森林。

### Z-Score检测器

```python
class ZScoreDetector:
    """基于Z-score的异常检测器"""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X):
        """计算每个特征的均值和标准差"""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def predict(self, X):
        """返回布尔数组：True表示异常"""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        return np.any(z_scores > self.threshold, axis=1)
    
    def score(self, X):
        """返回最大绝对Z-score作为异常分数"""
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        return np.max(z_scores, axis=1)
```

简单且向量化。如果任何特征超过阈值，则标记该点。

### IQR检测器

```python
class IQRDetector:
    """基于IQR的异常检测器"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        self.Q1_ = None
        self.Q3_ = None
        self.iqr_ = None
    
    def fit(self, X):
        """计算每个特征的四分位数"""
        self.Q1_ = np.percentile(X, 25, axis=0)
        self.Q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.Q3_ - self.Q1_
        return self
    
    def predict(self, X):
        """返回布尔数组：True表示异常"""
        lower = self.Q1_ - self.factor * self.iqr_
        upper = self.Q3_ + self.factor * self.iqr_
        return np.any((X < lower) | (X > upper), axis=1)
    
    def score(self, X):
        """返回最大超出界限距离作为异常分数"""
        lower = self.Q1_ - self.factor * self.iqr_
        upper = self.Q3_ + self.factor * self.iqr_
        below = (lower - X) / (self.iqr_ + 1e-10)
        above = (X - upper) / (self.iqr_ + 1e-10)
        return np.max(np.maximum(0, np.maximum(below, above)), axis=1)
```

### 从头实现孤立森林

从头实现构建孤立树，随机划分特征空间：

```python
class IsolationTree:
    """通过随机分割构建的孤立树"""
    
    def __init__(self, height_limit=None):
        self.height_limit = height_limit
        self.root = None
    
    def _build_tree(self, X, depth=0):
        """递归构建树，直到所有点孤立或达到高度限制"""
        n_samples, n_features = X.shape
        
        # 停止条件
        if depth >= self.height_limit or n_samples <= 1:
            return {'size': n_samples, 'is_leaf': True}
        
        # 随机选择特征和分割值
        feature_idx = np.random.randint(n_features)
        feature_values = X[:, feature_idx]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return {'size': n_samples, 'is_leaf': True}
        
        split_value = np.random.uniform(min_val, max_val)
        
        # 分割数据
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {'size': n_samples, 'is_leaf': True}
        
        # 递归构建左右子树
        return {
            'feature': feature_idx,
            'split': split_value,
            'left': self._build_tree(X[left_mask], depth+1),
            'right': self._build_tree(X[right_mask], depth+1),
            'size': n_samples,
            'is_leaf': False
        }
    
    def _path_length(self, x, node, depth=0):
        """计算从根到叶子的路径长度"""
        if node['is_leaf']:
            return depth + _c(node['size'])  # 对叶节点大小进行校正
        if x[node['feature']] < node['split']:
            return self._path_length(x, node['left'], depth+1)
        else:
            return self._path_length(x, node['right'], depth+1)
    
    def fit(self, X):
        self.root = self._build_tree(X)
    
    def path_length(self, x):
        return self._path_length(x, self.root)

def _c(n):
    """给定样本量n，二叉搜索树中搜索失败的期望路径长度"""
    if n <= 1:
        return 0
    return 2 * (np.log(n-1) + 0.5772156649) - 2*(n-1)/n
```

孤立一个点的路径长度决定了它的异常分数。路径越短，越异常。

`IsolationForest`类包装了多棵树：

```python
class IsolationForest:
    """孤立森林异常检测器"""
    
    def __init__(self, n_estimators=100, max_samples=256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
        self._c = None
    
    def fit(self, X):
        n_samples = X.shape[0]
        sample_size = min(self.max_samples, n_samples)
        self._c = _c(sample_size)
        
        for _ in range(self.n_estimators):
            # 对数据采样子集
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = X[indices]
            
            # 构建孤立树
            tree = IsolationTree(height_limit=np.ceil(np.log2(sample_size)))
            tree.fit(X_sample)
            self.trees.append(tree)
        
        return self
    
    def score_samples(self, X):
        """计算每个样本的异常分数"""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for i, x in enumerate(X):
            path_lengths = [tree.path_length(x) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            scores[i] = 2 ** (-avg_path_length / self._c)
        
        return scores
    
    def predict(self, X, threshold=0.5):
        scores = self.score_samples(X)
        return scores > threshold
```

归一化因子`c(n)`是n个元素的二叉搜索树中搜索失败的期望路径长度。它等于`2 * H(n-1) - 2*(n-1)/n`，其中`H`是调和数。这种归一化确保了分数在不同大小的数据集之间具有可比性。

### 演示场景

代码生成了多个测试场景：

1. **单簇含异常值。** 一个二维高斯簇，在远离中心的地方注入异常。所有方法都应该有效。
2. **多峰数据。** 三个不同大小和密度的簇。簇之间的点是异常的。Z-score由于每个特征的范围较宽而失效。
3. **高维数据。** 50个特征，但异常仅在其中5个特征上不同。测试方法是否能在特征子集中找到异常。

每个演示使用精确率、召回率、F1和Precision@k比较所有方法。

## 使用它

使用sklearn（使用库实现，而非从头实现）：

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 孤立森林
iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X)  # 1表示正常，-1表示异常

# 局部异常因子
lof = LocalOutlierFactor(contamination=0.1)
y_pred = lof.fit_predict(X)
```

注意`contamination`设置了预期的异常比例。正确设置它很重要——太低会漏掉异常，太高会产生误报。

`anomaly_detection.py`中的代码将从头实现与sklearn在同一数据上进行了比较。

### sklearn的contamination参数

sklearn中的`contamination`参数决定了将连续异常分数转换为二元预测的阈值。它不会改变底层分数。

```python
iso_5 = IsolationForest(contamination=0.05).fit(X)
iso_10 = IsolationForest(contamination=0.10).fit(X)

# 分数相同，但预测不同
scores = iso_5.score_samples(X)  # 负值：越负越异常
preds_5 = iso_5.predict(X)       # 前5%为-1
preds_10 = iso_10.predict(X)    # 前10%为-1
```

两者产生相同的异常分数。但`iso_5`标记前5%，而`iso_10`标记前10%。如果你不知道真正的异常率（通常不知道），将contamination设置为"auto"并直接使用原始分数。根据误报和漏报之间的成本权衡设置你自己的阈值。

### 一类SVM

另一个值得了解的无监督异常检测器。一类SVM在高维特征空间中围绕正常数据拟合一个边界（使用核技巧）。

```python
from sklearn.svm import OneClassSVM

svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
y_pred = svm.fit_predict(X)  # 1表示正常，-1表示异常
```

`nu`参数近似表示异常的比例。一类SVM在中小型数据集上表现良好，但无法扩展到非常大的数据（核矩阵呈二次增长）。

### 自编码器方法（预览）

自编码器是一种神经网络，学习压缩和重建数据。在正常数据上进行训练。在测试时，异常具有较高的重建误差，因为网络只学会了重建正常模式。

这将在第三阶段（深度学习）中介绍，但原理相同：对正常数据建模，标记偏离的数据。

### 集成异常检测

正如集成方法改进分类（第11课）一样，组合多个异常检测器可以改进检测。最简单的方法：

1. 运行多个检测器（Z-score、IQR、孤立森林、LOF）
2. 将每个检测器的分数归一化到[0,1]
3. 对归一化分数取平均
4. 标记平均分数高于阈值的点

这减少了误报，因为不同方法有不同的失败模式。一个被所有四种方法标记的点几乎肯定是异常的。只被一种方法标记的点可能是该方法的偶然结果。

更复杂的集成会对每个检测器按其估计的可靠性（在已知异常的验证集上测量，如果有的话）进行加权。

### 生产注意事项

1. **阈值漂移。** 随着数据分布发生变化，固定阈值会过时。监控异常分数的分布并定期调整。
2. **警报疲劳。** 太多误报会导致操作员不再关注。从高阈值开始（更少但更可靠的警报），随着信任的建立逐步降低。
3. **集成方法。** 在生产中，组合多个检测器。只有当多个方法一致认为某个点异常时才标记它。这显著减少了误报。
4. **特征工程。** 原始特征通常不够。添加滚动统计量、比率、距上次事件的时间以及领域特定特征。好的特征集比检测器的选择更重要。
5. **反馈循环。** 当操作员调查标记项并确认或驳回它们时，将此信息反馈回系统。随时间累积标记数据以评估和改进检测器。

## 交付物

本课程产出：
- `outputs/skill-anomaly-detector.md` —— 选择合适检测器的决策技能
- `code/anomaly_detection.py` —— 从头实现的Z-score、IQR和孤立森林，并与sklearn比较

### 选择阈值

异常分数是连续的。你需要一个阈值来做出二元决策。这是一个业务决策，而非技术决策。

考虑两个场景：
- **欺诈检测。** 漏报欺诈代价高昂（退单、客户信任损失）。误报花费人工分析师5分钟进行调查。将阈值设置得低一些以捕获更多欺诈，接受更多误报。
- **设备维护。** 一次误报意味着一次不必要的停机，损失50,000美元。一次漏报失效意味着500,000美元的维修。设置阈值以平衡这些成本。

在这两种情况下，最优阈值取决于误报和漏报之间的成本比率。绘制不同阈值下的精确率和召回率，叠加成本函数，并选择成本最低的点。

### 扩展到生产

对于生产环境中的实时异常检测：

1. **批量训练，在线评分。** 定期（每日、每周）在最近的正常数据上训练模型。对每个新到达的观察值进行评分。
2. **特征计算必须一致。** 如果你使用30天的滚动统计量进行训练，你需要30天的历史数据来计算新观察值的特征。缓冲所需的历史数据。
3. **分数分布监控。** 跟踪异常分数随时间的变化。如果中位数分数上升，要么是数据在变化，要么是模型过时了。
4. **可解释性。** 当你标记一个异常时，说明原因。Z-score：“特征X高于正常值4.2个标准差。”孤立森林：“该点平均被3.1次分割孤立（正常点需要8.5次）。”

## 练习

1. **阈值调优。** 使用从1.0到5.0、步长为0.5的阈值运行Z-score检测器。在每个阈值下绘制精确率和召回率。你的数据的理想点在哪里？

2. **多变量异常。** 创建二维数据，其中每个特征单独看起来正常，但组合起来是异常的（例如，远离主簇对角线的点）。证明按特征的Z-score漏掉了这些异常，而孤立森林能捕获它们。

3. **从头实现LOF。** 使用k最近邻实现局部异常因子。与sklearn的LocalOutlierFactor在同一数据上进行比较。使用k=10和k=50 —— k的选择如何影响结果？

4. **流式异常检测。** 修改Z-score检测器以在流式设置中工作：随着新点的到达更新运行均值和方差（Welford在线算法）。与批处理Z-score在同一数据上进行比较。

5. **真实世界评估。** 取一个已知异常的数据集（例如，来自Kaggle的信用卡欺诈）。使用Precision@100、Precision@500和AUPRC评估所有四种方法。哪种方法效果最好？为什么？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| 异常（Anomaly） | “离群点，不寻常的点” | 显著偏离正常数据预期模式的数据点 |
| 点异常（Point anomaly） | “单个奇怪的值” | 无论上下文如何都不寻常的单个观测值 |
| 情境异常（Contextual anomaly） | “正常值，错误上下文” | 在其上下文（时间、位置等）中不寻常，但在另一个上下文中可能正常的观测值 |
| 孤立森林（Isolation Forest） | “随机分割找离群点” | 一组随机树，用比正常点更少的分割来孤立异常 |
| 局部异常因子（Local Outlier Factor） | “与邻居密度比较” | 一种方法，标记局部密度远低于邻居密度的点 |
| Z-score | “距离均值的标准差数” | (x - mean) / std，以标准差为单位测量点离中心有多远 |
| IQR | “四分位距” | Q3 - Q1，测量中间50%数据的散布，用于鲁棒的离群点检测 |
| Contamination | “预期异常比例” | 一个超参数，告诉检测器应该将多大比例的数据标记为异常 |
| Precision@k | “在排名前k的标记中，有多少是真实的” | 仅在最可疑的k个点上计算的精确率，对不平衡异常检测有用 |
| AUPRC | “精确率-召回率曲线下面积” | 一个指标，总结所有阈值下的精确率-召回率性能，对于不平衡数据优于AUROC |

## 进一步阅读

- [Liu et al., Isolation Forest (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) —— 原始孤立森林论文
- [Breunig et al., LOF: Identifying Density-Based Local Outliers (2000)](https://dl.acm.org/doi/10.1145/342009.335388) —— 原始LOF论文
- [scikit-learn Outlier Detection docs](https://scikit-learn.org/stable/modules/outlier_detection.html) —— 所有sklearn异常检测器的概述
- [Chandola et al., Anomaly Detection: A Survey (2009)](https://dl.acm.org/doi/10.1145/1541880.1541882) —— 异常检测方法的全面综述
- [Goldstein and Uchida, A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms (2016)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173) —— 在真实数据集上对10种方法的实证比较
