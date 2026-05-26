# 朴素贝叶斯

> “朴素”假设是错误的，但它仍然有效。这正是它的美妙之处。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段，第01-07课（分类，贝叶斯定理）
**时长：** ~75分钟

## 学习目标

- 从头实现带拉普拉斯平滑的多项式朴素贝叶斯，用于文本分类
- 解释为何朴素独立性假设在数学上是错误的，但在实践中却能产生正确的类别排名
- 比较多项式朴素贝叶斯、伯努利朴素贝叶斯和高斯朴素贝叶斯三种变体，并为给定特征类型选择合适的变体
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归的性能，说明其中的偏差-方差权衡

## 问题

你需要对文本进行分类。将电子邮件分为垃圾邮件或非垃圾邮件。将客户评论分为正面或负面。将支持工单分类。你有成千上万个特征（每个词一个）和有限的训练数据。

大多数分类器在这里都会失效。逻辑回归需要足够的样本才能可靠地估计成千上万个权重。决策树一次只在一个词上分裂，导致严重过拟合。在10000维空间中，KNN毫无意义，因为每个点与其他所有点的距离都一样远。

朴素贝叶斯能够处理这种情况。它做了一个数学上错误（假设在给定类别条件下，每个特征与其他每个特征独立）的假设，却仍然在文本分类上胜过“更聪明”的模型，尤其是在训练集较小的情况下。它只需一次遍历数据即可完成训练。它能扩展到数百万个特征。它能产生概率估计（尽管由于独立性假设，概率往往校准不佳）。

理解为什么一个错误的假设能带来好的预测，能教会你关于机器学习的基本道理：最好的模型不是最正确的模型，而是对你的数据具有最佳偏差-方差权衡的模型。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理翻转了条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(类别 | 特征)` —— 给定文档中的词，文档属于某个类别的概率。我们可以通过以下方式计算：
- `P(特征 | 类别)` —— 在该类别的文档中看到这些词的可能性
- `P(类别)` —— 类别的先验概率（垃圾邮件在总体中常见吗？）
- `P(特征)` —— 证据，对所有类别相同，因此在比较时可以忽略

具有最高 `P(类别 | 特征)` 的类别获胜。

### 朴素独立性假设

精确计算 `P(特征 | 类别)` 需要估计所有特征在一起的联合概率。对于10000个词的词汇表，你需要估计一个超过2^10000种可能组合的分布。不可能。

朴素假设：在给定类别的条件下，每个特征条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

不是一个不可能的联合分布，而是估计n个简单的每个特征分布。每个分布只需要一个计数。

这个假设显然是错误的。在任何文档中，单词“machine”和“learning”都不是独立的。但分类器不需要正确的概率估计。它需要正确的排名——哪个类别的概率最高。独立性假设引入了系统误差，但这些误差对所有类别的影响相似，因此排名保持正确。

### 它为何仍然有效

三个原因：

1. **排名优于校准。** 分类只需要排名第一的类别正确。即使当真实概率为0.7时，P(垃圾邮件) = 0.99999，分类器仍然正确选择了垃圾邮件。我们不需要正确的概率。我们需要正确的胜出者。

2. **高偏差，低方差。** 独立性假设是一个强先验。它严重约束了模型，从而防止过拟合。当训练数据有限时，一个略有错误但稳定的模型胜过一个理论上正确但极不稳定的模型。这就是偏差-方差权衡在起作用。

3. **特征冗余相互抵消。** 相关特征提供冗余的证据。分类器会重复计算这些证据，但它也会为正确的类别重复计算。如果“machine”和“learning”总是同时出现，两者都为“科技”类别提供证据。NB将它们计算了两次，但它是为正确的类别计算了两次。

第四个实际原因：朴素贝叶斯非常快。训练只需一次遍历数据统计频率。预测是一次矩阵乘法。你可以在几秒内训练百万个文档。这种速度意味着你可以更快地迭代，尝试更多特征集，并进行比慢模型更多的实验。

### 逐步数学推导

让我们通过一个具体例子来追踪。假设有两个类别：垃圾邮件和非垃圾邮件。词汇表有三个词：“free”、“money”、“meeting”。

训练数据：
- 垃圾邮件中提到“free”80次，“money”60次，“meeting”10次（共150个词）
- 非垃圾邮件中提到“free”5次，“money”10次，“meeting”100次（共115个词）
- 40%的邮件是垃圾邮件，60%是非垃圾邮件

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含：“free”（2次），“money”（1次），“meeting”（0次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以很大优势胜出。“free”出现两次是强有力的垃圾邮件证据。注意“meeting”未出现对两个对数之和贡献为零（0 * log(P)）——在多项式NB中，不存在的词没有影响。显式建模词缺失的是伯努利NB。

### 三种变体

朴素贝叶斯有三种形式。每种对 `P(特征 | 类别)` 的建模方式不同。

#### 多项式朴素贝叶斯

将每个特征建模为计数。最适合文本数据，其中特征是词频或TF-IDF值。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是拉普拉斯平滑（见下文解释）。这种变体是文本分类的主力。

#### 高斯朴素贝叶斯

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别都有每个特征自己的均值和方差。当特征在每个类别内确实遵循钟形曲线时效果良好。

#### 伯努利朴素贝叶斯

将每个特征建模为二值（存在或不存在）。最适合短文本或二值特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式不同，伯努利明确惩罚词的缺失。如果“free”通常出现在垃圾邮件中但该邮件中没有，伯努利将其算作反对垃圾邮件的证据。

### 何时使用每种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| 多项式 | 计数或频率 | 文本分类，词袋模型 | 邮件垃圾过滤，主题分类 |
| 高斯 | 连续值 | 具有近似正态特征的表格数据 | Iris分类，传感器数据 |
| 伯努利 | 二值（0/1） | 短文本，二值特征向量 | 短信垃圾过滤，存在/不存在特征 |

### 拉普拉斯平滑

当测试数据中出现一个词，但训练数据中针对某个类别从未出现过这个词时，会发生什么？

没有平滑：`P(词 | 类别) = 0/N = 0`。一个零乘遍整个乘积使得 `P(类别 | 特征) = 0`，无论其他所有证据如何。单个未见过的词破坏了整个预测，无论其他证据多么支持它。

拉普拉斯平滑给每个特征计数添加一个小计数 `alpha`（通常为1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当alpha=1时，每个词至少得到一个微小的概率。测试邮件中出现“discombobulate”这个词不再会摧毁垃圾邮件概率。平滑具有贝叶斯解释：相当于在词分布上放置一个均匀的Dirichlet先验。

较高的alpha意味着更强的平滑（更均匀的分布）。较低的alpha意味着模型更信任数据。Alpha是一个需要调整的超参数。

Alpha的效果：

| Alpha | 效果 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，信任数据 | 非常大的训练集，预期无未见特征 |
| 0.1 | 轻微平滑 | 大型训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，使分布扁平 | 非常小的训练集，预期有许多未见特征 |

### 对数空间计算

将数百个概率（每个小于1）相乘会导致浮点数下溢。乘积在浮点数中变为零，即使真实值是一个非常小的正数。

解决方案：在对数空间中工作。不乘概率，而是加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这将预测转化为点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快的原因——它与单层线性模型的操作相同。

### 朴素贝叶斯 vs 逻辑回归

两者都是用于文本的线性分类器。区别在于它们建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模P(X\|Y)） | 判别式（建模P(Y\|X)） |
| 训练 | 计数频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不足以估计权重） |
| 大数据 | 更差（错误假设受损） | 更好（灵活决策边界） |
| 特征 | 假设独立 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率不佳 | 概率较好 |

经验法则：从朴素贝叶斯开始。如果数据足够且NB进入平台期，则切换到逻辑回归。

### 分类流程

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们通常在对数空间中工作以避免浮点下溢。不乘许多小的概率，而是加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 动手构建

`code/naive_bayes.py` 中的代码从头实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从头实现：

1. **fit(X, y)**：对于每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类频率的对数）。

2. **predict_log_proba(X)**：对于每个样本，对所有类别计算 log P(类别) + sum(log P(特征_i | 类别))。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回具有最高对数概率的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：训练完成后，预测只是矩阵乘法加上一个偏置。这就是朴素贝叶斯如此快的原因。

### GaussianNB

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯概率密度函数（PDF），然后在特征间相乘（在对数空间中相加）。

### 演示：文本分类

代码生成合成词袋数据，模拟两个类别（科技文章 vs 体育文章）。每个类别具有不同的词频分布。MultinomialNB 使用词频对它们进行分类。

合成数据的工作方式如下：我们创建200个“词”（特征列）。词0-39在科技文章中高频出现，在体育文章中低频出现。词80-119在体育文章中高频出现，在科技文章中低频出现。词40-79在两个类别中都中等频率出现。这创建了一个现实的场景：一些词是强类别指标，其他是噪声。

### 演示：连续特征

代码生成类似鸢尾花的数据（3个类别，4个特征，高斯簇）。GaussianNB 使用每个类别的均值和方差进行分类。每个类别具有不同的中心（均值向量）和不同的散布（方差），模拟真实数据中测量结果在不同类别之间系统性不同的情况。

代码还演示了：
- **平滑比较：** 使用不同alpha值训练MultinomialNB，展示平滑强度对准确率的影响。
- **训练规模实验：** 随着训练数据从20个样本增加到1600个样本，NB准确率如何提高。即使样本非常少，NB也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵：** 每个类别的精确率、召回率和F1分数，展示NB在哪些地方犯错。

### 预测速度

朴素贝叶斯预测是一种矩阵乘法。对于n个样本，d个特征，k个类别：
- MultinomialNB：一次矩阵乘 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯PDF计算，每次跨d个特征 = O(n * d * k)

两者在每个维度上都是线性的。相比之下，KNN（需要到所有训练点的距离计算）或带RBF核的SVM（需要对所有支持向量进行核评估）则慢得多。NB在预测时间上快数个数量级。

## 使用它

用sklearn，两种变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用sklearn进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码在相同数据上比较了从头实现与sklearn的实现，以验证正确性。

### TF-IDF与朴素贝叶斯

原始词频给每个词每次出现相等的权重。但像“the”和“is”这样的常见词在每个类别中都频繁出现——它们不携带信息。TF-IDF（词频-逆文档频率）降低常见词的权重，提高稀有、有区分力词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF值是非负的，因此它们可以与MultinomialNB一起使用。TF-IDF + MultinomialNB的组合是文本分类最强大的基线之一。在训练样本少于10,000的数据集上，它常常胜过更复杂的模型。

### 用于短文本的BernoulliNB

对于短文本（推文、短信、聊天消息），BernoulliNB可以优于MultinomialNB。短文本的词频很低，因此MultinomialNB依赖的频率信息噪声很大。BernoulliNB只关心存在与否，在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

`CountVectorizer` 中的 `binary=True` 标志将所有计数转换为0/1。没有它，BernoulliNB仍然可以工作，但看到的是它并非设计用于的计数。

### 校准NB概率

NB的概率校准不佳。当NB说P(垃圾邮件) = 0.95时，真实概率可能只有0.7。如果你需要可靠的概率估计（例如，设置阈值或与其他模型结合），请使用sklearn的CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这会使用交叉验证在NB的原始分数之上拟合一个逻辑回归。得到的概率更接近真实的类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB需要非负特征。如果你有负值（比如某些设置下的TF-IDF或标准化特征），请改用GaussianNB，或将特征平移为正。

2. **零方差特征。** GaussianNB除以方差。如果某个特征对于某个类别方差为零（所有值相同），概率计算会出错。代码在所有方差上添加了一个小的平滑项（1e-9）以防止这种情况。

3. **类别不平衡。** 如果99%的邮件不是垃圾邮件，先验P(非垃圾邮件) = 0.99 如此之强，以至于压倒了似然证据。你可以手动设置类别先验，或在sklearn中使用class_prior参数。

4. **特征缩放。** MultinomialNB不需要缩放（它基于计数工作）。GaussianNB也不需要缩放（它估计每个特征的统计量）。这是相比逻辑回归和SVM的一个优势，后两者对特征尺度敏感。

## 交付成果

本课程产生：
- `outputs/skill-naive-bayes-chooser.md` —— 一个用于选择正确NB变体的决策技能
- `code/naive_bayes.py` —— 从头实现的MultinomialNB和GaussianNB，附有sklearn比较

### 朴素贝叶斯何时失败

NB在独立性假设导致不正确排名（而不仅仅是概率不正确）时失败。这发生在：

1. **强特征交互。** 如果类别取决于两个特征的组合，但单独任何一个特征都不起作用（类似XOR的模式），NB会完全错过。每个特征单独不提供证据，而NB无法非线性地组合它们。

2. **具有相反证据的高度相关特征。** 如果特征A表示“垃圾邮件”，特征B表示“非垃圾邮件”，但A和B完全相关（它们在现实中总是一致），NB会看到矛盾证据，而实际上没有。

3. **非常大的训练集。** 有了足够的数据，判别式模型如逻辑回归可以学习真实的决策边界，并超越NB。在数据量少时有所帮助的独立性假设现在反而限制了模型。

在实践中，对于文本分类，这些失败模式很少见。文本特征数量多、单独作用弱，且独立性假设的误差往往相互抵消。对于具有少量强相关特征的表格数据，请首先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 在文本数据上以alpha值为0.01、0.1、1.0、10.0和100.0训练MultinomialNB。绘制准确率对alpha的曲线。性能峰值在哪？为什么非常高的alpha会损害性能？

2. **特征独立性测试。** 选取一个真实的文本数据集。选择两个明显相关的词（“machine”和“learning”）。计算P(词1 | 类别) * P(词2 | 类别)并与P(词1 AND 词2 | 类别)比较。独立性假设有多错误？它是否影响分类准确率？

3. **伯努利实现。** 扩展代码，添加BernoulliNB类。将词袋转换为二值（存在/不存在），并在文本数据上与MultinomialNB比较准确率。伯努利何时胜出？

4. **NB vs 逻辑回归。** 在文本数据上训练两者。从100个训练样本开始，逐步增加到10,000个。绘制准确率对训练集大小的曲线。逻辑回归在哪个点上超越朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本进行分词，构建词汇表，创建词袋特征，训练MultinomialNB，使用精确率和召回率（不仅仅是准确率——为什么？）进行评估。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯 | “简单的概率分类器” | 应用贝叶斯定理，假设在给定类别条件下特征条件独立的分类器 |
| 条件独立 | “特征之间互不影响” | P(A, B \| C) = P(A \| C) * P(B \| C) —— 一旦知道C，知道B并不能告诉你关于A的新信息 |
| 拉普拉斯平滑 | “加一平滑” | 为每个特征添加一个小计数，以防止零概率主导预测 |
| 先验 | “你在看到数据之前相信的” | P(类别) —— 在观察任何特征之前每个类别的概率 |
| 似然 | “数据拟合得如何” | P(特征 \| 类别) —— 如果已知类别，观察到这些特征的概率 |
| 后验 | “看到数据后你相信的” | P(类别 \| 特征) —— 观察到特征后更新过的类别概率 |
| 生成模型 | “建模数据如何生成” | 学习P(X \| Y) 和 P(Y)，然后使用贝叶斯定理得到P(Y \| X)的模型 |
| 判别模型 | “建模决策边界” | 直接学习P(Y \| X)而不建模X如何生成的模型 |
| 对数概率 | “避免下溢” | 使用log P而不是P，以防止多个小数的乘积在浮点数中变为零 |

## 扩展阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) —— 所有三种变体的数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) —— 多项式vs伯努利用于文本的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) —— NB用于文本的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) —— 证明了NB比LR用更少数据收敛更快
