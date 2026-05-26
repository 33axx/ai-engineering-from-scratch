# 采样方法

> 采样是AI探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段1，第06–07课（概率，贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- 仅使用均匀随机数从头实现逆CDF采样、拒绝采样和重要性采样
- 为语言模型词元生成构建温度采样、top‑k采样和top‑p（核）采样
- 解释重参数化技巧，以及它为何能实现在VAE中通过采样进行反向传播
- 运行Metropolis‑Hastings MCMC，从未归一化的目标分布中采样

## 问题

语言模型处理完你的提示后，会生成一个包含50,000个logits的向量。每个logits对应词汇表中的一个词元。现在它必须选出一个。怎么选？

如果它总是选择概率最高的词元，那么每次回答都相同。确定性。乏味。如果它均匀随机选择，输出就是胡言乱语。答案存在于这两个极端之间，而这个“之间”是由采样控制的。

采样并不局限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE通过从学习到的分布中采样并通过随机性进行反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡罗方法估计没有闭合形式的积分。MCMC算法探索不可能枚举的高维后验分布。

每个生成式AI系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本节课将从均匀随机数开始，逐步构建每一种主要的采样方法，最终涵盖驱动现代LLM和生成模型的技术。

## 概念

### 为什么采样很重要

采样在AI和机器学习中扮演四种基本角色：

**生成。** 语言模型、扩散模型和GAN都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top‑k和核采样是工程师每天调节的旋钮。

**训练。** 随机梯度下降对小批量进行采样。Dropout随机停用神经元。数据增强随机应用变换。重要性通过对样本重新加权来降低强化学习（PPO、TRPO）中的梯度方差。

**估计。** 机器学习中的许多量没有闭合解。数据分布上的期望损失、能量基模型的配分函数、贝叶斯推断中的证据。蒙特卡罗估计通过对样本取平均来近似所有这些量。

**探索。** MCMC算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。汤普森采样在强盗问题中平衡探索与利用。

核心挑战是：你只能直接从简单分布（均匀、正态）中采样。对于其他所有情况，你需要一种方法将简单样本转换为目标分布的样本。

### 均匀随机采样

每种采样方法都从这里开始。均匀随机数生成器产生[0, 1)范围内的值，其中每个等长子区间具有相等的概率。

```python
import random
import numpy as np
import matplotlib.pyplot as plt

# 基本的均匀随机数
u = random.random()  # 均匀[0,1)
```

要从n个离散项的集合中均匀采样，生成U并返回floor(n * U)。要从连续区间[a, b]中采样，计算 a + (b - a) * U。

关键见解是：单个均匀随机数恰好包含足够的随机性来从任何分布中产生一个样本。诀窍在于找到正确的变换。

### 逆CDF方法（逆变换采样）

累积分布函数（CDF）将值映射到概率：

```python
# 指数分布 CDF: F(x) = 1 - exp(-lambda * x), x >= 0
# 对于lambda=0.5, 在x=1.0处
lam = 0.5
cdf_value = 1 - np.exp(-lam * 1.0)
print(f"CDF(1.0) = {cdf_value:.3f}")  # 0.393
```

逆CDF将概率映射回值。如果 U ~ Uniform(0, 1)，则 X = F_inverse(U) 服从目标分布。

```python
def inverse_cdf_exponential(u, lam=1.0):
    """指数分布的逆CDF: F_inverse(u) = -ln(1 - u) / lambda"""
    return -np.log(1 - u) / lam

# 生成10000个指数样本
u_samples = np.random.uniform(0, 1, 10000)
x_samples = inverse_cdf_exponential(u_samples, lam=0.5)
```

**指数分布示例：**

```python
lam = 0.5
x = np.linspace(0, 10, 1000)
cdf = 1 - np.exp(-lam * x)
inv_cdf = -np.log(1 - np.random.uniform(0, 1, 10000)) / lam
```

当你能够写出F_inverse的闭合形式时，这种方法完美工作。对于正态分布，没有闭合形式的逆CDF，因此我们使用其他方法（Box‑Muller或数值近似）。

**离散版本：** 对于离散分布，将CDF构建为累积和，生成U，然后找到累积和超过U的第一个索引。这就是第06课中`sample_categorical`的工作原理。

### 拒绝采样

当你无法求逆CDF，但可以评估目标PDF（至多一个常数因子）时，拒绝采样就能工作。

```python
def rejection_sampling(target_pdf, proposal_pdf, proposal_sampler, M, n_samples):
    """
    target_pdf: 目标分布（至多一个常数）
    proposal_pdf: 提议分布
    proposal_sampler: 从提议分布中采样的函数
    M: 包围常数，使得 target_pdf(x) <= M * proposal_pdf(x) 对所有x成立
    """
    samples = []
    while len(samples) < n_samples:
        x = proposal_sampler()
        u = np.random.uniform(0, 1)
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
    return np.array(samples)
```

边界M越紧，接受率越高。在低维度（1‑3）中，拒绝采样效果很好。在高维度中，接受率呈指数下降，因为大部分提议体积会被拒绝。这就是拒绝采样的维度灾难。

**示例：从截断正态分布中采样。** 在截断范围内使用均匀提议。包络M是该范围内正态PDF的最大值。

**示例：从半圆中采样。** 在边界矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡罗计算pi的方式：接受率等于面积比 pi/4。

### 重要性采样

有时你不需要来自目标分布 p(x) 的样本。你需要估计在 p(x) 下的期望，但你拥有来自不同分布 q(x) 的样本。

```python
def importance_sampling(x_samples, target_pdf, proposal_pdf):
    """
    使用来自q的样本来估计E_p[f(x)]
    x_samples: 来自 q(x) 的样本
    """
    weights = target_pdf(x_samples) / proposal_pdf(x_samples)
    # 对f(x)=x的情况，估计E_p[x]
    estimate = np.mean(x_samples * weights)
    return estimate, weights
```

这在强化学习中至关重要。在PPO（近端策略优化）中，你在旧策略 pi_old 下收集轨迹，但想要优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO会裁剪这些权重，以防止新策略偏离旧策略过远。

重要性采样估计器的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少数样本会获得巨大权重并主导估计。自归一化重要性采样通过除以权重之和来缓解这个问题：

```python
def self_normalized_importance_sampling(x_samples, target_pdf, proposal_pdf):
    weights = target_pdf(x_samples) / proposal_pdf(x_samples)
    normalized_weights = weights / np.sum(weights)
    estimate = np.sum(x_samples * normalized_weights)
    return estimate
```

### 蒙特卡罗估计

蒙特卡罗估计通过对随机样本取平均来近似积分。大数定律保证了收敛性。

```python
def monte_carlo_estimate(f, sampler, n_samples):
    """使用蒙特卡罗估计E[f(X)]"""
    samples = sampler(n_samples)
    return np.mean(f(samples))
```

误差率与维度无关。这就是为什么蒙特卡罗方法在高维度中占主导地位，而基于网格的积分在高维度中不可行。

**估计pi：**

```python
# 在单位正方形[0,1]^2上均匀采样
n = 100000
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * np.sum(inside) / n
print(f"预计pi = {pi_estimate:.4f} (实际 = {np.pi:.4f})")
```

**估计期望：**

```python
# 估计E[sin(X)]，其中X ~ Uniform(0, pi)
n = 100000
x = np.random.uniform(0, np.pi, n)
estimate = np.mean(np.sin(x))
print(f"E[sin(X)] ≈ {estimate:.4f} (实际 = 2/pi ≈ {2/np.pi:.4f})")
```

### 马尔可夫链蒙特卡罗（MCMC）：Metropolis‑Hastings

MCMC构造一个马尔可夫链，其平稳分布是目标分布 p(x)。经过足够的步骤后，链中的样本（近似）是 p(x) 的样本。

```python
def metropolis_hastings(target_pdf, initial_x, proposal_std, n_samples):
    x = initial_x
    samples = [x]
    accept = 0
    
    for _ in range(n_samples):
        # 从对称提议分布（正态）中提出新状态
        x_proposal = x + np.random.normal(0, proposal_std)
        
        # 计算接受概率
        alpha = target_pdf(x_proposal) / (target_pdf(x) + 1e-12)
        
        if np.random.uniform(0, 1) < alpha:
            x = x_proposal
            accept += 1
            
        samples.append(x)
    
    return np.array(samples), accept / n_samples
```

对于对称提议分布（q(x'|x) = q(x|x')），比值简化为 p(x')/p(x)。这就是原始的Metropolis算法。

**为什么它有效。** 接受规则保证了细致平衡：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。

**实践中的考虑：**
- 预烧期（Burn‑in）：丢弃链达到平衡前的早期样本
- 稀疏化（Thinning）：每 k 个样本保留一个，以减少自相关
- 提议尺度：太小时链移动缓慢（高接受率，慢探索）；太大时大多数提议被拒绝（低接受率，停滞在原地）
- 在高维空间中，高斯提议的最优接受率约为0.234

### Gibbs采样

Gibbs采样是多变量分布中MCMC的一种特例。它不一次性在所有维度上提议移动，而是每次从其条件分布中更新一个变量。

```python
def gibbs_sampling(conditional_samplers, initial_state, n_samples):
    """
    conditional_samplers: 每个变量的条件采样函数列表
    conditional_samplers[i](state) 返回从 p(x_i | x_{-i}) 中采样的新x_i
    """
    state = np.array(initial_state)
    samples = [state.copy()]
    
    for _ in range(n_samples):
        for i in range(len(state)):
            # 从条件分布中重新采样x_i
            state[i] = conditional_samplers[i](state)
        samples.append(state.copy())
    
    return np.array(samples)
```

Gibbs采样要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。这对于许多模型来说很直接：
- 贝叶斯网络：条件分布遵循图结构
- 高斯混合：条件分布是高斯分布
- Ising模型：每个自旋的条件分布仅依赖于其邻居

接受率始终为1（每个提议都被接受），因为从精确条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs采样混合缓慢，因为一次只更新一个变量无法在分布中做出大的对角移动。

### 温度采样（用于LLM）

语言模型输出每个词元的logits z_1, ..., z_V。Softmax将这些转换为概率。温度在softmax之前对logits进行重新缩放：

```python
def temperature_sampling(logits, temperature=1.0):
    """使用温度T进行采样"""
    if temperature == 0.0:
        # 贪心解码：选择概率最高的词元
        return np.argmax(logits)
    
    # 应用温度
    scaled_logits = logits / temperature
    
    # softmax得到概率
    # 数值稳定版本: 减去最大值
    scaled_logits = scaled_logits - np.max(scaled_logits)
    probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
    
    # 从概率分布中采样
    return np.random.choice(len(logits), p=probs)
```

**为什么它有效。** 除以 T < 1 会放大logits之间的差异。如果 z_1 = 2, z_2 = 1，除以 T = 0.5 得到 z_1/T = 4, z_2/T = 2，使差距变大。经过softmax后，最高logits的词元获得更大的份额。

**实践中：**
- T = 0.0：贪心解码，最适合事实性问答
- T = 0.3‑0.7：略带创造性，适合代码生成
- T = 0.7‑1.0：平衡，适合一般对话
- T = 1.0‑1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不会改变哪些词元是可能的。它改变分配给每个词元的概率质量。

### Top‑k采样

Top‑k采样将候选集限制为概率最高的k个词元，然后重新归一化并从该限制集中采样。

```python
def top_k_sampling(logits, k=40):
    """从top‑k个logits中采样"""
    # 找到第k大的logits
    indices = np.argsort(logits)[-k:]
    top_k_logits = np.take(logits, indices)
    
    # 只保留top‑k，其余设为负无穷
    masked_logits = np.full_like(logits, -np.inf)
    masked_logits[indices] = top_k_logits
    
    # softmax得到概率
    masked_logits = masked_logits - np.max(masked_logits)
    probs = np.exp(masked_logits) / np.sum(np.exp(masked_logits))
    
    # 采样
    return np.random.choice(len(logits), p=probs)
```

Top‑k防止模型选择词汇分布长尾中极不可能的词元（拼写错误、无意义）。问题是：k是固定的，与上下文无关。当模型很自信时（一个词元有95%概率），k=40仍然允许39个备选项。当模型不确定时（概率分布在1000个词元上），k=40会截断合理的选项。

### Top‑p（核）采样

Top‑p采样动态调整候选集大小。它不保留固定数量的词元，而是保留累积概率超过p的最小词元集。

```python
def top_p_sampling(logits, p=0.9):
    """从核（top‑p）中采样"""
    # 按概率降序排序
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # softmax
    sorted_logits = sorted_logits - np.max(sorted_logits)
    sorted_probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    
    # 找到累积概率超过p的最小集
    cumulative_probs = np.cumsum(sorted_probs)
    mask = cumulative_probs <= p
    # 至少保留一个
    if not np.any(mask):
        mask[0] = True
    
    # 创建掩码
    selected_indices = sorted_indices[mask]
    
    # 只保留选中的logits
    masked_logits = np.full_like(logits, -np.inf)
    masked_logits[selected_indices] = logits[selected_indices]
    
    # softmax得到概率并采样
    masked_logits = masked_logits - np.max(masked_logits)
    probs = np.exp(masked_logits) / np.sum(np.exp(masked_logits))
    
    return np.random.choice(len(logits), p=probs)
```

当模型很自信时，核采样保留很少的词元（也许2‑3个）。当模型不确定时，它保留很多（也许200个）。这种自适应行为是核采样通常比top‑k产生更好文本的原因。

**常见组合：**
- 温度0.7 + top‑p 0.9：良好的通用设置
- 温度0.0（贪心）：最适合确定性任务
- 温度1.0 + top‑k 50：Fan等人（2018）原始论文中的设置

Top‑k和top‑p可以结合使用。先应用top‑k，然后在保留的集合上应用top‑p。

### 重参数化技巧（用于VAE）

变分自编码器（VAE）通过将输入编码为潜在空间中的分布、从该分布中采样并将样本解码回来进行学习。问题：你无法通过采样操作进行反向传播。

```python
# 直接采样（不可微分）
z = np.random.normal(mu, sigma)
```

重参数化技巧将随机性与参数分开：

```python
# 重参数化采样（可微分）
epsilon = np.random.normal(0, 1)  # 从固定分布中采样
z = mu + sigma * epsilon  # 可微分的变换
```

这是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同分布。关键见解：将随机性移到无参数源（epsilon），然后将样本表示为参数的可微函数。

**在VAE训练循环中：**
1. 编码器对每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤4、3、2、1进行反向传播（因为步骤3是可微的，所以可能）

如果没有重参数化技巧，VAE无法用标准反向传播进行训练。这个单一的见解使VAE变得实用。

### Gumbel‑Softmax（可微分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散分类分布，我们需要不同的方法。Gumbel‑Softmax提供了分类采样的可微近似。

**Gumbel‑Max技巧（不可微）：**

```python
def gumbel_max_sample(logits):
    """Gumbel-Max技巧: 从分类分布中采样"""
    # 从Gumbel(0,1)中采样并加到logits上
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(logits))))
    return np.argmax(logits + gumbel_noise)
```

**Gumbel‑Softmax（可微近似）：**

```python
def gumbel_softmax(logits, temperature=1.0):
    """Gumbel-Softmax: 分类采样的连续可微近似"""
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(logits))))
    y = logits + gumbel_noise
    # 带温度的softmax
    y = y / temperature
    y = y - np.max(y)
    return np.exp(y) / np.sum(np.exp(y))
```

Gumbel‑Softmax产生离散样本的连续松弛。输出是一个概率向量（软独热）而不是硬独热。梯度通过softmax流动。在训练的前向传播中，你可以使用“直通”估计器：前向传播使用硬argmax，但反向传播使用软Gumbel‑Softmax梯度。

**应用：**
- VAE中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 使用离散动作的强化学习

### 分层采样

标准的蒙特卡罗采样可能会偶然在样本空间中留下空隙。分层采样通过将空间划分为层（strata）并从每层中采样来强制均匀覆盖。

```python
def stratified_sampling(n_strata=10, n_per_stratum=1):
    """
    在[0,1]上分层采样
    将[0,1]划分为n_strata层，每层内均匀采样n_per_stratum个点
    """
    samples = []
    stratum_size = 1.0 / n_strata
    for i in range(n_strata):
        low = i * stratum_size
        high = (i + 1) * stratum_size
        for _ in range(n_per_stratum):
            samples.append(np.random.uniform(low, high))
    return np.array(samples)
```

分层采样总是具有比标准蒙特卡罗更低或相等的方差：

```python
# 分层采样（较低方差）
n_per_layer = 100
layers = 10
stratified_samples = stratified_sampling(layers, n_per_layer)

# 蒙特色卡罗（可能较高方差）
monte_carlo_samples = np.random.uniform(0, 1, layers * n_per_layer)
```

**应用：**
- 数值积分（拟蒙特卡罗）
- 训练数据划分（确保每折中的类别平衡）
- 带分层的重要性采样（结合两种技术）
- NeRF（神经辐射场）沿相机光线使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在T步内逐步向图像添加高斯噪声，直到变成纯噪声。逆向过程学习去噪，逐步恢复原始图像。

```python
def forward_diffusion(x_0, noise_schedule, T):
    """向前扩散过程: 逐步添加噪声"""
    x_t = x_0
    for t in range(T):
        alpha_t = noise_schedule[t]
        epsilon = np.random.normal(0, 1, x_0.shape)
        x_t = np.sqrt(alpha_t) * x_t + np.sqrt(1 - alpha_t) * epsilon
    return x_t

def reverse_diffusion_step(x_t, t, denoise_model, alpha_t):
    """单步逆向扩散: 预测噪声并移除"""
    # 从模型中预测噪声
    predicted_noise = denoise_model(x_t, t)
    # 从预测的去噪分布中采样
    z = np.random.normal(0, 1, x_t.shape) if t > 0 else 0
    x_t_minus_1 = (1 / np.sqrt(alpha_t)) * (
        x_t - (1 - alpha_t) / np.sqrt(1 - np.cumprod(alpha_t)[t]) * predicted_noise
    ) + np.sqrt(1 - alpha_t) * z
    return x_t_minus_1
```

与本节课方法的联系：
- 每个去噪步骤使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制一种温度退火形式
- 训练使用蒙特卡罗估计来近似ELBO（证据下界）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步只依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，在每一步，采样一个略少噪音的版本，其条件是基于学习到的去噪模型。

## 动手构建

### 步骤1：均匀采样和逆CDF采样

```python
import numpy as np
import matplotlib.pyplot as plt

# 逆CDF采样：指数分布
def inverse_cdf_exponential(u, lam=1.0):
    return -np.log(1 - u) / lam

# 生成样本
n = 10000
u = np.random.uniform(0, 1, n)
samples = inverse_cdf_exponential(u, lam=0.5)

print(f"样本均值: {np.mean(samples):.3f} (理论: {1/0.5:.3f})")
print(f"样本标准差: {np.std(samples):.3f} (理论: {1/0.5:.3f})")

# 可视化
x = np.linspace(0, 15, 1000)
pdf = 0.5 * np.exp(-0.5 * x)
plt.hist(samples, bins=50, density=True, alpha=0.6, label='样本')
plt.plot(x, pdf, 'r-', label='理论PDF')
plt.legend()
plt.show()
```

生成10,000个指数样本，并验证均值为1/lambda。

### 步骤2：拒绝采样

```python
# 目标：截断正态分布 N(0, 1) 截断到 [-2, 2]
def truncated_normal_pdf(x, mu=0, sigma=1, a=-2, b=2):
    from scipy.stats import norm
    if x < a or x > b:
        return 0.0
    return norm.pdf(x, mu, sigma) / (norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma))

# 提议：在 [-2, 2] 上的均匀分布
def uniform_proposal():
    return np.random.uniform(-2, 2)

# 包络常数 M：截断正态PDF在[-2,2]上的最大值
# 对于mu=0, sigma=1，最大值在x=0处
from scipy.stats import norm
a = -2; b = 2
M = norm.pdf(0) / (norm.cdf(b) - norm.cdf(a))  # ~1.0 / 0.9545 ≈ 1.047

def rejection_sample(n, target_pdf, proposal_sampler, M):
    samples = []
    while len(samples) < n:
        x = proposal_sampler()
        u = np.random.uniform(0, 1)
        if u < target_pdf(x) / M:  # 因为均匀提议的PDF在区间上是1/(b-a)=1/4，但我们对它进行了缩放
            # 实际上我们需要 target_pdf(x) / (M * proposal_pdf(x))，proposal_pdf=1/(b-a)=1/4
            # 所以条件应该是 u < target_pdf(x) / (M * (1/4)) = 4*target_pdf(x)/M
            pass
    return np.array(samples)
```

使用拒绝采样从截断正态分布中采样。通过直方图验证形状。

### 步骤3：重要性采样

```python
# 估计 E[X^2]，其中 X ~ N(0,1)，使用均匀提议 U[-3,3]
target_pdf = lambda x: (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2)
proposal_pdf = lambda x: 1/(6) if -3 <= x <= 3 else 0  # 均匀[－3,3]的PDF是1/6

n = 100000
x_samples = np.random.uniform(-3, 3, n)
weights = target_pdf(x_samples) / proposal_pdf(x_samples)
estimate = np.mean(x_samples**2 * weights)
actual = 1.0  # N(0,1)的方差是1，E[X^2]=1
print(f"估计 E[X^2] = {estimate:.4f} (实际 = {actual})")
```

使用均匀提议估计在正态分布下的E[X^2]。与已知答案（mu^2 + sigma^2）进行比较。

### 步骤4：蒙特卡罗估计pi

```python
n_points = [100, 1000, 10000, 100000]
for n in n_points:
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    inside = (x**2 + y**2) <= 1
    pi_est = 4 * np.sum(inside) / n
    error = abs(pi_est - np.pi)
    print(f"n={n:6d}: pi≈{pi_est:.4f}, 误差={error:.4f}")
```

### 步骤5：Metropolis‑Hastings MCMC

```python
# 目标：双峰分布（两个高斯混合）
def bimodal_pdf(x):
    return 0.3 * np.exp(-0.5 * ((x - 3)/0.8)**2) + 0.7 * np.exp(-0.5 * ((x + 2)/1.0)**2)

# Metropolis-Hastings
def mh_sampler(target, initial, proposal_std, n):
    x = initial
    samples = [x]
    accepts = 0
    for _ in range(n):
        x_prop = x + np.random.normal(0, proposal_std)
        alpha = target(x_prop) / (target(x) + 1e-12)
        if np.random.uniform() < alpha:
            x = x_prop
            accepts += 1
        samples.append(x)
    return np.array(samples), accepts / (n+1)

samples, acc_rate = mh_sampler(bimodal_pdf, initial=0.0, proposal_std=1.0, n=20000)
print(f"接受率: {acc_rate:.3f}")

# 可视化（丢弃前1000个预烧样本）
burnin = 1000
plt.hist(samples[burnin:], bins=50, density=True, alpha=0.6)
x = np.linspace(-6, 6, 1000)
plt.plot(x, bimodal_pdf(x) / np.trapz(bimodal_pdf(x), x), 'r-')
plt.show()
```

从双峰分布（两个高斯的混合）中采样。可视化链的轨迹。

### 步骤6：Gibbs采样

```python
# 二维高斯，均值[0,0]，协方差 [[1, 0.8], [0.8, 1]]
# 条件分布：p(x1|x2) ~ N(0.8*x2, 1-0.8^2) = N(0.8*x2, 0.36)

def gibbs_2d_gaussian(n_samples, initial_state=(0, 0)):
    x, y = initial_state
    samples = []
    for _ in range(n_samples):
        # 更新x：给定y，x的条件分布
        x = np.random.normal(0.8 * y, np.sqrt(0.36))
        # 更新y：给定x，y的条件分布
        y = np.random.normal(0.8 * x, np.sqrt(0.36))
        samples.append([x, y])
    return np.array(samples)

samples = gibbs_2d_gaussian(5000)
# 可视化二维散点图
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3)
plt.axis('equal')
plt.show()
```

### 步骤7：温度采样

```python
logits = np.array([2.0, 1.5, 0.5, 0.0, -0.5, -1.0, -2.0])
temperatures = [0.0, 0.5, 1.0, 2.0]

for T in temperatures:
    if T == 0.0:
        probs = np.zeros_like(logits)
        idx = np.argmax(logits)
        probs[idx] = 1.0
        print(f"T={T}: {probs.round(3)}")
    else:
        scaled = logits / T
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        print(f"T={T}: {probs.round(3)}")
```

展示温度如何改变一组词元logits的输出分布。

### 步骤8：Top‑k和top‑p采样

```python
logits = np.random.randn(10) * 2
print("Top-k:")
for k in [1, 3, 5]:
    print(f"  k={k}: sample {top_k_sampling(logits, k)}")

print("Top-p:")
for p in [0.5, 0.9, 1.0]:
    print(f"  p={p}: sample {top_p_sampling(logits, p)}")
```

### 步骤9：重参数化技巧

```python
import torch
import torch.nn as nn

# 演示梯度通过重参数化样本流动
mu = torch.tensor([0.5], requires_grad=True)
log_sigma = torch.tensor([0.1], requires_grad=True)

# 直接采样（不可微分）
# z_direct = torch.normal(mu, torch.exp(log_sigma))  # 梯度不会流动

# 重参数化采样（可微分）
epsilon = torch.randn_like(mu)
z = mu + torch.exp(log_sigma) * epsilon

# 简单的损失函数
loss = (z - 1.0) ** 2
loss.backward()

print(f"mu的梯度: {mu.grad.item():.4f}")
print(f"log_sigma的梯度: {log_sigma.grad.item():.4f}")
```

演示梯度流过重参数化样本，但不流过直接采样。

### 步骤10：Gumbel‑Softmax

```python
logits = np.array([2.0, 1.0, 0.5, 0.1])
temperatures = [0.1, 0.5, 1.0, 5.0]

for T in temperatures:
    soft_sample = gumbel_softmax(logits, temperature=T)
    print(f"T={T}: {soft_sample.round(3)}")
```

展示降低温度如何使输出接近独热向量。

完整的实现及所有可视化见 `code/sampling.py`。

## 使用它

使用NumPy和SciPy的生产版本：

```python
from scipy.stats import norm, truncnorm
from scipy.special import log_softmax

# 分布采样
samples = norm.rvs(loc=0, scale=1, size=1000)  # 正态
samples = truncnorm.rvs(-2, 2, loc=0, scale=1, size=1000)  # 截断正态

# MCMC
# 使用PyMC, emcee, NumPyro等库

# 语言模型采样
def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    logits = np.array(logits)
    if temperature == 0:
        return np.argmax(logits)
    logits = logits / temperature
    if top_k is not None:
        # 只保留前k个
        indices = np.argsort(logits)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[indices] = logits[indices]
        logits = mask
    if top_p is not None:
        # 核采样
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs = sorted_probs / np.sum(sorted_probs)
        cumsum = np.cumsum(sorted_probs)
        mask = cumsum <= top_p
        if not np.any(mask):
            mask[0] = True
        selected = sorted_indices[mask]
        mask = np.full_like(logits, -np.inf)
        mask[selected] = logits[selected]
        logits = mask
    # softmax
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return np.random.choice(len(probs), p=probs)
```

对于大规模MCMC，使用专用库：
- PyMC：完整的贝叶斯建模，带有NUTS（自适应HMC）
- emcee：集合MCMC采样器
- NumPyro/JAX：GPU加速的MCMC

你是从头构建这些的。现在你知道库调用在做什么了。

## 练习

1. 实现柯西分布的逆CDF采样。CDF为 F(x) = 0.5 + arctan(x)/pi。生成10,000个样本，用直方图与真实PDF对比。注意重尾（远离中心的极端值）。

2. 使用拒绝采样从 Beta(2, 5) 分布中生成样本，提议使用 Uniform(0, 1)。绘制接受的样本和真实的Beta PDF。理论接受率是多少？

3. 使用蒙特卡罗方法估计 sin(x) 从 0 到 pi 的积分，分别使用1,000、10,000和100,000个样本。比较每个水平下的误差。验证误差随 O(1/sqrt(N)) 缩放。

4. 实现Metropolis‑Hastings，从2D分布 p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链的轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含10个单词的词汇表和它们的logits，使用（a）贪心、（b）温度=0.7、（c）top‑k=3、（d）top‑p=0.9 生成20个词元的序列。比较5次运行中输出的多样性。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------------|----------------------|
| 采样 | “抽取随机值” | 根据概率分布生成值。所有生成式AI背后的机制 |
| 均匀分布 | “所有可能等概率” | [a, b]中的每个值具有相等的概率密度1/(b-a)。所有采样方法的起点 |
| 逆CDF | “概率变换” | F_inverse(U) 将均匀样本转换为已知CDF的任何分布的样本。精确且高效 |
| 拒绝采样 | “提议并接受/拒绝” | 从简单提议分布生成，以与目标/提议比率成正比的概率接受。精确但浪费样本 |
| 重要性采样 | “重加权样本” | 使用来自q(x)的样本估计p(x)下的期望，每个样本加权 p(x)/q(x)。在RL中PPO的核心 |
| 蒙特卡罗 | “平均随机样本” | 将积分近似为样本均值。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | “收敛的随机游走” | 构造一个马尔可夫链，其平稳分布是目标分布。Metropolis‑Hastings是基础算法 |
| Metropolis‑Hastings | “接受上坡，有时下坡” | 提议移动，根据密度比决定接受。细致平衡保证收敛到目标分布 |
| Gibbs采样 | “一次一个变量” | 在固定其他变量的条件下，从每个变量的条件分布中更新该变量。100%接受率 |
| 温度 | “置信度旋钮” | 在softmax之前将logits除以T。T<1 锐化（更自信），T>1 平坦化（更多样） |
| Top‑k采样 | “保留最好的k个” | 将所有除概率最高的k个词元之外的词元置零，重新归一化，采样。固定候选集大小 |
| 核采样（top‑p） | “保留可能的部分” | 保留累积概率超过p的最小词元集。自适应候选集大小 |
| 重参数化技巧 | “将随机性移到外面” | 将 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微分。VAE训练的关键 |
| Gumbel‑Softmax | “软分类采样” | 使用Gumbel噪声和带温度的softmax对分类采样进行可微近似 |
| 分层采样 | “强制覆盖” | 将样本空间划分为层，从每层中采样。方差总是低于朴素蒙特卡罗 |
| 预烧期 | “热身期” | MCMC中的初始样本被丢弃，直到链达到其平稳分布 |
| 细致平衡 | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。p成为马尔可夫链平稳分布的充分条件 |
| 扩散采样 | “迭代去噪” | 从噪声开始并应用学到的去噪步骤生成数据。每一步都是一个条件采样操作 |

## 进一步阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) — 关于MCMC基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) — Gumbel‑Softmax原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — 核（top‑p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — 引入重参数化技巧的VAE论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM将采样与图像生成联系起来
