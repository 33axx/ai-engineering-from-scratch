# 从仿真到现实迁移

> 一个在仿真器中训练好的策略如果在真实硬件上失效，说明这个策略只是记住了仿真器。域随机化、域适应和系统辨识是让习得控制器跨越现实鸿沟的三种工具。

**类型：** 学习  
**语言：** Python  
**先修知识：** 第9阶段·08 (PPO)，第2阶段·10 (偏差/方差)  
**时间：** 约45分钟  

## 问题

训练真实机器人缓慢、危险且昂贵。一个双足机器人需要数百万个训练回合才能学会行走；真实的双足机器人哪怕只摔倒一次就会损坏硬件。仿真让你拥有无限的复位、确定性的可重复性、并行环境以及无物理损伤。

但仿真器是错的。轴承的摩擦力比MuJoCo模型中的大。相机的镜头畸变是仿真器没考虑到的。电机有延迟、回差和饱和，这些在99%的仿真模型中被省略了。风、灰尘和变化的照明会破坏在干净渲染上训练的策略。**现实鸿沟**——仿真分布与真实分布之间的系统性差异——是机器人领域部署强化学习的核心问题。

你需要一个对仿真到现实分布偏移**鲁棒**的策略。历史上主要有三种方法：随机化仿真器（域随机化）、用少量真实数据适应策略（域适应/微调），或者辨识真实系统的参数并与之匹配（系统辨识）。到了2026年，主流方案是将这三种方法与大规模并行仿真（Isaac Sim、Isaac Lab、基于GPU的Mujoco MJX）结合起来。

## 概念

![三种仿真到现实迁移方案：域随机化、域适应、系统辨识](../assets/sim-to-real.svg)

**域随机化 (DR).** Tobin 等 2017，Peng 等 2018。在训练期间，对真实机器人上可能不同的每个仿真参数进行随机化：质量、摩擦系数、电机PD增益、传感器噪声、相机位置、光照、纹理、接触模型。策略会学习一个关于“今天处于哪个仿真中”的条件分布，并在整个跨度上进行泛化。如果真实机器人落在训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一套方案，多款机器人。
- **缺点：** 过度随机化的训练会产生一个“万能”但过于保守的策略。噪声太多 ≈ 正则化太多。

**系统辨识 (SI).** 在训练前先让仿真器的参数拟合真实世界数据。如果你能测量真实机器人手臂关节的摩擦力，就把该值代入仿真器。然后训练一个预期这些值的策略。需要接触真实系统，但能直接缩小现实鸿沟。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残留的模型误差对策略不可见；小的未被辨识出的效应（如电机死区）仍会破坏部署。

**域适应。** 在仿真中训练，然后用少量真实数据微调。两种形式：

- **Real2Sim2Real：** 使用真实轨迹学习一个残差仿真器 `f(s, a, z) - f_sim(s, a)`，然后在修正后的仿真器中训练。无需大量真实数据即可缩小鸿沟。
- **观测适应：** 训练一个策略，通过一个学习的特征提取器（如GAN像素到像素）将真实观测映射到类似仿真的观测。控制器仍保持在仿真中。

**特权学习 / 师生模型.** Miki 等 2022 （ANYmal四足机器人）。在仿真中训练一个可以访问特权信息（地面真相摩擦、地形高度、IMU漂移）的**教师**。然后蒸馏出一个只能看到真实传感器观测的**学生**。学生学会从历史中推断特权特征，从而在各种物理参数下保持鲁棒。

**大规模并行仿真.** 2024–2026年。Isaac Lab、Mujoco MJX、Brax都能在单个GPU上并行运行成千上万个机器人。使用4096个并行人形机器人进行PPO训练，几小时内就能积累数年的经验。随着训练分布变宽，“现实鸿沟”缩小；当这4096个环境中的每一个都具有不同的随机参数时，域随机化几乎变得免费。

**2026年现实世界方案（以四足行走为例）：**

1. 带有域随机化重力、摩擦、电机增益、负载的大规模并行仿真。
2. 使用特权信息（地形图、身体速度真值）训练教师策略。
3. 从教师策略蒸馏出学生策略，仅使用本体感觉（腿部关节编码器）。
4. （可选）通过真实IMU上的自编码器进行观测适应。
5. 部署。零样本在10+个环境上工作。如果失败，用安全约束PPO进行几分钟的真实世界微调。

## 构建

本课的代码是一个简短的演示：在具有**噪声**转换的GridWorld上应用域随机化。我们训练一个经历随机滑移概率（在“仿真”中）的策略，然后在一个训练期间从未见过的滑移水平“真实”环境中进行评估。其形式直接映射到MuJoCo到硬件的迁移。

### 步骤1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在真实机器人中，它可能对应摩擦力、质量、电机增益——任何在仿真与现实之间变化的量。

### 步骤2：使用域随机化训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练PPO / Q学习 / 任何算法。重复大量回合。

### 步骤3：在“真实”滑移值上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支撑集内，`0.5` 和 `0.7` 在支撑集外。使用域随机化训练的策略应该在支撑集内保持接近最优，并在支撑集外优雅地退化。而固定滑移值训练的策略在其训练滑移值之外将非常脆弱。

### 步骤4：与窄域训练对比

训练第二个策略，只使用 `slip = 0.0`。在同样的滑移值扫描上评估。你应该会看到一旦真实滑移 > 0，性能就会灾难性下降。

## 常见陷阱

- **随机化过度。** 在 `slip ∈ [0, 0.9]` 上训练，策略会变得极度风险厌恶，从不尝试最优路径。要与*预期*的真实世界分布匹配，而不是“什么都有可能发生”。
- **随机化不足。** 在一个薄片上训练，策略几乎无法泛化。使用自适应课程学习（自动域随机化），随着策略改进而拓宽分布。
- **参数空间辨识错误。** 随机化了错误的东西（相机色相，而实际差距是电机延迟），域随机化也无济于事。先分析真实机器人。
- **特权信息泄露。** 一个使用全局状态（而非仅观测）来采取动作的教师，可能产生一个无法跟上的学生。确保教师的策略是在给定观测历史的情况下学生可以实现的。
- **仿真到仿真迁移失败。** 如果策略对更难的仿真变体都不鲁棒，那它对真实世界也不会鲁棒。在部署前始终在一个保留的仿真变体上测试。
- **没有真实世界的安全防护。** 一个在仿真中有效并且在“真实”中有效但是缺少底层安全防护的策略仍然可能损坏硬件。在非学习控制中增加速率限制、力矩限制、关节限位。

## 应用

2026年从仿真到现实的典型技术栈：

| 领域 | 技术栈 |
|------|--------|
| 腿式运动（ANYmal、Spot、人形机器人） | Isaac Lab + 域随机化 + 特权教师/学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + 域随机化 + DR-GAN处理视觉 |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + 域随机化 + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + 域随机化 + 在线适应 |
| 手指/手内操作 | OpenAI Dactyl（前所未有的规模域随机化） |
| 工业机械臂 | MuJoCo-Warp + 系统辨识 + 小规模真实微调 |

对于各种规模的控制，工作流程是一致的：尽可能拟合仿真，对无法拟合的部分进行随机化，训练规模巨大的策略，蒸馏，在带有安全防护的情况下部署。

## 产出

将以下内容保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [robotics, reinforcement-learning, sim-to-real]
---

Given:
- Robot platform
- Task
- Observations / actions
- Current simulator
- Real-world constraints

Output:
1. Reality-gap diagnosis: what parameters are most likely mismatched
2. DR plan: what to randomize, with ranges
3. SI plan: what real data to collect and how to fit the sim
4. Training loop: RL algorithm, sim scale, curriculum
5. Deployment safety: limits, fallback controller, evaluation gates

Refuse to recommend deployment without at least:
- held-out sim evaluation
- real-world safety envelope
- rollback / human override plan
```

## 练习

1. **简单。** 在固定滑移值（slip=0.0）的GridWorld上训练一个Q学习智能体。在滑移值 ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报 vs 滑移值的曲线。
2. **中等。** 训练一个域随机化Q学习智能体，采样 `slip ~ Uniform[0, 0.3]`。在同样的滑移值扫描上评估。域随机化在滑移值=0.5（分布外）时带来了多少收益？
3. **困难。** 实现一个课程学习：从 slip=0.0 开始，每当策略达到最优的90%时，扩大域随机化范围。测量达到对 slip=0.3 零样本所需的总环境步数，并与固定的域随机化基线进行比较。

## 关键术语

| 术语 | 大家通常说的 | 实际含义 |
|------|-----------------|-----------|
| 现实鸿沟 (Reality gap) | “仿真与现实的差异” | 训练与部署阶段物理/感知之间的分布偏移。 |
| 域随机化 (DR) | “在各种随机仿真中训练” | 训练时随机化仿真参数，使策略泛化。 |
| 系统辨识 (SI) | “测量真实并拟合仿真” | 估计真实物理参数；设置仿真以匹配。 |
| 域适应 | “用真实数据微调” | 仿真训练后用少量真实数据微调；可能适应观测或动力学。 |
| 特权信息 | “教师的真实信息” | 只有仿真拥有的信息；学生必须从观测历史中推断。 |
| 教师/学生 | “把特权信息蒸馏为可观测” | 教师使用捷径训练；学生学会在没有捷径的情况下模仿。 |
| ADR | “自动域随机化” | 随策略改进而扩大域随机化范围的课程学习。 |
| Real2Sim | “用真实数据缩小鸿沟” | 学习一个残差，使仿真模仿真实轨迹。 |

## 进一步阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 域随机化的原始论文（机器人视觉方面）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 动力学域随机化，四足行走。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，大规模ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— ANYmal的教师-学生方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 推动2025–2026年部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna框架（使用模型进行规划+ rollout），是现代仿真到现实流水线的基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— sim-to-real方法的分类及基准结果。
