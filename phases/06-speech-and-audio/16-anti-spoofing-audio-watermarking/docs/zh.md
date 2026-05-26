# 语音反欺骗与音频水印 — ASVspoof 5、AudioSeal、WaveVerify

> 声音克隆的部署速度远快于防御措施。2026 年的生产级语音系统需要两样东西：一个检测器（AASIST、RawNet2），用于区分真实语音与伪造语音；以及一个水印（AudioSeal），能在压缩和编辑后依然存活。两者都要部署，否则就不要上线声音克隆功能。

**类型：** 构建  
**语言：** Python  
**前置条件：** 阶段 6 · 06（说话人识别）、阶段 6 · 08（声音克隆）  
**时间：** 约 75 分钟

## 问题

三项相关的防御措施：

1. **反欺骗 / 深度伪造检测。** 给定一个音频片段，它是合成的还是真实的？ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成的音频中嵌入一个不可察觉的信号，以便后续检测器能够提取出来。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证溯源。** 对音频文件+元数据进行加密签名。C2PA / 内容真实性倡议。

检测处理的是不合作的攻击者。水印处理的是合规性——AI生成的音频应当可被识别为 AI 生成。到 2026 年，两者都是必需的。

## 概念

![反欺骗 vs 水印 vs 溯源——三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准测试

与先前版本相比最大的变化：

- **众包数据**（非录音室干净数据）——更真实的条件。
- **约 2000 名说话人**（之前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗扰动。
- **两个赛道。** 对抗措施（CM）独立检测；抗欺骗的 ASV（SASV）用于生物识别系统。

ASVspoof 5 上的当前最优水平：等错误率约 7.23%。在较早的 ASVspoof 2019 LA 上：等错误率 0.42%。实际部署：在野生音频片段上预期等错误率 5-10%。

### AASIST 和 RawNet2 — 检测模型家族

**AASIST**（2021 年，更新至 2026 年）。基于图注意力的谱特征处理。当前在 ASVspoof 5 对抗措施任务上的 SOTA。

**RawNet2.** 基于原始波形的卷积前端 + TDNN 骨干网络。更简单的基线；在微调后仍具有竞争力。

**NeXt-TDNN + SSL 特征.** 2025 年变体：ECAPA 风格 + WavLM 特征 + 焦点损失。在 ASVspoof 2019 LA 上达到了 0.42% 的等错误率。

### AudioSeal — 2024 年的默认水印方案

Meta 的 **AudioSeal**（2024 年 1 月，2024 年 12 月 v0.2）。关键设计：

- **局部化。** 以 16 kHz 采样分辨率（1/16000 秒）逐帧检测水印。
- **生成器与检测器联合训练。** 生成器学习嵌入不可听见的信号；检测器学习在增强变换下找到水印。
- **鲁棒。** 能够经受 MP3 / AAC 压缩、均衡器、±10% 的速度偏移、信噪比 +10 dB 的噪声混合。
- **快速。** 检测器运行速度是实时 485 倍；比 WavMark 快 1000 倍。
- **容量。** 每个话语可嵌入 16 位载荷（可编码模型 ID、生成时间戳、用户 ID）。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 位/秒。问题：

- 同步暴力破解速度慢。
- 可被高斯噪声或 MP3 压缩去除。
- 不适用于实时场景。

### WaveVerify（2025 年 7 月）

解决了 AudioSeal 的弱点——尤其是时间域操作（反转、变速）。使用 FiLM 生成器 + 混合专家检测器。在标准攻击上与 AudioSeal 竞争；能够处理时间域编辑。

### 攻击者利用的漏洞

来自 AudioMarkBench：“在音高偏移下，所有水印的位恢复准确率均低于 0.6，表明水印几乎被完全去除。” **音高偏移是通用攻击。** 2026 年没有一种水印对激进的音高修改完全鲁棒。这就是为什么你需要检测（AASIST）与水印同时部署。

### C2PA / 内容真实性倡议

不是机器学习技术——是一种清单格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 使用它。对溯源有效，但如果恶意行为者重新编码并剥离元数据，则不起作用。

## 构建

### 步骤 1：一个简单的谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常具有异常平坦的高频能量。生产级检测器使用 AASIST，而不是这个。但这个直觉是正确的。

### 步骤 2：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 步骤 3：评估 — 等错误率

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都附带：（1）水印，（2）签名清单，（3）符合保留策略的审计日志。

## 使用场景

| 使用场景 | 防御措施 |
|----------|----------|
| 上线 TTS / 声音克隆 | 每个输出都嵌入 AudioSeal（不可商量） |
| 生物识别语音解锁 | AASIST + ECAPA 集成；活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 的来电样本运行 AASIST |
| 播客真实性 | 上传时进行 C2PA 签名，如果是 AI 生成则加上 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 训练/开发/评估集 |

## 陷阱

- **有水印但从未运行检测器。** 毫无意义。在 CI 中部署检测器。
- **检测未校准。** 在 ASVspoof LA 上训练的 AASIST 过拟合；实际准确率下降。在你的领域上校准。
- **音高偏移漏洞。** 激进的音高偏移会去除大多数水印。准备好检测作为后备。
- **元数据剥离与重新托管。** C2PA 可以通过重新编码轻易绕过。始终同时部署加密和感知（水印）防御。
- **将活体检测当作唯一防线。** 要求用户说一个随机短语。可以防止重放攻击，但不能防止实时克隆。

## 交付

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择一个检测模型、水印方案、溯源清单和操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上运行玩具检测器 + 玩具水印嵌入/检测。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位载荷，重新解码。用噪声破坏音频，测量位恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量等错误率。在保留的 F5-TTS 生成片段上测试——观察 OOD 检测性能如何下降。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| ASVspoof | 基准测试 | 双年挑战赛；2024 年为 ASVspoof 5 |
| CM（对抗措施） | 检测器 | 分类器：真实语音 vs 合成/转换语音 |
| SASV | 说话人验证 + CM | 集成的生物识别 + 欺骗检测 |
| AudioSeal | Meta 水印 | 局部化，16 位载荷，比 WavMark 快 485 倍 |
| 位恢复准确率 | 水印存活率 | 攻击后恢复的载荷比特比例 |
| C2PA | 溯源清单 | 关于创作/作者身份的加密元数据 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准测试
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 默认水印方案
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的 MoE 检测器
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 检测骨干网络
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评估
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 溯源清单格式
