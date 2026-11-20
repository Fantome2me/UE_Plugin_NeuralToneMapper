# NTM — Neural Tone Mapper (神经网络动态色调映射）

> 一个 Unreal Engine 5.8 插件：**用一个小神经网络每帧重新决定色调映射曲线**，并根据你的**显示器实际亮度能力**自动调整。
>
> 315K→327K 参数，1.3 MB ONNX，GPU 上推理，插件整体开销约 **0 毫秒**。

[![UE](https://img.shields.io/badge/Unreal%20Engine-5.8-0E1128)](https://www.unrealengine.com)
[![Platform](https://img.shields.io/badge/Platform-Win64%20%7C%20D3D12-blue)]()
[![Params](https://img.shields.io/badge/Params-327K-green)]()
[![Cost](https://img.shields.io/badge/Overhead-~0%20ms-brightgreen)]()

---

## 目录

- [一、色调映射到底是什么](#一色调映射到底是什么通俗版)
- [二、UE 自带的色调映射，问题在哪](#二ue-自带的色调映射问题在哪)
- [三、神经网络在这里做什么](#三神经网络在这里做什么)
- [四、NN 怎么替代 LUT、怎么接管 tonemapper](#四nn-怎么替代-lut怎么接管-tonemapper)
- [五、如何根据显卡和显示器自动调节](#五如何根据显卡和显示器自动调节)
- [六、安装与使用](#六安装与使用)
- [七、实测数据](#七实测数据)
- [八、已知限制](#八已知限制)
- [九、版本历史](#九版本历史)

---

## 一、色调映射到底是什么

### 问题：一万倍塞进一百倍

真实世界的光照范围极大。太阳直射的地面可能是阴影里的 **一万倍** 亮。

但屏幕是有上限的：一块普通 SDR 显示器最亮也就 100nit，HDR 屏能到 2000 nit，但依然是**有限的**。

于是每帧画面都要做一件事：**把「无限的亮度范围」压缩到「屏幕能显示的范围」**。这个压缩就叫**色调映射（Tone Mapping）**。

压缩不是简单的除法。如果直接除以 100：

- 亮的地方全部变成纯白，糊成一片
- 暗的地方全部变成纯黑，细节全丢

所以真实做法是一条**曲线**：暗部尽量保留、中段基本不变、亮部平滑地压向屏幕峰值。

```
屏幕亮度
  ↑
1.0├─────────────────────────╱──────  ← 曲线：越亮压得越狠
   │                     ╱
   │                ╱
   │           ╱
   │      ╱
0.0├──╱──────────────────────────→ 场景亮度（可以远超 1.0）
   0                          1.0      4.0   16.0
```

**关键点：不同的曲线 = 不同的画面风格。** 曲线压得狠，画面"电影感"强但暗部丢失；压得缓，细节多但可能发灰。这就是为什么各家引擎/游戏都有自己的色调映射曲线。

---

## 二、UE 自带的色调映射，问题在哪

Unreal 用的是 **ACES 家族**的固定曲线，质量很好，但有两个根本限制：

### 限制 1：不管玩家和用户硬件的区别

无论你的显示器是 100 nit 的办公屏，还是 1600 nit 的 HDR 电视，UE 用的都是**同一条曲线**。

后果：

- 在 **HDR 屏**上，曲线压得太保守，你花大价钱买的亮度范围**大半没用上**
- 在 **低亮度屏**上，曲线又压得不够，高光糊成一片

### 限制 2：不看画面表现内容

一条固定曲线，对"昏暗的室内"和"烈日下的雪地"是一视同仁的。

但这两者需要的压缩策略完全不同：雪地要保住高光的层次，室内要保住暗部的细节。

UE 通过**自动曝光**缓解了一部分，但曝光只调整体亮度，**不改变曲线的形状**。

### 我们做的事

> **把那条固定曲线，换成每帧由神经网络根据「画面长什么样」+「你的屏幕能多亮」现算的一条。**

---

## 三、神经网络在这里做什么

计算曲线的 8 个控制点：

```
输出（log2，0 = 屏幕峰值）
  0 ┤                                    ●  ← 第 8 点：曲线最高到哪
    │                              ●
    │                        ●
 -4 ┤                  ●
    │            ●
    │      ●
 -8 ┤  ●
    └──┬────┬────┬────┬────┬────┬────┬───→ 8 个固定输入位置
      -10  -8   -6   -4   -2   0   +2  +4   （场景亮度，log2，0 = 白）
```

8 个点之间线性连接，得到一条平滑曲线。这足以描述任何合理的色调曲线。

### 它靠什么决定这 8 个数字

**材料一：一张缩小的画面（49 通道张量，128×72）**

| 通道 | 内容 | 为什么需要 |
|---|---|---|
| 0–2 | PQ 编码的 RGB | 画面本身长什么样 |
| 3 | **边缘强度（Sobel）** | 知道哪里有强烈明暗边界 |
| 4–5 | 全画面平均亮度、最亮 0.01% 的亮度 | 判断整体曝光水平、有没有极端高光 |
| 6–16 | 显示器参数（见第五节） | 屏幕能显示多少 |
| 17–32 | 4×4 分块的平均亮度 | 明暗**分布**（天空在上还是在下） |
| 33–48 | 16 档亮度直方图 | 多少像素在暗部、多少在亮部 |

**材料二：网络结构**

```
49×72×128 输入
   ├─ 图像分支：4 层卷积（stride 2）→ 全局池化 → 128 维
   └─ 上下文分支：13+32 个标量 → 全连接 → 128 维
                    ↓ 相加（不用 Concat，见第四节）
              LayerNorm → 128 维
                    ↓
        ┌───────────┴───────────┐
   曲线头（8 点）          局部参数头（64 个数）
```

总计 **327K 参数**，ONNX 文件 1.3 MB。

### 为什么 8 个点必须是"严格递增"的

如果曲线的两个点顺序颠倒，画面会出现**对比度反转**（亮的地方变暗）。

所以输出层不是普通线性层，而是这么构造的：

```
第 8 点  = 起点 + 跨度 × sigmoid(z)
中间 7 段 = softmax(7 个 logit)   ← 7 段占比之和恒为 1
每个点   = 起点 + 跨度 × 前缀和
```

数学上**保证**严格递增，不需要事后修补。

---

## 四、NN 怎么替代 LUT、怎么接管 tonemapper

### 4.1 接管：一个后处理接缝

UE 的后处理链有一个枚举值叫 `ReplacingTonemapper`。当有插件注册了它，UE 就会**跳过自己的 `AddCombineLUTPass` 和 `AddTonemapPass`**。

我们注册的就是这个：

```cpp
void FNTMOSceneViewExtension::SubscribeToPostProcessingPass(
    EPostProcessingPass Pass, const FSceneView& View,
    FPostProcessingPassDelegateArray& InOutPassCallbacks, bool bIsPassEnabled)
{
    if (Pass != EPostProcessingPass::ReplacingTonemapper) return;
    // 注册我们的 pass，UE 自己的 tonemapper 就不再执行
}
```

**不改引擎源码**，纯插件。

### 4.2 LUT：为什么还需要它

曲线确定了之后，理论上每个像素都要算一次「场景亮度 → 曲线 → 屏幕亮度」。

4K 有 830 万像素，每个都算一遍太浪费。

所以：**先把曲线烘焙成一张查找表（3D LUT，33×33×33）**。

```
每帧开始：
   神经网络 → 8 个点 → 曲线 → 烘焙成 33³ 的 3D 纹理（约 3.6 万个格子）
每个像素：
   查一次表  ← 就这么简单
```

**注意这里是"NN 替代 LUT"的正确理解：**

| 说法 | 是否准确 |
|---|---|
| "神经网络替代了 LUT" | ❌ 不准确。LUT 还在，还是每个像素查表 |
| "神经网络决定了 LUT 的内容" | ✅ 这才是准确的 |

传统做法里，LUT 的内容来自一条**写死的曲线**（烘焙时算好，运行时不变）。
我们的做法里，LUT **每帧都被重新烘焙一次**，内容来自网络输出。

所以准确说法是：**神经网络把 LUT 从"静态"变成了"每帧自适应"**。

### 4.3 为什么不用 Concat 拼输出

网络的输出是 72 个数（8 曲线 + 64 局部）。直觉上应该用 `Concat` 拼起来。

但 UE 的 `NNERuntimeRDGHlsl` 后端**不支持 Concat 算子**（`Pad` 也会被降级成 Concat）。

所以用一个 **0/1 放置矩阵 + MatMul** 来做同样的事：

```python
# out = knots @ place_knots + local @ place_local
place_knots: [8, 72]   每行只有一个 1，把第 i 个 knot 放到第 i 位
place_local: [64, 72]  每行只有一个 1，把第 j 个局部参数放到第 8+j 位
```

MatMul 是支持的，效果完全等价。

### 4.4 完整的一帧流程

```
场景色（曝光前）
   │
   ├─ 亮度金字塔：G0(1/2) G1(1/4) G2(1/8) G3(1/16)，都是 log2 亮度
   ├─ 统计：4×4 分块 + 16 档直方图（uint 原子累加，卡上无浮点原子操作）
   ├─ 组装：49 通道 × 72 × 128 张量
   │
   ├─ 【神经网络推理】NNE Runtime 直接发 RDG 计算指令，无 CPU 回读
   │     → 72 个数写入 GPU buffer
   │
   ├─ 时域平滑：EMA（半衰期 0.15 秒），防止曲线跳变
   │
   ├─ 局部校正（见 4.5）
   ├─ 烘焙 LUT（33³）
   └─ 最终 pass：曝光 → 局部增益 → 查 LUT → 输出编码 → 抖动
```

### 4.5 局部校正：NN也管一部分

除了全局曲线，网络还输出 **4×4 网格 × 4 个通道 = 64 个局部参数**：

| 通道 | 作用 | 应用位置 |
|---|---|---|
| `s_base` | 基频非线性压缩 | 曲线**前** |
| `s_mid` / `s_high` | 粗/细频带细节 | 曲线**后** |
| `s_low` | **局部曝光**（暗区提亮） | 曲线**前** |

```
校正量(RGB 通道) = s_low × clamp(局部亮度 − 帧均值, −3, +3) × 曝光强度
```

**为什么分两个域**：局部曝光物理上就是场景亮度操作，放在曲线前才能被曲线陡峭的暗部放大；细节补偿必须放在曲线后，否则曲线会把它重新压回去。

---

## 五、如何根据显卡和显示器自动调节

### 5.1 读真实 EDID

每帧（节流到 0.5 秒一次）调用 UE 的 `HDRGetMetaDataForWindow()` 读显示器真实能力：

| 参数 | 含义 | 影响 |
|---|---|---|
| `MaximumLuminanceInNits` | 面板峰值亮度 | 曲线最高点 |
| `MaximumFullFrameLuminanceInNits` | **全屏全白时**的峰值（ABL 限制） | 见 5.2 |
| `MinimumLuminanceInNits` | 黑位 | 曲线最低点 |
| `HDRPaperWhiteInNits` | SDR 白在 HDR 下的对应亮度 | 映射基准 |
| `DisplayOutputFormat` | sRGB / Rec709 / PQ / scRGB | 输出编码 |
| `DisplayColorGamut` | 色域 | 饱和度处理 |

读不到 EDID 时退回引擎 cvar（`r.HDR.Display.*`），再不行按 100 nit SDR 处理。

**没有 EDID 的设备**（投影机、采集卡、虚拟显示器、转换器）可以手动覆盖：

```
Project Settings → Plugins → Neural Tone Mapper
  Override Display Metadata  ✔
  Override Peak Nits         1000
  Override Full Frame Nits   400
  Override Output Device     3   (3 = PQ / ST.2084)
```

### 5.2 ABL：为什么"全屏亮度"和"峰值亮度"是两回事

很多 HDR 屏（尤其 OLED）标称 1000 nit，但**全屏显示白色时会自动降到 300~400 nit**（ABL，自动亮度限制），否则会过热或烧屏。

如果只按 1000 nit 做色调映射，在明亮的大场景里画面会被屏本身压暗，看起来"怎么调都不亮"。

我们把 **峰值** 和 **全屏峰值** 都作为输入给网络，让它学会在 ABL 严重的屏上更保守。

实测（同一帧画面、同一个网络，只改显示器参数）：

| 显示器上下文 | 网络输出的曲线峰值 |
|---|---|
| SDR 100 nit，ABL 1.00 | 100 nit |
| HDR 1000 nit，**全屏 400 nit，ABL 0.40** | **约 430 nit** |

峰值精确跟踪了 ABL 限制后的全屏亮度。这是"硬件自适应"的硬证据。

### 5.3 四种输出编码

| 设备 | 编码 | 说明 |
|---|---|---|
| 0 | sRGB（分段 OETF） | 普通 SDR 屏 |
| 1 | Rec.709 | 广播标准 |
| 2 | 显式 gamma | 手动指定幂函数 |
| 3 / 4 | **PQ (ST.2084)** | HDR10 |
| 5 / 6 | scRGB | Windows HDR 桌面（线性，1.0 = 80 nit） |

### 5.4 显卡自适应

| 机制 | 说明 |
|---|---|
| **零 CPU 同步** | 用 `NNERuntimeRDGHlsl`，推理是真正的 RDG 计算指令，**不调用** `SubmitAndBlockUntilGPUIdle` |
| **每 4 帧推理一次** | 曲线是慢变量，30 Hz 更新肉眼无法分辨；实测画质差异在小数点后第 4 位 |
| **半分辨率局部计算** | 局部校正只在 1/2 分辨率算，增益由已有的 tonemap pass 消费，不增加全屏 pass |
| **自动降级** | 若 NNE runtime 或模型不可用，自动退回解析曲线；再不行交还 UE 原生 tonemapper |

实测开销（1600×900 视口，解除 vsync，400 帧平均）：

| 配置 | 相对 UE 原生 |
|---|---|
| 全开（AI 曲线 + AI 局部） | **-0.04 ms** |
| 每帧推理（`r.NTMO.Interval 1`） | +0.81 ms |
| 只跑 LUT + tonemap | **-0.30 ms** |

**我们自己的 LUT + tonemap 比 UE 的 tonemapper 还快 0.30 ms**（UE 那一趟还要做 bloom 混合、暗角、色彩分级、颗粒、抖动）。多出来的只有神经网络那部分。

---

## 六、安装与使用

### 6.1 安装

**方式 A：直接下载 release 里的 zip**

1. 下载 `NTM1.0.4.zip`
2. 解压，把 `NeuralToneMapper` 文件夹放到：
   ```
   <你的工程>/Plugins/NeuralToneMapper/
   ```
3. 启动编辑器，`Edit → Plugins` 勾选 **Neural Tone Mapper**，重启
4. 确认引擎插件 **NNERuntimeRDG** 已启用（Edit → Plugins 搜 "NNE"）

**方式 B：用仓库里的源码**

```
<你的工程>/Plugins/NeuralToneMapper/
```
需要有对应版本的编译环境（Visual Studio 2019/2022）。

### 6.2 使用

- **菜单**：`构建 → AI动态色调映射`（勾选即启用）
- **控制台**：

| 命令 | 作用 |
|---|---|
| `r.NTMO.Enable 1` / `0` | 启用 / 关闭（关闭后回到 UE 原生） |
| `NTMO.Toggle` | 切换 |
| `r.NTMO.DumpKnots 1` | 打印网络实际输出的 8 个曲线点 + 局部参数 |
| `r.NTMO.Interval <1..8>` | 推理间隔（默认 4） |
| `r.NTMO.Neural <0/1>` | 强制关/开神经网络（0 = 用解析曲线） |
| `r.NTMO.Local <0/1>` | 强制关/开局部校正 |

### 6.3 主要设置项

`Project Settings → Plugins → Neural Tone Mapper`

| 设置 | 默认 | 说明 |
|---|---|---|
| `Curve Source` | Neural | Analytic = 解析曲线（不用 AI） |
| `Curve Model File` | （空） | 空 = 自动定位插件内的最新模型 |
| `Saturation` | 1.08 | 饱和度 |
| `Highlight Desaturation` | 0.05 | 高光去饱和 |
| `Local Detail Mix` | 0.25 | 局部细节强度 |
| `Local Exposure Scale` | 0.5 | 局部曝光强度（暗区提亮） |
| `Local Gain Clamp Stops` | 1.5 | 局部校正硬上限 |
| `Curve Smoothing Half-Life` | 0.15 s | 曲线时域平滑 |
| `Inference Interval` | 4 | 每 N 帧推理一次 |

---

## 七、实测数据

### 7.1 与 UE 原生同机位对比（v1.0.4）

同一编辑器会话、同一相机、同一帧，只切 `r.NTMO.Enable`：

```
亮度偏差     +0.003 档     ← 几乎完全一致
对比度 std 比 0.964
逐像素相关   0.9663
平均误差     7.5 显示级（of 255）
削顶         黑 0.044%   白 0.094%（UE：黑 0.002% 白 0.067%）
```

### 7.2 模型质量（离线验证，100k 样本）

| 指标 | v1.0.1 | v1.0.2 | v1.0.3 |
|---|---|---|---|
| 曲线 L1（越小越好） | 0.360 | 0.323 | **0.321** |
| 预测/真值离散度比（1.0 最佳） | 0.977 | 1.000 | 0.994 |
| corr(第一点) | 0.900 | 0.914 | 0.914 |
| 单调性 | 100% | 100% | 100% |
| 局部参数 corr | — | 0.53 | **0.76** |

### 7.3 局部校正的逐亮度带分布（v1.0.3）

| 输出亮度 | 占比 | 校正量 |
|---|---|---|
| 0.05–0.15（暗） | 13.4% | +0.024 档 |
| 0.15–0.35（中暗） | 11.3% | **+0.203 档** |
| 0.35–0.60（中） | 22.2% | +0.165 档 |
| 0.60–0.85（亮） | 53.0% | +0.116 档 |

方向正确：中暗区提升最多，亮区最少。

---

## 八、已知限制

诚实列出，不粉饰：

1. **整体比 UE 原生亮约 +0.3 档**。其中 +0.19 来自曲线本身（模型在 100 nit 面板上预测峰值偏高），+0.12 来自局部校正。
2. **最深的阴影（输出 < 0.15）变化仍然小**。局部曝光用 1/4 分辨率参考，对成片暗区有效，对亮表面旁的细窄阴影检测不到。
3. **局部校正强度不能无限加大**。显示域有 1.0 上限，任何对称的对比度增强在钳制后都会产生净提亮（高光被削、阴影不被削），强度过大画面会发灰。
4. **局部参数的时域没有 EMA**（只有曲线有）。极端内容切换时局部校正可能有轻微跳变。
5. **替换而非叠加** UE 的 Local Exposure（后者在被替换的 pass 内部）。
6. **VignetteMask 按 1 处理**，未做完整暗角等价。
7. **每个进程只加载一次模型**，换 `.onnx` 需重启编辑器。

---

## 九、版本历史

| 版本 | 关键内容 |
|---|---|
| [1.0.1](docs/RELEASE_NOTES_1.0.1.md) | 首个可用版本：修复致命 PSO 错误、曝光链、HDR/EDID 自适应、同机位 A/B 验证 |
| [1.0.2](docs/RELEASE_NOTES_1.0.2.md) | 补齐论文的局部 AI：边缘条件输入、48 个学习局部参数、Durand 分解重写局部算子 |
| [1.0.3](docs/RELEASE_NOTES_1.0.3.md) | 修正「效果反了」：混合域校正、局部曝光 DC 通道、G1 参考、冻结主干训练 |
| [1.0.4](docs/RELEASE_NOTES_1.0.4.md) | **修复 RDG 资源顺序错误**（帧统计读到未写入纹理），亮度偏差 +0.353 → +0.003 档 |

详细技术说明见 [docs/技术说明.md](docs/技术说明.md)。

---

## 许可

MIT，见 [LICENSE](LICENSE)。

## 致谢

局部算子架构参考 NeurIPS 2023 论文 **"Lookup Table meets Local Laplacian Filter"**（LLF-LUT）。
神经网络的训练管线、渲染管线、UE 集成均为本项目独立实现。

---

**English abstract**

NTM is an Unreal Engine 5.8 plugin that replaces the engine's fixed ACES tone curve with a 327K-parameter neural network that predicts an 8-knot tone curve plus 64 local Laplacian strengths every frame, conditioned on the frame's content and on the display's real EDID capabilities (peak nits, ABL-limited full-frame nits, paper white, colour gamut, output transfer function). The predicted curve is baked into a 33³ 3D LUT each frame and sampled once per pixel, so runtime cost stays at parity with the engine tonemapper (~0 ms overhead at the default 4-frame inference interval). Exposure remains Unreal's own eye adaptation, so every exposure setting keeps working. Measured against Unreal's tonemapper on the same camera: 0.934 per-pixel luma correlation, mid-tone deviation +0.009 stops, no clipping.
