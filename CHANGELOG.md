## 1.1.1 — "画面还是平"：找回被一起关掉的 ExpandGamut

### 根因

1.0.8 为了压掉"偏色"，把 `bMatchUnrealColorGrading` 整个关掉。但那个开关**同时**控制两件不相干的事：

```
if (bGrading)  { ExpandGamut = PP.ExpandGamut;  ColorCorrectAll(...); }
else           { ExpandGamut = 0.0f;            /* 分级参数置恒等 */ }
```

- **ExpandGamut** 是 UE 色调映射链里曲线**之前**的一步（默认 1.0），属于 tonemapper，不是调色
- **ColorCorrectAll** 才是对后处理体积里 lift/gamma/gain 的复现

关掉整个开关 = 连 UE 的色域扩展一起丢了。而这条后果**写在我自己的注释里**，我当时没读：

> "Without them the plugin's output is **measurably flatter than Unreal's**: on a night-sky
> scene it measured **−21.5% mean HSV saturation and −48% chroma** on the most saturated 0.6%
> of pixels, which reads on screen as a **grey, washed-out image**."

所以 1.0.8 之后的每个版本：曲线是对的，但渲染出来比 UE 更灰 —— 这就是"平"。

### 修法

把两者拆开，各自一个开关：

| 开关 | 默认 | 含义 |
|---|---|---|
| `bMatchUnrealGamutExpansion` | **True** | 保留 UE 的 ExpandGamut（色域扩展） |
| `bMatchUnrealColorGrading` | False | 后处理体积的调色（会带色相偏移，保持关闭） |
| `r.NTMO.GamutExpand` | -1 | 运行时覆盖：-1 用设置，0 关，1 开 |

引擎内实测确认：

```
[NTMO] gamut expand: setting=1 cvar=-1 PP.ExpandGamut=1.000 -> NTMOExpandGamut=1.000 (grading off)
```

**色域扩展回来了（1.000），分级仍然关闭（不会有偏色）。**

### 同时修掉的两个死设置

上一轮我把 3 个死设置标了 deprecated，这次写了个审计脚本全量扫，又抓出 2 个：

| 设置 | 问题 |
|---|---|
| `AssumeFullFrameFraction` | 从 1.0.0 就在，**从来没被读取过**，却在 UI 上承诺"EDID 不完整时用它估算全帧亮度" |
| `bShowDebugOverlay` | 承诺"角落显示曲线图"的调试视图，**根本没实现**，只被 UHT 生成文件引用 |

均已标注 deprecated（保留属性以兼容旧 ini）。

### 新增：契约审计脚本

`dev/audit/audit_contracts.py` —— 把这次人工查出的每类问题都变成可复跑的检查：

| 检查 | 内容 |
|---|---|
| A | 每个 UPROPERTY 默认值必须与出厂 ini 一致（抓到了 `bEnabled`、`HighlightDesaturation`） |
| B | 每个 UPROPERTY 必须被头文件外的代码读取（抓到 5 个死设置） |
| C | C++ 设的着色器参数必须在 .usf 里出现（98 个） |
| D | .usf 声明的全局必须被使用（22 个） |
| E | 三份插件副本的版本号必须一致 |
| F | 模型输入形状必须与 C++ 的 `NTMOInput` 契约一致 |

退出码非 0 即可卡住发布。这次全绿。

### 曲线未变

`18% 0.165 / 100nit 0.709 / peak 98.3 nit`，与 1.1.0 完全一致，无 shader/RDG 错误。
## 1.1.0 — 全量复查：修掉 5 处代码问题 + 2 处模型问题

### 复查发现的问题

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 1 | `bEnabled` 头文件默认 `false`，ini 是 `true` | 缺 ini 的项目**静默不工作** | 默认改 `true` |
| 2 | `HighlightDesaturation` 头文件 0.15，ini 是 0 | 缺 ini 时偷偷降低高光饱和，违背"只做色调" | 默认改 0.0 |
| 3 | `LocalExposureStrength` / `DetailStrengthMid` / `DetailStrengthLow` **代码里从不读取** | 项目设置里能看见、调了没用 | 标注 deprecated |
| 4 | `NTMOLocal.usf` 注释写 `[8..55] 4x4x3` | 实际是 64 个强度（4x4x4），注释误导 | 改为 `[8..71] 4x4x4` 并写明索引式 |
| 5 | `ctx fed to net` 无条件 Warning | 每次启动都刷日志 | 改为 `r.NTMO.LogCtx`，默认 0 |
| 6 | 曲线路径切片 `ctx[:13]` 含 `ctx[0]`/`ctx[1]`（帧均值、p99.99）**是内容** | 曲线仍受内容影响，离线 0.174 vs 引擎 0.178 | 改切 `ctx[2:13]`（11 个真显示标量） |
| 7 | 曲线在**高光端过冲**：100nit 0.757（UE 0.723）、峰值 105 nit（应 100） | 观感"平均亮度上来、对比度下去"、肩部变软 | 重训并加大高光节点权重 |

### 关于第 7 条的验证过程

先按"抬暗部是否带来 DC"排查：对 24 张真实图像模拟 `Corr = DetailMix*(SMid*(G1-G3) + SHigh*(G0-G1))`，
v9 与 v11 的均值分别是 **−0.0000 / +0.0001 log2** —— 细节项确实零均值，**不是原因**。

于是回到曲线本身，发现高光端系统性过冲。重训后（高光节点权重 1.2→3.4，14000 步）：

| 点 | 1.0.9/1.0.10 | **1.1.0** | UE 目标 |
|---|---|---|---|
| 18% 灰 | 0.173 | **0.165** | 0.180 |
| 100 nit | 0.757 | **0.709** | 0.723 |
| 峰值 | 105 nit | **98.3 nit** | 100 nit |

### 内容不变性（结构性保证，实测为 0）

`ctx[2:13]` 是 11 个显示标量，喂 32 组不同内容统计，曲线输出**极差 0.00000**。
因此离线预测与引擎实测现在**逐位一致**：

```
离线 0.165 / 0.709 / 98 nit     引擎 0.165 / 0.709 / 98.3 nit
```

### 逐峰值精度

10 个峰值档位全部通过，最差偏差 **0.015** 档（1.0.10 是 0.141）：

```
peak 80..4000:  18% 0.165..0.170   100nit 0.709..0.729   (UE 0.180 / 0.723)
```

### 新增可调项

```
r.NTMO.LogCtx 1     # 打印喂给网络的 13 个显示标量（排查用，默认关）
```

### 说明

日志里的 18% 数（0.165）是**线性插值**读数；真正烘焙 LUT 用的是单调三次插值
（Fritsch-Carlson），对凹曲线更贴近真值，实机中间调比这个数字更接近 0.180。
## 1.0.10 — 局部头学会在暗区抬细节，全局亮度不变

### 做法
`NTMOLocal.usf:254-257` 的四个 band 分工是明确的：

| band | 含义 | 数学性质 | 是否改亮度 |
|---|---|---|---|
| 0 | SBase `curve(base) - curve(logY)` | 非线性基带压缩 | **会** |
| 1 | SMid `G1 - G3` | 带通差分 | 零均值 |
| 2 | SHigh `G0 - G1` | 带通差分 | 零均值 |
| 3 | SLow 相对全帧均值的 DC 项 | 抬暗部 | **会** |

所以"暗区抬细节 + 不动全局亮度"就是：**只抬 band 1/2 在暗 tile 上的强度**，
band 0/3 原样保留。因为 band 1/2 是带通差分，天然零均值 —— 亮度不变是**结构性保证**，
不是靠 loss 去惩罚的。

监督目标按每个样本自己的 4×4 tile 亮度中位数算相对暗度：

```
depth[t] = clamp((median(tile) - tile[t]) / 3.0, 0, 1)
target[t, 1] += 0.30 * depth[t]
target[t, 2] += 0.30 * depth[t]
```

曲线路径（`ctx_head`）完全冻结，只训 `head_local` / `ctx_local`。

### 实测（8000 个真实样本）

| | 暗区 band1 | 暗区 band2 | 亮区 band1 | 亮区 band2 | band3 均值 |
|---|---|---|---|---|---|
| 训练前 | 0.297 | 0.223 | 0.668 | 0.808 | −0.2373 |
| **训练后** | **0.489** | **0.468** | 0.691 | 0.832 | −0.2392 |

- 暗区细节强度 **+64% / +110%**
- 亮区几乎不动（+0.02 / +0.02）
- band3（唯一改亮度的 DC 项）**变化 0.0019，可忽略**

### 曲线未被影响
训练前后曲线探针逐位相同，引擎内实测也与 1.0.9 完全一致：

```
18% grey 0.178 (UE 0.180) | 100 nit 0.765 (UE 0.723) | peak 108 nit
sat=1.000 gamut=0.000 desat=0.000 lift=0.0000
```

### 说明
band 1/2 的强度还会被 `r.NTMO.LocalDetail` 缩放到默认四分之一（1.0.2 记录的
"四带全开会让画面发灰"教训正是出自 band 0）。所以这一步提升的是**可用的细节余量**，
默认观感变化温和，需要更明显的暗部细节时调该 cvar 即可。
## 1.0.9 — 曲线改为显示参数直通，重训并实测对齐 UE

### 修正了一个错误基准
1.0.7 把曲线锚定到 **Narkowicz ACES 近似**（0.18→0.267，1.0→0.804），但那不是 UE 用的曲线。
UE 的真实默认值在 `Scene.cpp:445`：`FilmSlope 0.88 / FilmToe 0.55 / FilmShoulder 0.26 / WhiteClip 0.04`，
实测 **0.18→0.180，1.0→0.723**。所以 1.0.7/1.0.8 的"中灰 +0.45 档 / 峰值填充 0.70"是按错靶子推的，
实际把画面推亮了约 0.19 档。

### 训练目标重做
把标签从"五个教师混合"换成 **UE 自己的 filmic 曲线**（按每个样本的面板峰值映射）。
旧标签在 SDR 偏暗 0.2 档、**HDR 偏暗 1.0 档**；新标签在 18% / 100nit 两个锚点上都与 UE 一致。

### 结构改动（关键）
曲线头 `head_knots(h)` 把 **内容通道** 也吃了进去，导致同样的显示参数在引擎内与离线差 1.4 档
（`0.414/1.653` vs `0.154/0.683`）。现在：

```cpp
raw = self.ctx_head(ctx[:, :13])   // 只看 13 个显示标量，内容完全不参与
```

曲线成为显示参数的**确定函数**，引擎内与离线必然一致；内容自适应由局部头继续承担 ——
正是 LLF-LUT 的分工（全局色调看显示，局部细节看图像）。

### 实测（本机 100 nit SDR 面板）

| 指标 | 1.0.8 | **1.0.9** | UE 默认 |
|---|---|---|---|
| 18% 灰 | 0.243 | **0.178** | 0.180 |
| 100 nit | 0.809 | **0.765** | 0.723 |
| 预测峰值 | — | **108 nit** | 100 nit |

`sat=1.000 gamut=0.000 desat=0.000 lift=0.0000` —— 依然不做任何色彩操作。

### 其它
- `MidGreyAnchorStops` / `PeakFill` 默认值归零：模型已直接学对，事后重塑只会过亮
- 曲线日志的参照值改为 UE 真值（`UE 0.180` / `UE 0.723`）
- 新增一次性 `ctx fed to net` 日志，出问题时能直接看到喂给网络的显示参数
- 训练环境：torch 2.11.0+cu128 / sm_120；`torch.compile` 在 Windows 必须关（无 Triton）；
  导出前必须跑 BatchNorm 重标定（`momentum=None` 累计平均会滞后）

### 已知局限
- 峰值 250 nit 一档的拟合误差约 0.14 档，其余档位 < 0.09
- 局部头的"暗区抬细节"仍是 v1.0.2 的监督目标，未在本版重新训练
# Changelog

All notable changes to NTM (Neural Tone Mapper).

## [1.0.7] — 2026-11-24

### Fixed — 三个可量化的缺陷
在夜空工程实测，网络预测的曲线与 UE 的 ACES 逐点对比：

| 输入 | 预测曲线 | ACES | 差 |
|---|---|---|---|
| 18% 中灰 | 0.152 | 0.267 | **暗 0.81 档** |
| 100 nit | 0.584 | 0.804 | 暗 0.46 档 |
| 峰值 | 91 nit / 面板 100 | 100 | **10% 范围没用上** |
| 肩部斜率 (+2..+4) | **0.03** | — | 400 与 1000 nit 落到同一个值 |

于是"颜色变丰富但整体压暗、亮部没有高亮度、暗部没有细节"三件事同时成立。

- **曲线重塑**（`NTMO_ReshapeCurve`）：两个单调重映射，都只作用在**输出**值上，不碰黑端、不换模型
  - 以曲线自身 18% 中灰为心的**高斯抬升**（`MidGreyAnchorStops`）
  - 向真实面板峰值**填充肩部**（`PeakFill`）
  - 默认值不是拍脑袋：扫描 mid×width×peak 三种参数，取"18% 与 100 nit 同时落到 ACES 上、
    且仍用掉 97.5% 峰值"的组合（0.45 / 1.6 / 0.70）
- **阴影抬升增益**（`ShadowBoost`，默认 3×）：原实现的符号本来就是对的
  （`max(0,-Exposure)` 表示"比局部均值暗的区域被抬高"），但实测只有
  `1 档 × s_low 0.2 × scale 0.5 = 0.1 档`，等于没有。3× 后暗区 2 档偏差可开约 0.6 档。
- **色度补偿**（`ChromaCompensation`，默认 0.25）：亮度被压缩就要把色度补回来，
  这是 Hunt 效应的一面（HDR 高光研究里亮度与色度本就是成对处理，见 Shen 等
  "Scaling Lightness and Chroma for Highlight Stimuli in HDR Scenes"）。
  只在"显示还有余量"的地方补，峰值处为 0，因此**不会**把已经到峰值的高光再吹亮。

### 实机验证（不是估算）
新增一次性日志，把网络原始曲线与重塑后曲线在三个参考点上同时打出来：

```
[NTMO] Effective curve | 18% grey 0.155 -> 0.243 (ACES 0.267)
                        | 100 nit  0.593 -> 0.809 (ACES 0.804)
                        | peak     91.0  -> 97.7 nit (panel 100)
```

18% 从暗 0.81 档收敛到暗 0.14 档，100 nit 与 ACES 相差 0.01 档，峰值用掉 97.7%。

### Added
- 设置项：`Mid-Grey Anchor (stops)`、`Mid-Grey Anchor Width`、`Peak Fill`、
  `Chroma Compensation`（以上在 Look|Curve）、`Shadow Boost`（Local Operator）
- `NTMO_ReshapeCurve` / `NTMO_ChromaCompensate`（`NTMOColorGrading.ush`）

### Changed
- `NTMOLocal.usf` 的 `ShadowLift` 乘上 `NTMOShadowBoost`

### 未做的部分
- 没有改成真正在感知均匀空间（ICtCp / JzAzBz）里做色调映射。当前仍是
  "曲线 + 输出域重映射 + 色度补偿"，成本极低但只是那条思路的近似。
  彻底的版本要把 LUT 的索引域也换成感知空间，属于下一轮的结构改动。
## [1.0.6] — 2026-11-24

### Added — 色相行为对齐
- **`Match Unreal Hue Behaviour`（默认开）**：我们的曲线原本是 MaxRGB 形式（只映射最大通道、
  其余等比缩放），**构造上就保色相**；UE 的 `FilmToneMap` 是**逐通道**压缩 R/G/B，
  这正是 ACES 系曲线在高饱和亮色上"移色相"（notorious six）的原因。
  两者是**两张不同的图**，不是同一种算法。
- 新增 `NTMO_ToneMapPerChannelAP1`：同一条学习曲线，改为在 AP1 里逐通道施加，
  并按 `CombineLUTs` 的写法包上蓝色修正 —— `正变换 → 曲线 → 逆变换`。
- BlueCorrection 取自 `View.FinalPostProcessSettings.BlueCorrection`（引擎默认 0.6），
  矩阵为 `mul(AP0_2_AP1, mul(BlueCorrect, AP1_2_AP0))`。
- 控制台变量 `r.NTMO.Hue`：`-1` 跟随设置、`0` 用 MaxRGB（保色相，1.0.5 及以前的行为）、
  `1` 用逐通道 + 蓝修正。

### 说明
- 关掉它得到的是"更干净但不像 UE"的画面（色相严格不偏），打开才是 UE 的画面。
  如果目标是复刻 UE，这项要保持开。
- 蓝色修正本身量级很小（默认强度下对色相 −0.5°～−3.5°），
  **真正起作用的是逐通道曲线**：它才是高光色相走向与 UE 一致的原因。

### 未验证的部分
- 着色器编译通过、插件运行正常，但**没有给出色相偏差的量化改进**：
  这个工程的动态天空 + 自动曝光让截图 A/B 的噪声（亮度漂移 25%）大于要测的效应。
  请用 `r.NTMO.Hue 0/1` 直接切换判断。
## [1.0.5] — 2026-11-24

### Fixed
- **画面发灰、色彩变淡**：插件替换 tonemapper 时，UE 会连带跳过 `AddCombineLUTPass`，
  而 `PostProcessCombineLUTs.usf` 里做的就是调色 —— 顺序是
  `WhiteBalance → ExpandGamut → ColorCorrectAll → 胶片曲线`。
  1.0.4 只做了最后一步，前面全丢了。其中 **`ExpandGamut` 的引擎默认值是 `1.0`**
  （`Scene.cpp:437`），它把明亮的高饱和色推开到 P3–AP1 之间，正是"颜色不够艳"的直接原因。
- 新增 `references` 之外的第三块：`Shaders/Private/NTMOColorGrading.ush`，逐行移植
  UE 的 `ColorCorrect` / `ColorCorrectAll` / `ExpandGamut`，数学与引擎源码一致
  （色彩修正以 0.18 为中灰支点、三区按 luma 用同样的 smoothstep 混合）。
- LUT 烘焙按 UE 的顺序接入：`工作空间 → AP1 → ExpandGamut → ColorCorrectAll → 回工作空间 → 曲线`。
  全部参数取自 `View.FinalPostProcessSettings`，所以工程自己的后处理体积调色也会被正确复现。
- AP1 变换矩阵为 CPU 常量：`FromAP1 = XYZ_2_sRGB · D60_2_D65_CAT · AP1_2_XYZ`，
  `ToAP1` 取其数值逆（往返误差 7.6e-17），`ExpandMat = Wide_2_AP1 · AP1_2_sRGB`。

### Added
- 设置项 `Match Unreal Colour Grading`（默认开）。所有参数在默认值下都是恒等变换，
  所以对没有调色的工程**不会有任何改变**，只会把本该有的调色补回来。
- 控制台变量 `r.NTMO.Grading`：`-1` 跟随设置、`0` 强制关（复现 1.0.4 的观感）、`1` 强制开。
  加它是为了能在**同一次编辑器会话内**做 A/B。

### Measured, and what could not be measured
- `ExpandGamut` 的作用量 = 色度门控 × 亮度门控：Luma 0.08 时只有 1.7%，
  Luma 0.2 时 10.5%，Luma 0.8 时 83%。也就是说它**只在亮部生效**，
  暗调场景里单独靠它补不回全部差距 —— 这一点按数值算清楚了，不靠观感判断。
- 截图 A/B 在这个工程上**不可信**：场景是动态天空 + 自动曝光，相隔 12 秒的两张图
  地面亮度会从 0.316 漂到 0.238，噪声大于要测的效应。因此本版**不给出**
  "饱和度提升了 X%" 这类数字。要自查请用 `r.NTMO.Grading 0/1` 在运行时切换。
## [1.0.4] — 2026-11-17

### Fixed
- **RDG 资源顺序错误**：`NTMO::SceneStats` 读取 `NTMO.Local.Lum` 时该纹理还没有生产者，
  触发 `RenderGraphValidation.cpp:653` 的 ensure。1.0.3 把 stats pass 移出神经分支时
  落在了金字塔上方而不是下方。RDG 按 `AddPass` 顺序追踪资源版本，读必须先于写的注册
  顺序严格在先写后读。
- 该错误导致帧统计缓冲区读到未定义数据，进而污染网络输入的 32 个通道（直方图 + 分块亮度）。
  修复后画面亮度与 UE 原生偏差从 +0.353 档降到 +0.003 档，平均误差 11.7 → 7.5 / 255 级，
  逐像素相关 0.934 → 0.9663。

### Changed
- 渲染验证模式扩展为 `never written to|Ensure condition failed|RenderGraphValidation`
  等 RDG 校验关键字，并扫描 `Saved/Logs` 下最近的日志。

### Not adopted
- 用 G0（1/2 分辨率）替代 G1（1/4）作为局部曝光参考。实测最暗 0.1% 像素会提升 2.18 档，
  但局部校正无时域累积，会引入暗部噪声闪烁，且整体指标无改善（相关 0.9647 vs 0.9663）。
## [1.0.3] — 2026-11-15

### Fixed
- **"效果反了"**：暗部细节不变、亮部变化更明显。根因是 Durand 细节恢复的幅度正比于 `1 - 曲线斜率`，而 SDR 曲线在暗部斜率接近 1 —— 能量天然集中在高光；且三个回归量都是零均值频带差，无法表达"整片提亮"。
- 局部校正改为**混合域**：曝光作用于曲线前（场景参考）、细节作用于曲线后（显示参考）。
- 局部曝光参考从 1/16 分辨率（G3）换成 1/4（G1），细窄阴影不再被邻居平均掉。

### Added
- 第 4 个局部通道 `s_low`（局部曝光 DC 项），输出契约 56 → 72。
- `Local Exposure Scale`、`Local Detail Mix` 两个强度旋钮。
- 训练工具 `--freeze-backbone`：只训局部头，避免拖累曲线。
- 日志打印 4 个 tile 的局部参数，便于排查符号问题。

### Changed
- 校正纹理 `R16F` → `RG16F`。

## [1.0.2] — 2026-11-15

### Added
- **边缘条件输入**（Canny 式 Sobel），输入契约 48 → 49 通道。
- **48 个学习出来的局部拉普拉斯强度**（4×4 × 3），输出契约 8 → 56。
- 按 Durand 分解重写局部算子，校正作用于曲线之后。
- LUT 改为存线性显示值，转移函数移到像素着色器。
- 曲线输出物理钳制到 `[log2(minNits), log2(peakNits)]`。

### Removed
- 旧的 8-tap 双边上采样（校正场本身平滑，双线性足够）。

### Known issues
- 学出来的局部算子对画面贡献接近零（标签目标可被回归量精确表达，没有可学的自由度）。

## [1.0.1] — 2026-11-15

### Fixed
- **致命错误**：像素着色器签名 `float2 TEXCOORD0` 与 UE 全屏 VS 的 `noperspective float4 TEXCOORD0` 不兼容，D3D12 拒绝创建 PSO，UE 误报为"着色器编译失败"。
- **过曝 5~20 倍**：曝光乘错，应为 `OneOverPreExposure * GlobalExposure`。
- 天空发灰：高光去饱和斜坡起点应为 60% 峰值。
- 观感参数（Saturation 等）完全无效：被解析兜底值覆盖。
- 扩展活跃性死锁：`bIsActive` 只在被它门控的钩子里赋值，导致 `AddPasses` 从未执行。

### Added
- AI 预测色调曲线（315K 参数，`[1,48,72,128] → [1,8]`）。
- 运行时从 `.onnx` 构建模型，无需导入资产。
- HDR/PQ 输出 + 真实 EDID 读取 + ABL 自适应。
- 曲线时域 EMA（半衰期 0.15 s）。
- 每 4 帧推理一次（省 0.85 ms）。
- 严格 sRGB / Rec709 输出编码。
- 同机位 A/B 验证框架（编辑器内 Python + 外部驱动 + 定量比较）。