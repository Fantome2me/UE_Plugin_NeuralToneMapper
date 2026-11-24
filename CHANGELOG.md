# Changelog

All notable changes to NTM (Neural Tone Mapper).

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