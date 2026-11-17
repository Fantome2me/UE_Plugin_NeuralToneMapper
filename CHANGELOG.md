# Changelog

All notable changes to NTM (Neural Tone Mapper).

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