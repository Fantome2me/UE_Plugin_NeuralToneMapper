# NTM 1.0.1 — 首个可用版本

> **发布于 2026-11-15**
> 下载：`NTM1.0.1.zip`（1.34 MB）· UE 5.8 Win64

## 一句话

把一个能在 UE 编辑器里**稳定运行**的 AI 色调映射插件做出来：AI 驱动的色调曲线 + 硬件自适应 + 与引擎原生 tonemapper 的定量对比。

---

## 这个版本解决了什么

### 1. 致命错误：启用插件后编辑器直接崩溃

**症状**：勾选插件后编辑器立刻崩溃，日志只报一句

```
Fatal error: PipelineStateCache.cpp:712  Shader compilation failures are Fatal.
```

但着色器编译统计明明显示**成功**：

```
FNTMOTonemapPS (compiled 1 times, average 0.01 sec)
```

**排查过程**：逐项排除了着色器参数布局、根签名、纹理复用、`AddFullscreenPass` 用法、引擎/项目问题，最后打开 D3D12 调试层（`-d3ddebug`）才看到真因：

```
[D3DDebug] Vertex Shader - Pixel Shader linkage error:
            Signatures between stages are incompatible.
            Semantic 'TEXCOORD' is defined for mismatched hardware registers
error 80070057 (E_INVALIDARG)
```

**根因**：UE 的全屏顶点着色器（`ScreenPass.usf:12`）输出的是

```hlsl
out noperspective float4 OutUVAndScreenPos : TEXCOORD0;
```

而像素着色器声明的是 `float2 UV : TEXCOORD0` —— **float2 vs float4、缺 noperspective**，签名不兼容，D3D12 拒绝创建 PSO。UE 把这个失败升级成了"着色器编译失败"的致命错误，极具误导性。

**修复**：

```hlsl
void MainPS(
    noperspective float4 UVAndScreenPos : TEXCOORD0,
    float4 SvPosition : SV_POSITION,
    out float4 OutColor : SV_Target0)
{ float2 TextureUV = UVAndScreenPos.xy; ... }
```

### 2. 过曝 5~20 倍

原来只乘了 `GlobalExposure`，画面严重过曝。

`PostProcessTonemap.usf:523` 里 UE 乘的是：

```
SceneColor * Tint * (OneOverPreExposure * GlobalExposure * Vignette * LocalExposure)
```

而 `PostProcessEyeAdaptation.cpp:1586` 给出了契约：

```cpp
// This computation must match FinalLinearColor in PostProcessTonemap.usf
PreExposure = SceneColorTint * GlobalExposure * VignetteMask * LocalExposure;
```

所以 `OneOverPreExposure * GlobalExposure ≈ 1`。改成同样的公式后，自动曝光、手动曝光、曝光补偿全部照常工作。

### 3. 天空发灰

高光去饱和的斜坡写成了 `saturate(Nits / PeakNits)` —— 在 SDR 面板（峰值 100 nit）上，**任何 ≥100 nit 的像素（整个天空）都吃到满额去饱和**。

改成从 60% 峰值才开始渐入。

### 4. 观感参数完全无效（发现的真实 bug）

`Saturation` 从 1.00 改到 1.10，天空饱和度一个数字都没变（0.3011 → 0.3011）。

原因：这些参数存在 `FNTMOCurveParams` 里，由曲线 provider 填充；而神经 provider 的 `ComputeCurve()` **设计上就返回 false**（它的曲线在 GPU buffer 里），于是保留了**解析兜底**的值。用户在工程设置里调的参数从来没生效过。

修复：在 provider 选择之后，直接把设置里的观感参数写进 `Curve`。

### 5. 扩展活跃性死锁

```cpp
// 旧代码
bool IsActiveThisFrame_Internal(...) const { return bIsActive; }
```

而 `bIsActive` 只在 `SetupViewFamily()` 里赋值 —— 但引擎**先用 IsActiveThisFrame 过滤，再分发 setup 钩子**。

结果：扩展永远不活跃，`AddPasses` 一次都没执行过，画面和 UE 原生**完全一样**。

这就是最初"感觉差别不大"的真正原因。

---

## 这一版的能力

| 能力 | 状态 |
|---|---|
| AI 预测色调曲线（315K 参数，[1,48,72,128] → [1,8]） | ✅ |
| 运行时从 .onnx 构建模型（无需导入资产） | ✅ |
| HDR/PQ 输出 + 真实 EDID 读取 | ✅ |
| ABL 自适应（峰值 vs 全屏峰值） | ✅ |
| 局部校正（解析系数，拉普拉斯金字塔 + 引导上采样） | ✅ |
| 与 UE 原生同机位 A/B 验证 | ✅ |
| 曲线时域平滑（EMA） | ✅ |
| 严格 sRGB / Rec709 输出编码 | ✅ |
| 每 4 帧推理一次（省 0.85 ms） | ✅ |

---

## 实测数据

**模型质量**（100k 样本离线验证）

```
曲线 L1        0.360
离散度比       0.977   (1.0 = 完美)
corr(第一点)   0.900
单调性         100%
数值误差       1.9e-06
```

**网络实际输出**（引擎内 GPU 读回，SDR 100 nit 面板）

```
-11.04  -9.31  -7.07  -4.52  -2.15  -0.64  -0.16  -0.11
```

**同机位 A/B**（vs UE 原生）

```
中调偏差     +0.009 档
逐像素相关   0.974
平均误差     11.7 显示级 (of 255)
削顶         黑 0.000%  白 0.022%
```

**HDR 自适应**

| 显示器上下文 | 曲线峰值 |
|---|---|
| SDR 100 nit，ABL 1.00 | 100 nit |
| HDR 1000 nit，全屏 400 nit，ABL 0.40 | 约 430 nit |

**性能**（1600×900，解除 vsync）

```
每帧推理        +0.813 ms
每 4 帧推理     -0.036 ms   ← 默认
```

---

## 安装

1. 解压 `NTM1.0.1.zip`
2. 把 `NeuralToneMapper` 文件夹放到 `<你的工程>/Plugins/`
3. 启动编辑器 → `Edit → Plugins` 勾选 **Neural Tone Mapper** → 重启
4. 确认 **NNERuntimeRDG** 插件已启用
5. 菜单 `构建 → AI动态色调映射`

**零配置**：模型路径默认为空时自动定位插件内的 `Content/Models/NTMO_M_v*.onnx`。

---

## 已知限制

- 曲线比 UE 原生亮约 0.2 档
- 局部算子使用三个手调常数（0.60 / 0.25 / 0.10），不是学出来的
- 高光区局部对比度比 UE 低约 10%
- 未实现完整暗角 / bloom 等价

---

## 校验

解压后运行，日志应出现：

```
[NTMO] Neural Tone Mapper loaded
[NTMO] Scene view extension created
[NTMO] Added "AI动态色调映射" to the Build menu
[NTMO] Curve model loaded from '.../NTMO_M_v6.onnx'
[NTMO] NNE model ready. runtime='NNERuntimeRDGHlsl'
[NTMO]   model inputs : [1,48,72,128]
[NTMO]   model outputs: [1,8]
[NTMO] Neural inference enqueued
```

控制台 `r.NTMO.DumpKnots 1` 可打印网络实际输出的曲线点。
