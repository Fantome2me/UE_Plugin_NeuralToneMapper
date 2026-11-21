# 数据集清单

> **数据集本身不进仓库。** 100k 样本 = 3.5 GB，单个分片 1.78 GB，远超 GitHub 的 100 MB 单文件上限。
> 这里保存的是每个数据集的**生成参数与清单**（`*/meta.json`），配合 `training/gen_dataset.py`
> 可以确定性重建 —— 同一条命令，同样的字节（校验方法见 [`../REPRODUCE.md`](../REPRODUCE.md) 第 2 节）。

## 五个数据集

| 目录 | 样本数 | 大小 | 契约 | 用途 |
|---|---|---|---|---|
| `shards` | 100,000 | 3.89 GB | 48 通道 / 8 输出 | 第一版（只有全局曲线），训出 v6 系列 |
| `shards_sdr` | 60,000 | 1.56 GB | 48 / 8 | SDR 专项：v6 在暗部偏亮 0.7 档，用它补 |
| `shards_smoke` | 2,000 | 71 MB | 48 / 8 | 冒烟测试，验证管线能跑通 |
| `shards_v2` | 100,000 | 3.47 GB | 49 / 56 | 加边缘通道 + 48 个局部参数标签，训出 v7 |
| `shards_v3` | 100,000 | 3.48 GB | 49 / 72 | **最终版**：4 个局部通道（含局部曝光），训出 v8 |

**只有 `shards_v3` 是最终发布版用的数据集。** 其余四个是演进过程，留着为了说明"为什么变成现在这样"。

## meta.json 里有什么

```json
{
  "samples": 100000, "c_in": 49, "c_ctx": 45, "c_out": 72, "h": 72, "w": 128,
  "shards": 2,
  "teachers": ["clip","reinhard","hable","ue_uncharted","ue_hp","ue_legacy","ue_aces","analytic"],
  "base_teacher_weights": [0.04,0.1,0.2,0.14,0.1,0.08,0.14,0.2],
  "scene_teacher_boost": { "lowkey": {...}, "highkey": {...}, "specular": {...}, "marquee": {...} },
  "gen_seconds": 176.71, "samples_per_sec": 565.9
}
```

**教师混合权重是这份数据集的灵魂**，它们决定了"什么才算正确的色调曲线"：

- 8 条教师曲线**概率加权混合**，不是硬切换 —— 硬切换会让标签在场景类型边界不连续，实测把可达 L1 卡在 1.0 左右
- `scene_teacher_boost` 按场景类型（低调 / 高调 / 高光 / 夜景）重新加权：高调场景更信 ACES，低调场景更信解析策略
- SDR 单独提高 filmic 曲线的权重 —— 因为最早一版在 SDR 下阴影偏亮

## 重建命令

```powershell
# 最终版数据集（3.5 GB，3 分钟）
.venv311\Scripts\python training\gen_dataset.py --out .\shards_v3 --samples 100000 --ctx 4 --workers 12

# 冒烟测试（71 MB，5 秒）
.venv311\Scripts\python training\gen_dataset.py --out .\shards_smoke --samples 2000 --ctx 4 --workers 2
```

`meta.json` 会写进输出目录，`gen_seconds` 和 `samples_per_sec` 会不同（跟机器有关），其余字段应当与上面完全一致。

## 每个样本长什么样

| 字段 | 形状 | dtype | 内容 |
|---|---|---|---|
| `rgb_pq` | (n, 4, 72, 128) | uint8 | 前 3 通道 PQ 编码的 RGB 缩略图，第 4 通道边缘强度（Sobel） |
| `ctx` | (n, 45) | float32 | 13 个显示/场景标量 + 16 个 4×4 分块亮度 + 16 档直方图 |
| `y` | (n, 72) | float32 | 8 个曲线控制点（log2）+ 64 个局部强度（[-1,1]） |

标签的空域含义：

```
y[0:8]    曲线控制点，x = -10 + 2i（log2，0 = 面板峰值亮度）
y[8:72]   4×4 tiles × 4 bands，tile 主序，每 tile 依次是
          s_base  (curve(base) - curve(logY)，基频压缩)
          s_mid   (G1 - G3，粗频带)
          s_high  (G0 - G1，细频带)
          s_low   (G3 - 帧均值，局部曝光；负号约定见 ntmo_local.py)
```
