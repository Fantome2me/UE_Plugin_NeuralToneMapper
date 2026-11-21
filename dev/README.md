# 开发版源码与训练数据

这个目录放的是**训练侧**的东西：数据生成、模型定义、训练、导出、验证，以及 A/B 对比工具。
运行时的插件本体在仓库根目录的 `plugin/NeuralToneMapper/`。

> **想直接复现模型？先看 [`REPRODUCE.md`](REPRODUCE.md)** —— 那是一份从零到可部署 ONNX 的完整步骤，
> 每步都有可校验的输出与预期指标。

> 为什么数据只有一个小样本：完整的训练集是 **100,000 条样本、约 3.5 GB**（每个样本是一张 4×72×128 的 PQ 图 + 45 个上下文标量 + 72 维标签），
> 单个分片 1.78 GB，远超 GitHub 的 100 MB 单文件上限。所以这里只放样本与**生成配方**，
> 完整数据用 `training/gen_dataset.py` **三分钟可重新生成**，而且现在是**确定性**的
> （同样的命令行 → 同样的字节，进程数只影响速度）。

---

## 目录

```
dev/
├─ REPRODUCE.md         ★ 从零复现：环境 → 数据 → 训练 → 验证 → 导出
├─ training/            训练管线
│  ├─ ntmo_pack.py      输入/输出契约（C_IN=49, C_OUT=72）+ 打包
│  ├─ ntmo_model.py     网络定义（曲线头 + 局部参数头 + 放置矩阵）
│  ├─ ntmo_synth.py     合成 HDR 场景 + 显示器上下文采样
│  ├─ ntmo_teachers.py  教师曲线（ACES / Uncharted / Hable / Reinhard / 解析策略…）
│  ├─ ntmo_local.py     局部拉普拉斯参数标签（Durand 参考 + 4 通道最小二乘拟合）
│  ├─ gen_dataset.py    并行生成数据分片（确定性）
│  ├─ train.py          训练（支持 --init-from 热启动、--freeze-backbone）
│  ├─ export_onnx.py    导出 ONNX + 数值一致性检查 + 单调性检查
│  └─ _verify.py        离线评估（L1 / 离散度 / 单调性 / 相关性 / 局部参数误差）
├─ checkpoints/         ★ 三个已训练权重（v6b / v7b / v8b，共 8.8 MB）
├─ environment/         ★ 训练环境快照（requirements.txt + 硬件/版本记录）
├─ logs/                ★ 每次训练的完整日志（含失败与调参过程）
├─ manifests/           ★ 五个数据集的生成参数（meta.json）+ 数据集说明
├─ data/
│  └─ sample_shard.npz  32 条样本（279 KB），格式与正式分片完全相同
└─ ab/
   ├─ ntmo_ab_harness.py          编辑器内 Python：固定机位、文件握手切换插件
   ├─ _ab_run.ps1                 外部驱动：启编辑器、抓取 ON/OFF 两张图
   └─ _ab_compare_same_session.py 定量比较（逐亮度带、削顶率、相关性）
```

### 新增的三样东西是干什么用的

| 目录 | 内容 | 为什么需要 |
|---|---|---|
| `checkpoints/` | `best_v6b.pt` / `best_v7b.pt` / `best_v8b.pt` | 有它们就能**跳过训练直接导出 ONNX**；也可以拿来热启动继续微调 |
| `environment/` | `pip freeze` 全量 + Python/CUDA/torch/显卡/驱动版本 | 训练环境不可移植（venv 4.5 GB），可移植的是**精确的版本清单** |
| `logs/` | `_v6_train.log` … `_v8b_train.log`，共 10 份 | 复现失败时对照"正常应该长什么样"；里面也有失败尝试的记录 |
| `manifests/` | 每个数据集一份 `meta.json` + `SHARDS.md` | 数据不在仓库里，**生成参数**必须在，否则重建出来的不是同一份数据 |

---

## 环境

```bash
python -m venv .venv311
.venv311/Scripts/python -m pip install -r environment/requirements.txt
```

精确版本见 [`environment/ENVIRONMENT.md`](environment/ENVIRONMENT.md)（Python 3.11.9 / torch 2.11.0+cu128 / CUDA 12.8）。

PyTorch 版本取决于显卡：Blackwell（RTX 50 系）需要 cu128+，Hopper/Ada 用 cu124 即可。

---

## 完整流程

### 1. 生成数据（约 3 分钟，100k 条）

```bash
python gen_dataset.py --out ./shards --samples 100000 --ctx 4 --workers 12
```

每个样本包含：

| 字段 | 形状 | 内容 |
|---|---|---|
| `rgb_pq` | (n, 4, 72, 128) uint8 | 前 3 通道 PQ 编码 RGB，第 4 通道边缘强度 |
| `ctx` | (n, 45) float32 | 13 个显示/场景标量 + 16 个分块 + 16 档直方图 |
| `y` | (n, 72) float32 | 8 个曲线控制点（log2）+ 64 个局部强度（[-1,1]） |

**标签是怎么来的**（这是整条管线的核心）：

1. 教师曲线是**概率加权混合**，不是硬切换 —— 硬切换会让标签在场景类型边界不连续，实测把可达 L1 卡在 1.0 左右
2. SDR 单独加权偏向有真实 toe 的 filmic 曲线（`ue_aces` / `ue_uncched` / `ue_hp`），因为 1.0.0 版本在 SDR 下阴影偏亮 0.7 档
3. 局部参数用 **Durand 参考 + 局部曝光项** 做目标，在每个 4×4 tile 上解 4×4 最小二乘

### 2. 训练

```bash
# 从零开始
python train.py --shards ./shards --out ./run --size M --batch 256 --lr 6e-4 --minutes 45

# 热启动（推荐）：形状不匹配的张量会自动跳过
python train.py --shards ./shards --out ./run_ft --size M --batch 256 --lr 8e-5 \
                --minutes 40 --init-from ./run/best.pt

# 只训局部头，不动曲线（1.0.3 的做法）
python train.py --shards ./shards --out ./run_local --size M --batch 256 --lr 4e-4 \
                --minutes 30 --init-from ./run/best.pt --freeze-backbone
```

**踩过的坑，都写在代码注释里**：

| 现象 | 原因 | 解法 |
|---|---|---|
| 模型塌成常数曲线（预测 std 0.001） | LayerNorm 后接 tanh 头会饱和 | 头前加 LayerNorm + 从显示上下文直接加一条线性通路 |
| val 剧烈震荡（0.79 / 11.7） | fp16 autocast 下评估 | 评估固定 fp32 |
| 加局部头后曲线退化（0.32 → 0.54） | 局部损失占 72 个输出里的 64 | `--freeze-backbone` |
| 学习率 4e-4 微调反而变差 | OneCycle 峰值把模型踢出好盆地 | 微调用 6e-5 ~ 8e-5 |

### 3. 验证

```bash
python _verify.py ./run/best.pt ./shards
```

输出 L1、分位数、逐 knot 误差、预测/真值离散度比、单调性、`corr(knot0)`、局部参数误差与相关性。
**部署门槛**：单调性 ≥99%、离散度比 ≥0.5、`corr(knot0)` ≥0.5。

### 4. 导出

```bash
python export_onnx.py --ckpt ./run/best.pt --out ./export
```

脚本会检查：

- 算子是否都在 `NNERuntimeRDGHlsl` 支持列表内（**Concat / Pad 不支持**，所以用 MatMul 放置矩阵）
- ONNX 与 PyTorch 的数值一致性（正常 < 1e-5）
- 曲线单调性
- `DEPLOYABLE` 判定

产物 `ntmo_curve.onnx` + `ntmo_curve_report.json`。

### 5. 部署到插件

把 `.onnx` 拷到 `plugin/NeuralToneMapper/Content/Models/`，命名保持 `NTMO_M_v<N>.onnx`。
插件启动时**自动挑版本号最高的那个**，不需要改任何配置。

**契约必须同步**：`ntmo_pack.py` 的 `C_IN / C_OUT` 与
`plugin/.../Private/NTMOCurvePolicyNeural.h` 的 `NTMOInput::Channels / Outputs`
以及 `Shaders/Private/NTMOFeatures.usf` 的通道索引必须一致，否则插件喂给网络的是垃圾数据 —— 而且**不会报错**，只会得到一条看起来合理的错误曲线。

---

## A/B 验证

```powershell
pwsh -File ab/_ab_run.ps1 -OutDir E:\tmp\ab
```

它做的事：

1. 启动编辑器，Python harness 通过 `Content/Python/init_unreal.py` 哨兵文件加载
2. **固定关卡视口相机**到同一个位置
3. 等 TAA 和自动曝光收敛
4. 抓「插件开启」帧 → **通过文件握手**通知 harness 关闭插件 → 抓「UE 原生」帧

两次截图在同一个编辑器会话、同一个相机下，逐像素比较才有意义。

> 早期版本用固定帧数推进，结果 ON_READY 之后 0.9 秒才截图，而插件 0.5 秒时已被关闭 ——
> 两张图都是 UE 的 tonemapper，A/B 诚实地报告"完全相同"。

然后：

```bash
python ab/_ab_compare_same_session.py on.png off.png
```

输出逐通道差值、逐亮度带 dStops、削顶率、对比度比、逐像素相关。

---

## 当前模型（v8 / NTM 1.0.3）

| | |
|---|---|
| 结构 | 4 层卷积图像分支 + 上下文分支 + 融合 MLP |
| 参数 | 323,767（ONNX 1.3 MB） |
| 输入 | `[1, 49, 72, 128]` |
| 输出 | `[1, 72]`（8 曲线 + 64 局部） |
| 离线 L1 | 0.321（曲线）/ 0.188（局部） |
| 单调性 | 100% |
| 离散度比 | 0.994 |
| 数值误差 | 2.9e-06 |

## 已知的训练侧问题

1. **训练/验证差距约 3~4 倍**（train 0.09 / val 0.32）。合成数据分布有限，加大数据量比加大训练时长有效。
2. **局部参数离散度不足**（预测 std 0.256 vs 真值 0.373）。
3. **局部算子的标签目标可被回归量精确表达**，导致网络学到的接近常数解 —— 这是 1.0.2 "局部 AI 贡献接近零"的原因，1.0.3 用第 4 个通道（局部曝光）改善了一部分，但**标签设计仍是主要瓶颈**。
