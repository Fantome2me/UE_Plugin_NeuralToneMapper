# 复现指南 —— 从零到可部署的 ONNX

这份文档的目标：**在一台干净的机器上，从本仓库的代码和数据配方，产出和仓库里 `Content/Models/NTMO_M_v8.onnx` 等价的模型**。

按顺序做，每一步都有可校验的输出。

---

## 0. 你需要什么

| | 最低 | 本次训练实际用的 |
|---|---|---|
| GPU | 任意 CUDA 卡（≥6 GB 显存） | RTX 5060 Ti 16 GB（sm_120） |
| 磁盘 | 25 GB 空闲（数据 3.5 GB + 检查点 + torch） | — |
| 时间 | 数据 3 分钟 + 训练 45 分钟 + 导出 1 分钟 | — |
| Python | 3.11 | 3.11.9 |

网络受限的话，`registry.npmmirror.com` 的 PyTorch wheel 可以直接用（见 `environment/ENVIRONMENT.md`）。

---

## 1. 环境

```powershell
cd dev
python -m venv .venv311
.venv311\Scripts\python -m pip install --upgrade pip
.venv311\Scripts\python -m pip install -r environment\requirements.txt

# 自检：必须打印 True 和你的显卡名
.venv311\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**预期**：`True NVIDIA GeForce RTX 5060 Ti`（或你的卡）。

版本不一致也不用慌，但 `torch` 大版本别跨（2.11 → 2.9 可能改变默认 autocast 行为）。

---

## 2. 生成数据

### 2.1 先做 2000 样本的确定性自检（5 秒）

```powershell
.venv311\Scripts\python training\gen_dataset.py --out .\det_a --samples 2000 --ctx 4 --workers 2
.venv311\Scripts\python training\gen_dataset.py --out .\det_b --samples 2000 --ctx 4 --workers 6
```

```powershell
Get-FileHash .\det_a\shard_0000.npz -Algorithm SHA256
Get-FileHash .\det_b\shard_0000.npz -Algorithm SHA256
```

**必须完全相同**，且应等于：

```
AB6C5E54581AEE02734501795702E412850984B7793D846ED45BD0950B7E9294
```

> 如果两个哈希不同，说明你改动过 `gen_dataset.py` 的作业切分或采样顺序。
> 生成器的种子是固定的（`12345 + i*7919`），作业数固定 64 份，结果用 `imap` 有序收集 ——
> 这三件事共同保证「同样的命令行 → 同样的字节」。**进程数只影响速度，不影响内容。**

这个基准分片也可以直接下载核对：
[`reference_shard_2k.npz`](../../releases)（71 MB，SHA256 同上）。

### 2.2 生成正式训练集（约 3 分钟，3.5 GB）

```powershell
.venv311\Scripts\python training\gen_dataset.py --out .\shards_v3 --samples 100000 --ctx 4 --workers 12
```

**预期输出**：

```
[gen] target 100000 samples | 25000 images | 64 jobs | 12 workers -> ./shards_v3
[gen] ./shards_v3\shard_0000.npz  n=50008  (1843 MB rgb)
[gen] ./shards_v3\shard_0001.npz  n=49992  (1843 MB rgb)
[gen] DONE 100000 samples in 176.7s (566 samples/s)
[gen] meta.json written
```

**校验**（形状与标签分布）：

```powershell
.venv311\Scripts\python -c "import numpy as np, glob; z=np.load(sorted(glob.glob('shards_v3/*.npz'))[0]); print(z['y'].shape, z['ctx'].shape, z['rgb_pq'].shape); b=z['y'][:,8:].reshape(len(z['y']),16,4); print('band means', np.round(b.mean(axis=(0,1)),3))"
```

**预期**：

```
(50008, 72) (50008, 45) (50008, 4, 72, 128)
band means [ 0.308  0.408  0.443 -0.239]
```

四个 band 的含义是 `s_base / s_mid / s_high / s_low`。`s_low` 是负数是对的（见 `training/ntmo_local.py` 里 `TEACHER_EXPOSURE_GAIN=+0.35` 与 `T4 = clip(G3 - mean)` 的符号约定）。

---

## 3. 训练

### 3.1 从零训练（45 分钟）

```powershell
.venv311\Scripts\python training\train.py --shards .\shards_v3 --out .\run_v8 --size M `
    --batch 256 --lr 6e-4 --minutes 45 --workers 0 --no-compile
```

**预期**：`[train] DONE best_val=0.35±0.02 -> ./run_v8/best.pt`

### 3.2 冻结主干只训局部头（30 分钟，产出 1.0.3/1.0.4 用的模型）

这是**最终发布版的训练方式**：曲线已经好到不该再动，只学那 64 个局部强度。

```powershell
.venv311\Scripts\python training\train.py --shards .\shards_v3 --out .\run_v8b --size M `
    --batch 256 --lr 4e-4 --minutes 30 --workers 0 --no-compile `
    --init-from .\checkpoints\best_v7b.pt --freeze-backbone
```

**预期**：

```
[train] warm-started 49/55 tensors from .\checkpoints\best_v7b.pt; skipped 6 shape-mismatched
[train] backbone frozen; training only: ['head_local.weight','head_local.bias','ctx_local.weight','ctx_local.bias']
[train] DONE best_val=0.321±0.01 -> ./run_v8b/best.pt
```

> `skipped 6` 是正常的：局部头从 48 维扩到 64 维，`head_local` / `ctx_local` / 两个放置矩阵形状变了，其余张量全部继承。

### 3.3 三个检查点的来历

| 检查点 | 产物 | 训练方式 | 指标 |
|---|---|---|---|
| `checkpoints/best_v6b.pt` | `NTMO_M_v6.onnx`（NTM 1.0.1） | 从零，48 通道 / 8 输出 | L1 0.3601, std 0.977, corr 0.900 |
| `checkpoints/best_v7b.pt` | `NTMO_M_v7.onnx`（NTM 1.0.2） | 从零，49 通道 / 56 输出 | L1 0.3229, std 1.000, corr 0.916, local 0.528 |
| `checkpoints/best_v8b.pt` | `NTMO_M_v8.onnx`（NTM 1.0.3 / **1.0.4**） | 从 v7b 热启动 + 冻结主干，72 输出 | L1 0.3208, std 0.994, corr 0.914, local 0.764 |

---

## 4. 验证

```powershell
.venv311\Scripts\python training\_verify.py .\run_v8b\best.pt .\shards_v3
```

**预期**（这是判断"复现成功"的唯一标准）：

```
L1 (fp32) : MEAN 0.32x   MEDIAN 0.26x
local L1  : 0.18x
local corr: 0.7x
pred std  : [0.92 1.35 1.39 1.34 1.30 1.32 1.43 1.58]
true std  : [1.03 1.41 1.42 1.31 1.25 1.29 1.43 1.56]
std ratio : 0.99x     <- 低于 0.5 说明模型塌成常数，训练有问题
monotone  : 100.00%   <- 必须 100%
corr(knot0): 0.91x    <- 低于 0.5 说明网络没在用输入
```

**部署门槛**（`_verify.py` 会打印，但不会强制）：单调性 ≥99%、离散度比 ≥0.5、`corr(knot0)` ≥0.5。
三项都过，再往下走。

---

## 5. 导出 ONNX

```powershell
.venv311\Scripts\python training\export_onnx.py --ckpt .\run_v8b\best.pt --out .\export
```

**预期**：

```
[export] size=M params=326.6K epoch=1 best_val=0.32x
[export] ops: ['Add','Constant','Conv','Gather','Gemm','Identity','LayerNormalization','MatMul','Mul','ReduceMean','Relu','Sigmoid','Slice','Softmax','Tanh']
[export] numeric max err = 2.861e-06
[export] DEPLOYABLE = True
```

两个硬性条件：

1. **算子表里不能出现 `Concat` / `Pad`** —— UE 的 `NNERuntimeRDGHlsl` 后端不支持，会静默降级成错误的图
2. **`numeric max err` < 1e-4** —— PyTorch 与 ONNX 必须数值一致

### 5.1 命名与安装

把 `export/ntmo_curve.onnx` 改名为 `NTMO_M_v<N>.onnx`，放进：

```
Plugins/NeuralToneMapper/Content/Models/
```

插件启动时**自动挑版本号最高的那个**，不需要改任何配置。

> ⚠️ **契约必须同步**：`training/ntmo_pack.py` 的 `C_IN / C_OUT` 必须等于
> `Source/.../NTMOCurvePolicyNeural.h` 的 `NTMOInput::Channels / Outputs`，
> 且通道索引要和 `Shaders/Private/NTMOFeatures.usf` 一致。
> 不一致时**不会报错** —— 插件会喂给网络一段垃圾数据，然后得到一条看起来很合理的错误曲线。
> 这是本项目最难查的一类问题。

---

## 6. 训练要多久

| 阶段 | RTX 5060 Ti 16 GB | 备注 |
|---|---|---|
| 生成 100k 数据 | 3 分钟 | 12 线程，565 样本/秒 |
| 从零训练 45 分钟 | 45 分钟 | 1 epoch ≈ 1 分钟 |
| 冻结主干 30 分钟 | 30 分钟 | 只算局部头，速度快一点 |
| 验证 5000 样本 | <1 分钟 | |
| 导出 + 检查 | <1 分钟 | |

---

## 7. 已知偏差（诚实说明）

**仓库里的检查点，是用「修复确定性之前」的生成器产出的数据训练的。**

修复前后差异只有两点：

| | 修复前 | 修复后 |
|---|---|---|
| 作业切分 | `n_images_total // workers`（依赖进程数） | 固定 64 份（与进程数无关） |
| 结果收集 | `imap_unordered`（顺序随机） | `imap`（保持作业顺序） |

**采样逻辑、教师混合权重、标签公式一个字节都没改。** 所以用现在的生成器重建数据后：

- 数据分布、每张图的样本内容：相同
- 样本在分片里的顺序、以及因此决定的训练/验证划分：不同
- 训练出来的模型指标：**应当在 ±0.02 以内**，但不会逐位相同

如果你需要**逐位复现**某个检查点，只能用当时的数据。全部数据 3.5 GB，没法放进 git
（GitHub 单文件上限 100 MB，单个分片 1.78 GB），这也是为什么这里选择"确定性生成器 + 校验哈希"这条路：
**代码和数据配方能保证产出等价的模型，而不是保证产出同一个文件。**

要真正做到逐位复现，有两条路：

1. 用 Git LFS 传 3.5 GB 数据（免费额度 1 GB，不够，需要付费）
2. 把数据压成分片放到 Release（单个 1.78 GB，GitHub Release 附件上限 2 GB，可以传）

需要的话告诉我，我可以把 `shards_v3` 的两个分片作为 Release 附件传上去。

---

## 8. 常见失败

| 现象 | 原因 | 解法 |
|---|---|---|
| `pred std` 接近 0，`std ratio` < 0.2 | 模型塌成常数曲线 | 检查头部的 LayerNorm 是否还在；学习率是否过高 |
| `val` 在 0.8 / 11 之间剧烈跳 | fp16 下评估 | `_verify.py` 已固定 fp32；训练里 `evaluate()` 也要 fp32 |
| 加局部头后曲线明显退化 | 局部损失占 72 个输出里的 64，把主干拖走 | `--freeze-backbone` |
| 微调后指标变差 | 学习率过高，把模型踢出好的盆地 | 微调用 6e-5 ~ 8e-5，不要用 4e-4 以上 |
| 导出后 `DEPLOYABLE=False` | ONNX 里出现了 `Concat` | 检查是否有人把 MatMul 放置矩阵改回了 Concat |
| 插件加载模型后画面发灰/过曝 | 契约不一致，或忘了同步 `.usf` 的通道索引 | 见 5.1 的警告 |
