# 训练环境（生成这份记录时的实机状态）

这份文件记录 **NTM 模型实际训练时用的环境**。照着装就能复现。

## 硬件

`
NVIDIA GeForce RTX 5060 Ti, 576.88, 16311 MiB
`

## 软件

`
python=3.11.9
torch=2.11.0+cu128
numpy=2.4.6
cuda=12.8
cudnn=91900
gpu=NVIDIA GeForce RTX 5060 Ti
capability=(12, 0)
torch_cuda_available=True
OS: Microsoft Windows 10 专业版 build 19045
`

## 安装

`ash
python -m venv .venv311
.venv311\Scripts\python -m pip install --upgrade pip
.venv311\Scripts\python -m pip install -r requirements.txt
`

equirements.txt 是 pip freeze 的完整快照（17 个包），固定了 torch / numpy 等所有版本。

### 关于 CUDA

	orch==2.11.0+cu128 对应 CUDA 12.8 运行时。
Blackwell（RTX 50 系，sm_120）需要 CUDA 12.8 及以上的 torch 构建；更早的卡用 cu124 也可以。
PyTorch 官方 wheel 自带 CUDA 运行时，**不需要单独装 CUDA Toolkit**，只要显卡驱动够新。

## 数据规模参考

| 用途 | 样本数 | 生成时间（12 线程） | 磁盘 |
|---|---|---|---|
| 冒烟测试 | 2,000 | ~5 秒 | 71 MB |
| 正式训练 | 100,000 | ~3 分钟 | 3.5 GB |