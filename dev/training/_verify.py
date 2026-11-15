"""训练成果验收：给定 checkpoint，输出可信的 fp32 指标 + 健康度体检。

用法:  python _verify.py [ckpt] [shards_dir]
默认:  ckpt = ./run_local/last.pt     shards_dir = ./shards
"""
import os, sys, glob
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ntmo_pack import assemble_tensor_torch
from ntmo_model import build_model

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "run_local", "last.pt")
SHARDS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "shards")

# Prefer the small pre-extracted validation cache: reading the 4 GB shards on every
# check made a single acceptance pass take minutes. _mkvalcache.py writes this file
# using exactly the same split as train.py.
cache = os.path.join(HERE, "_valcache.npz")
if os.path.exists(cache) and len(sys.argv) <= 2:
    d = np.load(cache)
    rgb, ctx, y = d["rgb_pq"], d["ctx"], d["y"]
    sel = np.arange(rgb.shape[0])
    n = rgb.shape[0]
    print(f"[verify] using cache ({n} samples)")
else:
    fs = sorted(glob.glob(os.path.join(SHARDS, "shard_*.npz")))
    if not fs:
        raise SystemExit(f"no shards under {SHARDS}")
    rgb = np.concatenate([np.load(f)["rgb_pq"] for f in fs], 0)
    ctx = np.concatenate([np.load(f)["ctx"] for f in fs], 0)
    y = np.concatenate([np.load(f)["y"] for f in fs], 0)
    n = rgb.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    n_val = max(1, int(n * 0.05))
    sel = idx[:n_val]

ck = torch.load(CKPT, map_location="cpu")
model = build_model(ck.get("size", "M"))
model.load_state_dict(ck["model"], strict=False)
model.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)

preds = []
with torch.no_grad():
    for s in range(0, len(sel), 256):
        j = sel[s:s + 256]
        x = assemble_tensor_torch(torch.from_numpy(rgb[j]), torch.from_numpy(ctx[j]))
        preds.append(model(x.to(dev).contiguous(memory_format=torch.channels_last)).float().cpu().numpy())
P_all = np.concatenate(preds, 0)
Y_all = y[sel]
# v1.0.2 emits 56 values: 8 curve knots + 48 local Laplacian strengths. The knot
# metrics below must only look at the first 8 or "monotone" would be meaningless.
P = P_all[:, :8]
Y = Y_all[:, :8]
E = np.abs(P - Y)
per = E.mean(1)

print("=" * 68)
print(f"ckpt      : {os.path.basename(os.path.dirname(CKPT))}/{os.path.basename(CKPT)}")
print(f"epoch     : {ck.get('epoch')}   reported_val: {ck.get('best')}")
print(f"val_n     : {len(sel)} (of {n})")
print("-" * 68)
print(f"L1 (fp32) : MEAN {per.mean():.4f}   MEDIAN {np.median(per):.4f}")
print(f"quantiles : p75 {np.percentile(per,75):.3f}  p95 {np.percentile(per,95):.3f}  max {per.max():.3f}")
print(f"per-knot  : {np.round(E.mean(0), 3)}")
if P_all.shape[1] > 8:
    PL = P_all[:, 8:]
    YL = Y_all[:, 8:]
    print(f"local L1  : {np.abs(PL - YL).mean():.4f}  (48 learned strengths, range [-1,1])")
    print(f"local std : pred {PL.std():.3f}  true {YL.std():.3f}")
    print(f"local corr: {np.corrcoef(PL.ravel(), YL.ravel())[0,1]:.4f}")
print("-" * 68)
print("HEALTH")
print(f"  pred std : {np.round(P.std(0), 3)}")
print(f"  true std : {np.round(Y.std(0), 3)}")
ratio = P.std(0).mean() / max(Y.std(0).mean(), 1e-9)
print(f"  std ratio: {ratio:.3f}   (1.0 = matches target spread; <0.2 = collapsed)")
mono = float((np.diff(P, axis=1) > 0).mean())
print(f"  monotone : {mono*100:.2f}% of samples strictly increasing")
corr = np.corrcoef(P[:, 0], Y[:, 0])[0, 1]
print(f"  corr(knot0): {corr:.4f}   (>0.9 good, ~0 = not using input)")
print("-" * 68)
print("BUCKETS by true knot0 (dark -> bright)")
q = np.quantile(Y[:, 0], [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    m = (Y[:, 0] >= q[i]) & (Y[:, 0] <= q[i + 1])
    if m.sum() == 0:
        continue
    print(f"  [{q[i]:7.2f},{q[i+1]:7.2f}] n={m.sum():5d} err={per[m].mean():6.3f} "
          f"pred_k0={P[m,0].mean():7.2f} true_k0={Y[m,0].mean():7.2f}")
print("=" * 68)
