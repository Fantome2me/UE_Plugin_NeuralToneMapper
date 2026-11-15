"""
NTMO curve-prediction training.

Built for a hard wall-clock budget: pass --minutes and it trains until the
deadline, saving checkpoints, then exits with a usable model. Designed for an
H20 (96 GB, modest FLOPS) => use a big batch and let BF16 + channels_last carry
the throughput.

    # cloud (H20), 40 minute budget
    python train.py --shards /data/ntmo/shards --out /data/ntmo/run --size M \
                    --batch 1024 --minutes 40 --workers 10

    # local (5060 Ti), smaller model
    python train.py --shards ./shards --out ./run_s --size S --batch 128 --minutes 40
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ntmo_model import C_IN, KNOT_N, build_model, count_params
from ntmo_pack import assemble_tensor_torch

# Knots 3..6 are where a tone curve actually lives (mid grey -> shoulder). The
# deep-shadow and deep-highlight knots matter less and are noisier.
KNOT_W = torch.tensor([0.5, 0.5, 0.7, 1.0, 1.2, 1.5, 1.5, 1.2]).view(1, 8)
LOCAL_W = 0.60     # weight of the 48 local Laplacian strengths (v1.0.2)


class ShardDataset(Dataset):
    """Lazily materialises only the slice this split needs.

    The previous version concatenated EVERY shard into one array before slicing out
    the validation tail. That cost a full copy of the dataset (4.6 GB here) for the
    training split -- unavoidable -- but ALSO a second full 4.6 GB copy for the
    validation split, even though validation is only 5% of the data. Combined with
    the validation loader spawning >=2 worker processes (each pickling its own copy
    of the dataset under Windows spawn), a single training run sat at ~15 GB of
    resident memory. Slicing per shard as we load keeps peak memory at one shard.

    Validation order is ascending global index (the same order _mkvalcache.py and
    _verify.py use), so the three stay consistent.
    """

    def __init__(self, shards, train=True, seed=0, val_frac=0.05):
        self.train = train

        sizes = []
        for f in shards:
            with np.load(f) as d:
                sizes.append(d["rgb_pq"].shape[0])
        n = int(sum(sizes))

        idx = np.random.default_rng(seed).permutation(n)
        n_val = max(1, int(n * val_frac))
        wanted = idx[n_val:] if train else idx[:n_val]
        wanted.sort()          # ascending global index -> deterministic, cache-compatible

        offsets = np.cumsum([0] + sizes)
        rgb_l, ctx_l, y_l = [], [], []
        for k, f in enumerate(shards):
            lo, hi = int(offsets[k]), int(offsets[k + 1])
            m = (wanted >= lo) & (wanted < hi)
            if not m.any():
                continue
            local = wanted[m] - lo
            with np.load(f) as d:
                rgb_l.append(d["rgb_pq"][local])
                ctx_l.append(d["ctx"][local])
                y_l.append(d["y"][local])

        self.rgb = np.concatenate(rgb_l, axis=0)
        self.ctx = np.concatenate(ctx_l, axis=0)
        self.y = np.concatenate(y_l, axis=0)
        rgb_l.clear()
        ctx_l.clear()
        y_l.clear()

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, i):
        rgb = self.rgb[i]
        if self.train and np.random.random() < 0.5:
            rgb = rgb[:, :, ::-1].copy()
        return (
            torch.from_numpy(rgb),
            torch.from_numpy(self.ctx[i]),
            torch.from_numpy(self.y[i]),
        )


def collate(batch):
    rgb = torch.stack([b[0] for b in batch]).to(torch.uint8)
    ctx = torch.stack([b[1] for b in batch])
    y = torch.stack([b[2] for b in batch])
    x = assemble_tensor_torch(rgb, ctx)
    return x.contiguous(memory_format=torch.channels_last), y


def make_loader(shards, train, batch, workers):
    ds = ShardDataset(shards, train=train)
    return DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=workers,
        collate_fn=collate, pin_memory=True, drop_last=train,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
    ), len(ds)


def weighted_loss(pred, target):
    l_k = (torch.abs(pred[:, :8] - target[:, :8]) * KNOT_W.to(pred.device)).mean()

    # Smoothness: discourage kinks that would show up as banding in the LUT.
    d2 = pred[:, 2:8] - 2 * pred[:, 1:7] + pred[:, :6]
    l_s = (d2 * d2).mean()

    # Channels 8..55 are the 48 local Laplacian strengths (4x4 tiles x 3 bands) that
    # v1.0.2 introduced. They live in [-1,1] while the knots live in log2 (scale ~10),
    # so the weight keeps the local term visible without letting it dominate.
    l_local = torch.zeros((), device=pred.device)
    if pred.shape[1] > 8 and target.shape[1] > 8:
        l_local = F.smooth_l1_loss(pred[:, 8:], target[:, 8:])

    return l_k + 0.01 * l_s + LOCAL_W * l_local, l_k, l_local


@torch.no_grad()
def evaluate(model, loader, device, max_batches=40):
    """Validation in FP32, deliberately WITHOUT bf16 autocast.

    Training runs under autocast for speed, but measuring under autocast produced
    nonsense: the same checkpoint scored 0.79 on one epoch and 11.71 on the next,
    while a plain fp32 re-measure of that very checkpoint gave mean 0.762 /
    median 0.498 / max 4.07 over all validation samples. The spikes were a
    measurement artifact, and they made best-checkpoint selection a lottery.
    """
    model.eval()
    tot, n = 0.0, 0
    local_tot, local_n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = model(x).float()
        tot += F.l1_loss(p[:, :8], y[:, :8], reduction="sum").item()
        n += y.shape[0] * 8
        if p.shape[1] > 8:
            local_tot += F.l1_loss(p[:, 8:], y[:, 8:], reduction="sum").item()
            local_n += y.shape[0] * (p.shape[1] - 8)
        if i + 1 >= max_batches:
            break
    model.train()
    local_err = local_tot / max(local_n, 1)
    if local_n:
        print(f"[eval] local_param_l1={local_err:.4f}")
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="M", choices=["S", "M", "L"])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = auto by batch size")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-from", default="", help="warm-start weights only; fresh optimiser and LR schedule")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train ONLY the local Laplacian heads; keeps the curve exactly "
                         "as good as the warm-start checkpoint. Use when adding a new "
                         "local head must not disturb an already-good curve.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shards = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"no shards under {args.shards}")

    print(f"[train] device={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)
    print(f"[train] {len(shards)} shards", flush=True)

    train_loader, n_train = make_loader(shards, True, args.batch, args.workers)
    # Validation must honour --workers too. The old `max(2, workers // 2)` forced two
    # worker processes even at --workers 0, and under Windows spawn each one gets its
    # own pickled copy of the dataset -- 9 GB of extra resident memory for a 5% split.
    val_loader, n_val = make_loader(shards, False, max(64, args.batch // 2), max(0, args.workers // 2))
    print(f"[train] train={n_train} val={n_val}", flush=True)

    model = build_model(args.size).to(device)
    model = model.to(memory_format=torch.channels_last)
    print(f"[train] size={args.size} params={count_params(model)/1e3:.1f}K", flush=True)

    lr = args.lr if args.lr > 0 else 3e-3 * (args.batch / 256) ** 0.5
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=False)   # bf16 needs no scaling

    steps_per_epoch = max(1, n_train // args.batch)
    # OneCycle needs a realistic total_steps. Under-estimating it decays the LR to
    # ~0 long before the deadline, which manifests as a flat loss plateau while the
    # GPU keeps churning. Measured ~12 it/s at batch 192 on a 5060 Ti (and ~20/s on an H20), so 12/s
    # is a deliberate (slightly conservative) upper estimate.
    est_steps_per_sec = 12.0
    total_steps = int(max(steps_per_epoch, args.minutes * 60.0 * est_steps_per_sec))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.15, div_factor=10.0, final_div_factor=20.0)

    start_epoch, best = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best", float("inf"))
        print(f"[train] resumed from {args.resume} (epoch {start_epoch}, best {best:.4f})", flush=True)

    # Warm start from WEIGHTS ONLY, with a fresh optimiser and a fresh LR schedule.
    # Needed because the OneCycle peak is what keeps knocking the model out of its
    # good basin: on mixture labels the run hit L1 0.565 / corr 0.902 by epoch 5 and
    # then degraded to 1.637 by epoch 12 as the LR climbed. Fine-tuning from the good
    # checkpoint with a low peak LR is the cheap way to keep improving.
    if args.init_from and os.path.isfile(args.init_from):
        ck = torch.load(args.init_from, map_location=device)
        src = ck["model"]
        dst = model.state_dict()
        # Shape-aware partial load. NTM 1.0.2 changed the contract (image input grew
        # from 3 to 4 channels for the edge map, output from 8 to 56), so a plain
        # load_state_dict would raise on the first conv and on the new local heads.
        # Everything that still matches -- almost the whole trunk plus the curve head --
        # is transferred, which is exactly the warm start we want: the curve starts from
        # a known-good place and training can spend its budget on the local head.
        keep, skipped = {}, []
        for k, v in src.items():
            if k in dst and tuple(dst[k].shape) == tuple(v.shape):
                keep[k] = v
            else:
                skipped.append(k)
        dst.update(keep)
        model.load_state_dict(dst)
        print(f"[train] warm-started {len(keep)}/{len(src)} tensors from {args.init_from} "
              f"(its epoch {ck.get('epoch')}, val {ck.get('best')}); "
              f"skipped {len(skipped)} shape-mismatched", flush=True)

    # Freeze the backbone when the goal is "learn the new local head without losing the
    # curve". Measured motivation: adding the 64-wide local head and training everything
    # at lr 8e-5 pushed knot val from 0.32 to 0.54, because the local loss (48+ of the
    # outputs) drags the shared trunk away from the curve optimum.
    if args.freeze_backbone:
        trainable = []
        for name, prm in model.named_parameters():
            keep = name.startswith("head_local") or name.startswith("ctx_local")
            prm.requires_grad_(keep)
            if keep:
                trainable.append(name)
        print(f"[train] backbone frozen; training only: {trainable}", flush=True)
    if not args.no_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune-no-cudagraphs")
            print("[train] torch.compile enabled", flush=True)
        except Exception as e:
            print(f"[train] compile skipped: {e}", flush=True)

    t0 = time.time()
    deadline = t0 + args.minutes * 60
    log = []
    step = 0

    for epoch in range(start_epoch, 10_000):
        run, n_seen = 0.0, 0
        ep_t0 = time.time()

        for x, y in train_loader:
            if time.time() > deadline:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)

            loss, l_k, l_local = weighted_loss(pred.float(), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass

            run += l_k.item() * y.shape[0]
            n_seen += y.shape[0]
            step += 1

            if step % 50 == 0:
                r = n_seen / max(time.time() - ep_t0, 1e-6)
                eta = (deadline - time.time()) / 60.0
                print(f"[train] ep{epoch} step{step} lr={opt.param_groups[0]['lr']:.2e} "
                      f"knot_l1={run/max(n_seen,1):.4f} {r:.0f} samp/s eta={eta:.1f}min", flush=True)

        ep_loss = run / max(n_seen, 1)
        val = evaluate(model, val_loader, device)
        elapsed = (time.time() - t0) / 60.0
        print(f"[train] === epoch {epoch} done | knot_l1={ep_loss:.4f} val={val:.4f} "
              f"| {elapsed:.1f} min", flush=True)

        log.append({"epoch": epoch, "train_knot_l1": ep_loss, "val_knot_l1": val,
                    "minutes": round(elapsed, 2), "steps": step})

        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        ck = {"model": raw.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
              "best": min(best, val), "size": args.size, "c_in": C_IN, "knot_n": KNOT_N}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if val < best:
            best = val
            torch.save(ck, os.path.join(args.out, "best.pt"))
            print(f"[train] new best {best:.4f}", flush=True)

        with open(os.path.join(args.out, "train_log.json"), "w") as f:
            json.dump({"log": log, "best_val": best, "params": count_params(raw),
                       "size": args.size, "batch": args.batch, "lr": lr}, f, indent=2)

        if time.time() > deadline:
            print("[train] deadline reached", flush=True)
            break

    print(f"[train] DONE best_val={best:.4f} -> {args.out}/best.pt", flush=True)


if __name__ == "__main__":
    main()
