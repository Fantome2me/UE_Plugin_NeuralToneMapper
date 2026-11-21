"""
Dataset generator.

Generates synthetic HDR scenes, samples display contexts, runs the teacher
ensemble and fits the 8-knot curve label. Pure numpy -> a 21-core box saturates
without touching the GPU, and there is no download step.

Usage:
    python gen_dataset.py --out /data/ntmo/shards --samples 1000000 --workers 20
    python gen_dataset.py --smoke

Throughput reference: ~440 samples/s on 2 slow cores, so ~4000/s on 20 cores.
1M samples ~= 4 min of generation and ~28 GB on disk.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ntmo_local import derive_local_params
from ntmo_pack import C_IN, C_CTX, C_OUT, H, W, make_label, pack_sample
from ntmo_synth import sample_display, synth_hdr
from ntmo_teachers import TEACHERS, TEACHER_WEIGHTS, knots_dense

TEACHER_NAMES = list(TEACHERS.keys())
_BASE_P = np.array([TEACHER_WEIGHTS[n] for n in TEACHER_NAMES], dtype=np.float64)
_BASE_P /= _BASE_P.sum()
IDX = {n: i for i, n in enumerate(TEACHER_NAMES)}

# Scene-conditioned teacher preference. This is where content adaptivity enters
# the label: a low-key night scene should lean on shadow-lifting operators, a
# high-key daylight scene on highlight-rolling operators. The network has to infer
# which regime it is in from the image, which is exactly the skill we want.
_SCENE_BOOST = {
    "lowkey":   {"analytic": 2.2, "hable": 1.6, "ue_legendary": 0.0},
    "highkey":  {"ue_aces": 2.2, "ue_hp": 1.6, "clip": 0.3},
    "specular": {"ue_uncharted": 2.4, "ue_aces": 2.0, "reinhard": 0.4},
    "marquee":  {"analytic": 2.4, "ue_hp": 1.3},
    "normal":   {},
}

# SDR-specific correction, applied on top of the scene boost.
#
# Measured against Unreal's own tonemapper on the same frame (same-camera A/B, SDR
# 100 nit panel), the v5 curve rendered shadows 0.7-0.8 stops brighter than Unreal and
# about 10% less local contrast in bright areas. The cause is in the teacher mix, not
# the network: "analytic" carries a deliberate +0.20 stop toe LIFT, and "hable" also
# lifts the bottom, so an SDR label pulled toward them has a bright toe by
# construction. Unreal's renderer uses an ACES-family curve with a real toe.
#
# SDR is where artists judge the image and where the reference is Unreal's own output,
# so SDR labels are now biased toward the filmic operators with a proper toe, and away
# from the toe-lifting ones. HDR keeps the full mix, because there the extra shadow
# information is genuinely usable and the panel's black level is far lower.
_SDR_BOOST = {
    "ue_aces": 2.6,
    "ue_hp": 2.0,
    "ue_uncharted": 1.6,
    "analytic": 0.35,
    "hable": 0.45,
    "ue_legendary": 0.8,
}


def _scene_teacher_probs(kind, is_hdr):
    p = _BASE_P.copy()
    for name, mul in _SCENE_BOOST.get(kind, {}).items():
        if name in IDX and mul > 0.0:
            p[IDX[name]] *= mul
        elif name in IDX:
            p[IDX[name]] *= max(mul, 1e-3)

    if is_hdr == 0.0:
        for name, mul in _SDR_BOOST.items():
            if name in IDX:
                p[IDX[name]] *= mul

    return p / p.sum()


def _teacher_ctx(stats, disp, rng):
    """
    Display context as the TEACHER sees it.

    IMPORTANT: everything here is DETERMINISTIC given (stats, disp). Earlier
    versions randomised the working peak / abl_awareness / reinhard_white per
    sample, which made the label stochastic: the same (image, display) pair could
    carry several different target curves. The network then cannot fit the data
    and val plateaus while train keeps dropping -- which is exactly what happened
    (L model: train 0.645, val 0.755). Labels must be a function of the input.

    The content-adaptive working peak is the interesting part: if the scene's
    brightest 0.01% is only 250 nits, dragging it to a 1000 nit panel wastes range
    and flattens the image. Real tone mapping clamps the working peak to what the
    content actually contains, and the network must infer that from the image.
    """
    peak = float(disp["peak_nits"])
    content_peak_nits = float(stats["p9999_lum"]) * 100.0
    eff_peak = float(np.clip(content_peak_nits * 1.6, 80.0, peak))

    abl = 1.0 - float(disp["fullframe_nits"]) / max(peak, 1.0) if disp["is_hdr"] > 0 else 0.0
    eff_full = eff_peak * float(np.clip(1.0 - abl * 0.8, 0.3, 1.0))

    ctx = dict(disp)
    ctx["peak_nits"] = eff_peak
    ctx["fullframe_nits"] = eff_full
    # Deterministic, derived from the panel rather than drawn from an RNG.
    ctx["abl_awareness"] = 0.4 + 0.6 * float(np.clip(abl, 0.0, 1.0))
    ctx["reinhard_white"] = 1.5 + 6.5 * float(np.clip(eff_peak / max(peak, 1.0), 0.0, 1.0))
    return ctx


def _worker(job):
    seed, n_images, n_ctx = job
    rng = np.random.default_rng(seed)

    rgbs, ctxs, ys = [], [], []
    for _ in range(n_images):
        hdr, stats = synth_hdr(rng)

        # Teacher is a DETERMINISTIC function of the scene regime. The network's
        # job is to infer the regime from the image and emit the matching curve --
        # that is a learnable, content-adaptive mapping. Sampling the teacher at
        # random per image made the label stochastic and killed convergence.
        # Teacher assignment: probability-weighted MIXTURE, not argmax.
        #
        # Hard argmax picked one of 4 operator curves per scene kind, which made the
        # target piecewise-constant in kind. Measured consequences:
        #   * features identify the teacher only 69.7% of the time (chance 25%), so
        #     ~30% of the label variance was unlearnable by construction;
        #   * corr(pred knot0, true knot0) was capped at ~0.65 and L1 at ~1.0
        #     regardless of architecture or training time.
        # A convex blend of the operators weights each by how plausible its scene
        # regime is, so a misjudged scene costs a little instead of a lot, and the
        # curve moves continuously between regimes -- which also removes the visual
        # "pop" that hard switching would cause at runtime.
        for _ in range(n_ctx):
            disp = sample_display(rng)
            if disp["is_hdr"] == 0.0:
                disp["fullframe_nits"] = disp["peak_nits"]

            # The teacher mixture now depends on the display too (SDR is biased toward
            # filmic curves with a real toe), so it has to be resolved per context
            # rather than once per image.
            probs = _scene_teacher_probs(stats["kind"], disp["is_hdr"])
            keep = [i for i in range(len(TEACHER_NAMES)) if probs[i] >= 0.05]
            if not keep:
                keep = [int(np.argmax(probs))]
            wmix = np.asarray([probs[i] for i in keep], dtype=np.float64)
            wmix = wmix / wmix.sum()
            tmix = [TEACHERS[TEACHER_NAMES[i]] for i in keep]

            tctx = _teacher_ctx(stats, disp, rng)
            knots = np.zeros(8, dtype=np.float64)
            ok = False
            for wi, t in zip(wmix, tmix):
                kk = knots_dense(t, tctx)
                if kk is None:
                    continue
                knots += wi * kk
                ok = True
            if not ok:
                continue
            # A convex combination of strictly increasing knot sequences is itself
            # strictly increasing, so monotonicity survives the blend.
            if not np.all(np.diff(knots) > 0):
                continue

            rgb_u8, ctx_v = pack_sample(hdr, stats, disp)
            rgbs.append(rgb_u8)
            ctxs.append(ctx_v)
            # Learned LLF parameter map: best-fit per-tile band strengths against a
            # Durand reference (see ntmo_local.py). Deterministic given the image and
            # its curve, so the label is a function of the input.
            local = derive_local_params(hdr, knots)
            ys.append(make_label(knots, local))

    if not rgbs:
        return None
    return (np.stack(rgbs), np.stack(ctxs).astype(np.float32), np.stack(ys).astype(np.float32))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./shards")
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--ctx", type=int, default=4, help="display contexts per image")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--shard", type=int, default=50000, help="samples per shard file")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.samples, args.workers, args.shard = 2000, 2, 1000
        args.out = args.out + "_smoke"

    os.makedirs(args.out, exist_ok=True)

    per_image = max(1, args.ctx)
    n_images_total = int(np.ceil(args.samples / per_image))
    # Job partition must NOT depend on --workers.
    #
    # It used to be n_images_total // workers, which meant the same command line produced
    # different data on a 4-core and on a 16-core machine: the partition changed, so the
    # per-job seeds changed, so the samples changed. That silently breaks "clone the repo
    # and regenerate the dataset" -- you get a different dataset than the one the shipped
    # checkpoints were trained on. Partition into a fixed number of jobs instead, and let
    # the worker count only affect how fast they are executed.
    JOBS_TARGET = 64
    jobs_per_job = max(1, int(np.ceil(n_images_total / JOBS_TARGET)))

    jobs, remaining, i, seed0 = [], n_images_total, 0, 12345
    while remaining > 0:
        take = min(jobs_per_job, remaining)
        jobs.append((seed0 + i * 7919, take, per_image))
        remaining -= take
        i += 1

    print(f"[gen] target {args.samples} samples | {n_images_total} images | "
          f"{len(jobs)} jobs | {args.workers} workers -> {args.out}", flush=True)

    t0 = time.time()
    shard_idx, total = 0, 0
    buf_rgb, buf_ctx, buf_y = [], [], []

    def flush():
        nonlocal shard_idx, buf_rgb, buf_ctx, buf_y
        if not buf_rgb:
            return
        rgb = np.concatenate(buf_rgb, axis=0)
        ctx = np.concatenate(buf_ctx, axis=0)
        y = np.concatenate(buf_y, axis=0)
        path = os.path.join(args.out, f"shard_{shard_idx:04d}.npz")
        np.savez(path, rgb_pq=rgb, ctx=ctx, y=y)
        print(f"[gen] {path}  n={rgb.shape[0]}  "
              f"({rgb.nbytes / 1e6:.0f} MB rgb)", flush=True)
        shard_idx += 1
        buf_rgb, buf_ctx, buf_y = [], [], []

    seen = 0
    with Pool(args.workers) as pool:
        # imap (ordered), not imap_unordered: the order in which results are appended
        # decides the sample order inside each shard file, and therefore which samples
        # land in the validation split. Unordered collection made two runs of the same
        # command produce differently-ordered shards.
        for res in pool.imap(_worker, jobs):
            if res is None:
                continue
            rgb, ctx, y = res
            buf_rgb.append(rgb); buf_ctx.append(ctx); buf_y.append(y)
            seen += rgb.shape[0]
            total += rgb.shape[0]
            if seen >= args.shard:
                flush(); seen = 0
            if total >= args.samples:
                break
    flush()

    dt = time.time() - t0
    rate = total / max(dt, 1e-6)
    print(f"[gen] DONE {total} samples in {dt:.1f}s ({rate:.0f} samples/s)", flush=True)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "samples": int(total), "c_in": C_IN, "c_ctx": C_CTX, "c_out": C_OUT,
            "h": H, "w": W, "shards": shard_idx, "teachers": TEACHER_NAMES,
            "base_teacher_weights": _BASE_P.tolist(),
            "scene_teacher_boost": _SCENE_BOOST,
            "gen_seconds": round(dt, 2), "samples_per_sec": round(rate, 1),
        }, f, indent=2)
    print("[gen] meta.json written", flush=True)


if __name__ == "__main__":
    main()
