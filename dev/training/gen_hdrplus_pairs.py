"""Build pair-supervised local-parameter shards from HDR+ 480p PNG/JPEG pairs."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ntmo_local import derive_local_params_from_reference
from ntmo_pack import H, W, pack_sample


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_pair(root, image_id):
    hdr_bgr = cv2.imread(str(root / "middle_480p" / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
    jpg_bgr = cv2.imread(str(root / "output_480p" / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
    if hdr_bgr is None or jpg_bgr is None or hdr_bgr.dtype != np.uint16:
        raise ValueError(f"missing/invalid pair {image_id}")
    if hdr_bgr.shape[:2] != jpg_bgr.shape[:2] or hdr_bgr.ndim != 3 or hdr_bgr.shape[2] != 3:
        raise ValueError(f"shape mismatch for {image_id}: {hdr_bgr.shape}, {None if jpg_bgr is None else jpg_bgr.shape}")
    hdr = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    srgb = cv2.cvtColor(jpg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return hdr, srgb_to_linear(srgb)


def pack_one(sess, hdr, reference):
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    stats = {
        "mean_lum": float(lum.mean()),
        "p9999_lum": float(np.percentile(np.max(hdr, axis=-1), 99.99)),
    }
    disp = {
        "peak_nits": 100.0, "fullframe_nits": 100.0, "min_nits": 0.1,
        "paper_nits": 100.0, "is_hdr": 0.0, "outdev": 0, "gamut": 0,
    }
    small = cv2.resize(hdr, (W, H), interpolation=cv2.INTER_AREA)
    img, ctx = pack_sample(small, stats, disp)
    scene = np.empty((1, 49, H, W), np.float32)
    scene[0, :4] = img.astype(np.float32) / 255.0
    scene[0, 4:] = ctx[:, None, None]
    output = sess.run(None, {sess.get_inputs()[0].name: scene})[0][0]
    knots = output[:8].astype(np.float32)
    local = derive_local_params_from_reference(hdr, reference, knots)
    y = np.zeros(72, np.float32)
    y[:8], y[8:] = knots, local
    return img, ctx, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.data)
    split_ids = {
        "train": read_ids(root / "train.txt"),
        "validation": read_ids(root / "valid.txt"),
        "test": read_ids(root / "test.txt"),
    }
    # The supplied validation IDs are a subset of test.txt; reserve them for validation.
    overlap = set(split_ids["validation"]) & set(split_ids["test"])
    split_ids["test"] = [i for i in split_ids["test"] if i not in overlap]
    if set(split_ids["train"]) & (set(split_ids["validation"]) | set(split_ids["test"])):
        raise ValueError("train split overlaps validation/test")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    if shape != [1, 49, H, W] or sess.get_outputs()[0].shape[-1] != 72:
        raise ValueError(f"unexpected base model contract: {shape}, {sess.get_outputs()[0].shape}")

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for split, ids in split_ids.items():
        imgs, ctxs, labels = [], [], []
        for n, image_id in enumerate(ids, 1):
            hdr, ref = load_pair(root, image_id)
            img, ctx, y = pack_one(sess, hdr, ref)
            imgs.append(img)
            ctxs.append(ctx)
            labels.append(y)
            if n % 50 == 0 or n == len(ids):
                print(f"[{split}] {n}/{len(ids)}", flush=True)
        outdir = Path(args.out) / split
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outdir / "shard_0000.npz",
            rgb_pq=np.stack(imgs).astype(np.uint8),
            ctx=np.stack(ctxs).astype(np.float32),
            y=np.stack(labels).astype(np.float32),
        )
        counts[split] = len(ids)

    meta = {
        "source": "HDR+ 480p paired middle_480p PNG / output_480p JPEG",
        "split_counts": counts,
        "validation_ids_are_reserved_from_test": len(overlap),
        "input_assumption": "16-bit PNG values treated as relative linear RGB; files carry no ICC/gAMA calibration",
        "reference_assumption": "JPEG decoded with the standard sRGB EOTF to display-linear values",
        "base_model": os.path.abspath(args.model),
        "display_context": {"output": "SDR", "peak_nits": 100, "black_nits": 0.1},
        "label_method": "preserve base model global knots; least-squares fit of 4x4x4 local LLF basis to paired JPEG luminance residual",
    }
    Path(args.out, "pair_training_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
