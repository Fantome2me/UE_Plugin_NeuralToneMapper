"""Bake the shipped model into standalone .cube LUTs.

This is a faithful CPU port of Shaders/Private/NTMOCommon.ush + NTMOLutBake.usf. It
exists so the published model can be inspected and reused outside Unreal: the same
327K-parameter network, the same monotone cubic curve, the same MaxRGB tone map, the
same PQ-indexed grid -- just written to a file instead of a GPU texture.

Grid indexing matches the shader exactly:

    index  ->  PQ (ST-2084)  ->  linear nits / 100  ->  * exposure
    linear ->  NTMO_ToneMapMaxRGB  ->  clamp to [minNits, peakNits]  ->  linear display

The LUT stores LINEAR DISPLAY values, not encoded ones, because the tonemapper applies
the local Laplacian correction as a multiplicative exp2() in display space before the
transfer function. A .cube consumer that expects encoded values must apply sRGB/PQ
itself -- that is documented in the generated JSON.
"""

import argparse
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# The shader constants, copied from NTMOCommon.ush so they cannot drift silently.
# ---------------------------------------------------------------------------
PQ_M1 = 0.1593017578125
PQ_M2 = 78.84375
PQ_C1 = 0.8359375
PQ_C2 = 18.8515625
PQ_C3 = 18.6875
KNOT_COUNT = 8
KNOT_X0 = -10.0
KNOT_X1 = 4.0


def linear_to_pq(lin):
    y = np.clip(lin / 10000.0, 0.0, 1.0)
    ym1 = np.power(y, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * ym1) / (1.0 + PQ_C3 * ym1), PQ_M2)


def pq_to_linear(pq):
    np_ = np.power(np.clip(pq, 0.0, 1.0), 1.0 / PQ_M2)
    num = np.maximum(np_ - PQ_C1, 0.0)
    den = np.maximum(PQ_C2 - PQ_C3 * np_, 1e-6)
    return np.power(num / den, 1.0 / PQ_M1) * 10000.0


def eval_curve_log2(x, knots):
    """Fritsch-Carlson monotone cubic, identical to NTMO_EvalCurveLog2."""
    step = (KNOT_X1 - KNOT_X0) / (KNOT_COUNT - 1)
    t = np.clip((x - KNOT_X0) / step, 0.0, KNOT_COUNT - 1)
    i = np.minimum(np.floor(t).astype(np.int32), KNOT_COUNT - 2)
    u = t - i

    def k(idx):
        return knots[np.clip(idx, 0, KNOT_COUNT - 1)]

    y0 = k(i)
    y1 = k(i + 1)
    ym = k(i - 1)
    yp = k(i + 2)

    m0 = (y1 - ym) / (2.0 * step)
    m1 = (yp - y0) / (2.0 * step)

    delta = (y1 - y0) / step
    flat = np.abs(delta) < 1e-6
    a = np.where(flat, 0.0, m0 / np.where(flat, 1.0, delta))
    b = np.where(flat, 0.0, m1 / np.where(flat, 1.0, delta))
    s = a * a + b * b
    scale = np.where(s > 9.0, 3.0 / np.sqrt(np.maximum(s, 1e-12)), 1.0)
    m0 = np.where(flat, 0.0, np.where(s > 9.0, scale * a * delta, m0))
    m1 = np.where(flat, 0.0, np.where(s > 9.0, scale * b * delta, m1))

    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * y0 + h10 * step * m0 + h01 * y1 + h11 * step * m1


def tonemap_max_rgb(rgb, knots, saturation, gamut, highlight_desat, black_lift, peak_nits, min_nits):
    """Hue-preserving MaxRGB tone map, identical to NTMO_ToneMapMaxRGB."""
    max_c = np.maximum(np.max(rgb, axis=-1), 1e-8)
    out_log2 = eval_curve_log2(np.log2(max_c), knots)
    out_lin = np.exp2(out_log2)

    peak_lin = max(peak_nits * 0.01, 1e-4)
    floor_lin = min_nits * 0.01
    out_lin = np.clip(out_lin, floor_lin, peak_lin)

    ratio = (out_lin / max_c)[..., None]
    out = rgb * ratio

    lum = np.sum(out * np.array([0.2126, 0.7152, 0.0722]), axis=-1)

    desat_start = 0.60 * max(peak_nits, 1.0)
    desat_span = max(max(peak_nits, 1.0) - desat_start, 1.0)
    amount = np.clip(((out_lin * 100.0) - desat_start) / desat_span, 0.0, 1.0) * highlight_desat
    out = out + (lum[..., None] - out) * np.clip(amount, 0.0, 1.0)[..., None]

    if gamut > 0.0:
        lum2 = np.sum(out * np.array([0.2126, 0.7152, 0.0722]), axis=-1)[..., None]
        out = out + (lum2 - out) * min(gamut, 1.0)

    if abs(saturation - 1.0) > 1e-4:
        lum3 = np.sum(out * np.array([0.2126, 0.7152, 0.0722]), axis=-1)[..., None]
        out = lum3 + (out - lum3) * saturation

    return np.maximum(out + black_lift, 0.0)


def bake_cube(knots, size=33, exposure=1.0, saturation=1.08, gamut=0.0,
              highlight_desat=0.05, black_lift=0.0, peak_nits=100.0, min_nits=0.05):
    """Bake a size^3 LUT. Returns (rgb[Z,Y,X,3] linear display, knots after clamping)."""
    log_peak = np.log2(max(peak_nits * 0.01, 1e-4))
    log_floor = np.log2(max(min_nits * 0.01, 1e-6)) + 0.01
    k = np.clip(np.asarray(knots, dtype=np.float64), log_floor, log_peak)

    axis = np.linspace(0.0, 1.0, size)
    # The shader maps grid index -> PQ -> linear (nits) / 100.
    lin_axis = pq_to_linear(axis) * 0.01

    r, g, b = np.meshgrid(lin_axis, lin_axis, lin_axis, indexing="ij")
    scene = np.stack([r, g, b], axis=-1) * exposure

    mapped = tonemap_max_rgb(scene, k, saturation, gamut, highlight_desat, black_lift, peak_nits, min_nits)
    return mapped, k


def write_cube(path, lut, title):
    """.cube is R-fastest: the first axis of our array must vary slowest."""
    size = lut.shape[0]
    flat = lut.reshape(-1, 3)  # already r-major because of indexing='ij'
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for c in flat:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def sample_knots(onnx_path, shards_dir, n=2000):
    """Run the shipped ONNX on real dataset samples and return knots + labels.

    The exported graph takes ONE tensor, [B, 49, 72, 128]: the 4 image channels
    followed by the 45 context scalars broadcast over the spatial plane. That layout is
    what `assemble_tensor_torch` builds for training and what NTMOFeatures.usf builds on
    the GPU, so the same numbers reach the network in all three places.
    """
    import onnxruntime as ort

    files = sorted(f for f in os.listdir(shards_dir) if f.endswith(".npz"))
    z = np.load(os.path.join(shards_dir, files[0]))
    img = z["rgb_pq"][:n].astype(np.float32) / 255.0          # [n,4,72,128]
    ctx = z["ctx"][:n].astype(np.float32)                     # [n,45]

    b, c, h, w = img.shape
    spatial = [ctx[:, i].reshape(b, 1, 1, 1) * np.ones((1, 1, h, w), np.float32)
               for i in range(ctx.shape[1])]
    tensor = np.concatenate([img] + spatial, axis=1).astype(np.float32)
    assert tensor.shape[1] == 49, tensor.shape

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    # The exported graph has a fixed batch of 1 (the engine always feeds one view), so
    # the loop has to be here rather than a single batched call.
    knots = np.empty((b, 8), np.float32)
    for i in range(b):
        knots[i] = sess.run(None, {inp: tensor[i:i + 1]})[0][0, :8]
    return knots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", default="./lut")
    ap.add_argument("--size", type=int, default=33)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    knots_all = sample_knots(args.onnx, args.shards)

    # Pick the sample whose predicted peak best MATCHES each panel, not the extreme one.
    #
    # argmax() picked a sample the network believes sits on a ~4600 nit panel; forcing
    # that curve onto a 1000 nit panel clamps the top three knots flat and produces a LUT
    # that is not what any real 1000 nit display would render. Choosing the closest match
    # keeps the curve inside the panel's range and leaves the clamp as a pure safety net.
    peaks = knots_all[:, 7]
    i_sdr = int(np.argmin(np.abs(peaks - np.log2(100.0 / 100.0))))
    i_hdr = int(np.argmin(np.abs(peaks - np.log2(1000.0 / 100.0))))

    contexts = {
        "SDR_100nit": (i_sdr, 100.0, 0.05),
        "HDR_1000nit": (i_hdr, 1000.0, 0.05),
    }

    manifest = {"onnx": os.path.basename(args.onnx), "lut_size": args.size,
                "note": "Values are LINEAR display, not encoded. Apply sRGB/PQ yourself.",
                "contexts": {}}

    for name, (idx, peak, floor) in contexts.items():
        knots = knots_all[idx]
        lut, k_clamped = bake_cube(
            knots, size=args.size, peak_nits=peak, min_nits=floor,
            saturation=1.08, gamut=0.0, highlight_desat=0.05)
        path = os.path.join(args.out, f"NTMO_{name}_{args.size}.cube")
        write_cube(path, lut, f"NTMO {name} {args.size}^3")
        manifest["contexts"][name] = {
            "source_sample_index": idx,
            "knots_log2_raw": [round(float(v), 4) for v in knots],
            "knots_log2_clamped": [round(float(v), 4) for v in k_clamped],
            "curve_peak_nits": round(float(2 ** k_clamped[7] * 100.0), 1),
            "peak_nits": peak, "min_nits": floor,
            "file": os.path.basename(path),
        }
        print(f"  {name}: peak {2 ** k_clamped[7] * 100:.1f} nit -> {path}")

    with open(os.path.join(args.out, "NTMO_LUT_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("  manifest written")


if __name__ == "__main__":
    main()
