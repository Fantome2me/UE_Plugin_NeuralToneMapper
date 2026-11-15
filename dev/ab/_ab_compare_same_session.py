"""Quantitative A/B between the neural tone mapper and Unreal's own tonemapper.

Both images come from ONE editor session with the camera pinned, so a per-pixel
comparison is valid: identical geometry, lighting, sun angle and TAA history. The
editor UI is also identical in both, so the differing region is the 3D viewport and
its bounding box is found automatically.

Reports, over the viewport region only:
  * mean / median per-channel difference and RMS
  * luminance histogram comparison (does the curve move the distribution?)
  * shadow / mid / highlight band statistics, which is where a tone curve differs
  * clipping rates (fraction at 0 and 1), the classic tone mapper failure mode

Usage: python _ab_compare_same_session.py <on.png> <off.png>
"""

import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def load(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)

    on = load(sys.argv[1])
    off = load(sys.argv[2])

    if on.shape != off.shape:
        raise SystemExit("size mismatch: {} vs {}".format(on.shape, off.shape))

    diff = on - off
    activity = np.abs(diff).max(axis=2)

    # The UI is byte-identical in both captures, so anything above the noise floor is
    # the 3D viewport. Derive its bounding box rather than hardcoding pixel rects.
    mask = activity > (2.0 / 255.0)
    ys, xs = np.nonzero(mask)
    if len(xs) < 1000:
        raise SystemExit(
            "almost no difference found ({} px) - the plugin did not render".format(len(xs)))

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    print("viewport bbox  x {}..{}  y {}..{}  ({:.2f} Mpix)".format(
        x0, x1, y0, y1, (x1 - x0) * (y1 - y0) / 1e6))

    a = on[y0:y1, x0:x1]
    b = off[y0:y1, x0:x1]
    d = a - b

    def lum(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    la, lb = lum(a), lum(b)
    dl = la - lb

    print()
    print("--- display-referred (sRGB code values) ---")
    print("  mean diff per channel  R {:+.4f}  G {:+.4f}  B {:+.4f}".format(
        d[..., 0].mean(), d[..., 1].mean(), d[..., 2].mean()))
    print("  mean |diff|            {:.4f}   RMS {:.4f}".format(np.abs(d).mean(), np.sqrt((d ** 2).mean())))
    print("  luma mean  NTMO {:.4f}   UE {:.4f}   delta {:+.4f} ({:+.2f}%)".format(
        la.mean(), lb.mean(), dl.mean(), 100.0 * dl.mean() / max(lb.mean(), 1e-6)))
    print("  luma median NTMO {:.4f}   UE {:.4f}   delta {:+.4f}".format(
        float(np.median(la)), float(np.median(lb)), float(np.median(la) - np.median(lb))))
    print("  luma std   NTMO {:.4f}   UE {:.4f}  (contrast ratio {:.3f})".format(
        la.std(), lb.std(), la.std() / max(lb.std(), 1e-6)))
    print("  Pearson corr of luma   {:.4f}".format(
        float(np.corrcoef(la.ravel(), lb.ravel())[0, 1])))
    print("  mean |delta| stops     {:.4f}".format(
        float(np.abs(np.log2((la + 1e-4) / (lb + 1e-4))).mean())))

    print()
    print("--- band analysis (bands defined on Unreal's luma) ---")
    for name, lo, hi in (("shadow   <0.10", 0.0, 0.10),
                         ("low      0.10-0.25", 0.10, 0.25),
                         ("mid      0.25-0.60", 0.25, 0.60),
                         ("high     0.60-0.90", 0.60, 0.90),
                         ("specular >0.90", 0.90, 1.01)):
        sel = (lb >= lo) & (lb < hi)
        n = int(sel.sum())
        if n == 0:
            print("  {:20s} empty".format(name))
            continue
        print("  {:20s} {:7.2f}% of px   dLuma {:+.4f}   dStops {:+.3f}".format(
            name, 100.0 * n / lb.size,
            float(dl[sel].mean()),
            float(np.mean(np.log2((la[sel] + 1e-4) / (lb[sel] + 1e-4))))))

    print()
    print("--- clipping ---")
    for name, arr in (("NTMO", a), ("UE  ", b)):
        blk = (arr.max(axis=2) <= 1.0 / 255.0).mean()
        wht = (arr.min(axis=2) >= 254.0 / 255.0).mean()
        print("  {} black {:.3f}%   white {:.3f}%".format(name, 100 * blk, 100 * wht))

    print()
    print("--- interpretation ---")
    lr = la.std() / max(lb.std(), 1e-6)
    if lr > 1.06:
        verdict = "higher global contrast than Unreal"
    elif lr < 0.94:
        verdict = "lower global contrast than Unreal (flatter)"
    else:
        verdict = "comparable global contrast to Unreal"
    print("  {}".format(verdict))
    print("  mean absolute error {:.2f} display levels (of 255)".format(255 * np.abs(d).mean()))


if __name__ == "__main__":
    main()
