"""Draw the tone-curve comparison figure used in the README.

Three curves on one plot: Unreal's fixed ACES, and the network's output on an SDR panel
versus on a 1000 nit HDR panel. Drawn with PIL because the training environment has no
matplotlib (and pulling one in just for a figure is not worth the dependency).

    python plot_curves.py --manifest ../lut/NTMO_LUT_manifest.json --out ../images/curves.png
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 760
PAD_L, PAD_R, PAD_T, PAD_B = 92, 40, 56, 76

X0, X1 = -10.0, 4.0          # scene log2 luminance (0 = 100 nits)
Y0, Y1 = -10.0, 4.0          # display log2 luminance


def eval_curve_log2(x, knots):
    """Same Fritsch-Carlson curve as NTMO_EvalCurveLog2 in NTMOCommon.ush."""
    import numpy as np

    step = (4.0 - (-10.0)) / 7.0
    t = np.clip((x - (-10.0)) / step, 0.0, 7.0)
    i = np.minimum(np.floor(t).astype(np.int32), 6)
    u = t - i

    def k(idx):
        return knots[np.clip(idx, 0, 7)]

    y0, y1, ym, yp = k(i), k(i + 1), k(i - 1), k(i + 2)
    m0 = (y1 - ym) / (2.0 * step)
    m1 = (yp - y0) / (2.0 * step)
    delta = (y1 - y0) / step
    flat = np.abs(delta) < 1e-6
    a = np.where(flat, 0.0, m0 / np.where(flat, 1.0, delta))
    b = np.where(flat, 0.0, m1 / np.where(flat, 1.0, delta))
    s = a * a + b * b
    sc = np.where(s > 9.0, 3.0 / np.sqrt(np.maximum(s, 1e-12)), 1.0)
    m0 = np.where(flat, 0.0, np.where(s > 9.0, sc * a * delta, m0))
    m1 = np.where(flat, 0.0, np.where(s > 9.0, sc * b * delta, m1))
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * y0 + (u3 - 2 * u2 + u) * step * m0
            + (-2 * u3 + 3 * u2) * y1 + (u3 - u2) * step * m1)


def aces_knots():
    """Unreal's ACES curve, taken from the teacher that produced the training labels.

    Using the real teacher instead of a remembered Hill fit matters: an approximate ACES
    drawn next to the network would invite exactly the comparison this figure is meant to
    support, while being subtly wrong. knots_dense() evaluates the teacher at the same 8
    x positions the network outputs, so both curves go through identical interpolation.
    """
    import ntmo_teachers as tp

    ctx = dict(peak_nits=100.0, fullframe_nits=100.0, min_nits=0.05, paper_nits=100.0,
               is_hdr=0.0, outdev=0, gamut=0, bias=0.0, intensity=1.0, reinhard_white=1.5)
    return tp.knots_dense(tp.TEACHERS["ue_aces"], ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np

    with open(args.manifest, encoding="utf-8") as f:
        man = json.load(f)

    sdr = man["contexts"]["SDR_100nit"]["knots_log2_clamped"]
    hdr = man["contexts"]["HDR_1000nit"]["knots_log2_clamped"]
    xs = np.linspace(X0, X1, 400)

    def to_px(x, y):
        px = PAD_L + (x - X0) / (X1 - X0) * (W - PAD_L - PAD_R)
        py = H - PAD_B - (y - Y0) / (Y1 - Y0) * (H - PAD_T - PAD_B)
        return px, py

    img = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
        small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
        small = font

    # grid
    for v in range(-10, 5, 2):
        p = to_px(v, Y0)[0], to_px(v, Y1)[1]
        d.line([p[0], PAD_T, p[0], H - PAD_B], fill=(44, 48, 56))
        d.line([PAD_L, to_px(X0, v)[1], W - PAD_R, to_px(X0, v)[1]], fill=(44, 48, 56))
        d.text((p[0] - 12, H - PAD_B + 8), f"{v:+d}", fill=(150, 156, 166), font=small)
        d.text((PAD_L - 46, to_px(X0, v)[1] - 10), f"{v:+d}", fill=(150, 156, 166), font=small)

    # identity
    d.line([to_px(X0, X0), to_px(X1, X1)], fill=(90, 96, 108), width=2)
    d.text((to_px(1.4, 1.4)[0] + 8, to_px(1.4, 1.4)[1] - 26), "线性（不压缩）",
           fill=(120, 126, 138), font=small)

    # ACES (the real teacher curve, as 8 knots -> same interpolation as the network)
    ka = np.asarray(aces_knots(), dtype=np.float64)
    ya = eval_curve_log2(xs, ka)
    pts = [to_px(a, b) for a, b in zip(xs, ya) if b >= Y0 - 0.02]
    if len(pts) > 1:
        d.line(pts, fill=(158, 158, 158), width=3)
    ax_, ay_ = to_px(-3.6, float(ya[190]))
    draw_label(d, (ax_ - 168, ay_ + 14), "UE 原生 ACES（固定）", (178, 178, 178), small)

    # NTM SDR
    ys = eval_curve_log2(xs, np.array(sdr))
    d.line([to_px(a, b) for a, b in zip(xs, ys)], fill=(86, 180, 255), width=4)
    draw_label(d, to_px(-8.2, float(ys[95]) - 1.35), "NTM @ SDR 100 nit", (120, 200, 255), small)

    # NTM HDR
    yh = eval_curve_log2(xs, np.array(hdr))
    d.line([to_px(a, b) for a, b in zip(xs, yh)], fill=(255, 168, 72), width=4)
    draw_label(d, to_px(-9.5, float(yh[120]) + 0.5), "NTM @ HDR 1000 nit", (255, 190, 110), small)

    d.text((PAD_L, 18), "色调曲线：UE 固定曲线 vs 神经网络按显示器能力现算的曲线",
           fill=(235, 238, 244), font=font)
    d.text((PAD_L, H - 34), "场景亮度 log2（0 = 100 nit）", fill=(160, 166, 176), font=small)
    d.text((12, H // 2 - 40), "输出 log2", fill=(160, 166, 176), font=small)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    img.save(args.out)
    print(f"  wrote {args.out} ({W}x{H})")


def draw_label(d, xy, text, color, font):
    """Draw text on a dark plate so it stays readable over the curves."""
    x, y = xy
    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = 6
    d.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=(26, 28, 33))
    d.rectangle([x - pad, y - pad, x - pad + 3, y + h + pad], fill=color)
    d.text((x, y), text, fill=color, font=font)


def eval_at(xs, ys, x):
    import numpy as np
    return float(np.interp(x, xs, ys))


if __name__ == "__main__":
    main()
