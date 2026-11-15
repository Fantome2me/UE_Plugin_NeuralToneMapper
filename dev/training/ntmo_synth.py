"""
Synthetic HDR scene + display-context sampling.

Deliberately generates HDR at exactly the resolution the network consumes
(128x72), never full resolution. That is what makes a 2 hour training run
possible: we synthesise millions of *statistics distributions*, not images.

The synthetic radiance fields are built to cover the log-luminance histogram
shapes that matter for tone mapping:
  * low-key      (night, interior, mean ~1e-3)
  * normal       (mean ~0.18)
  * high-key     (daylight exterior, mean ~1..10)
  * specular     (small very bright sources, 1e3..1e5 nits)
  * marquee      (large bright area -> stresses ABL)
"""

import numpy as np

W, H = 128, 72


def _fbm(shape, octaves, rng, persistence=0.5, base=4):
    """Cheap fractal value noise via upsampled random grids."""
    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    total = 0.0
    res = base
    for _ in range(octaves):
        g = rng.random((max(2, res), max(2, int(res * w / h)))).astype(np.float32)
        # bilinear upsample to full res
        yi = np.linspace(0, g.shape[0] - 1, h, dtype=np.float32)
        xi = np.linspace(0, g.shape[1] - 1, w, dtype=np.float32)
        y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, g.shape[0] - 1)
        x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, g.shape[1] - 1)
        fy = (yi - y0)[:, None]; fx = (xi - x0)[None, :]
        up = (g[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) + g[np.ix_(y0, x1)] * (1 - fy) * fx
              + g[np.ix_(y1, x0)] * fy * (1 - fx) + g[np.ix_(y1, x1)] * fy * fx)
        out += amp * up
        total += amp
        amp *= persistence
        res *= 2
    return out / max(total, 1e-6)


def synth_hdr(rng):
    """
    Returns (hdr HxWx3 float32 linear, 1.0 == 100 nits, stats dict).

    Radiance is built in log2 space so the dynamic range is physically plausible
    rather than a uniform random field.
    """
    kind = rng.choice(
        ["lowkey", "normal", "highkey", "specular", "marquee"],
        p=[0.22, 0.34, 0.20, 0.14, 0.10],
    )

    shape = (H, W)
    base = _fbm(shape, 4, rng)                      # 0..1
    detail = _fbm(shape, 3, rng, base=16)

    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    if kind == "lowkey":
        log_mean = rng.uniform(-4.0, -1.5)
        contrast = rng.uniform(2.0, 4.5)
        grad = (yy * 1.2 - 0.4)
    elif kind == "highkey":
        log_mean = rng.uniform(-0.5, 2.0)
        contrast = rng.uniform(1.2, 2.6)
        grad = (yy * -1.5 + 1.2)
    elif kind == "specular":
        log_mean = rng.uniform(-3.0, -0.5)
        contrast = rng.uniform(2.5, 5.0)
        grad = (yy * 0.8 - 1.0)
    elif kind == "marquee":
        log_mean = rng.uniform(-1.0, 1.5)
        contrast = rng.uniform(1.5, 3.0)
        grad = (yy * -1.0 + 0.6)
    else:  # normal
        log_mean = rng.uniform(-2.0, 0.5)
        contrast = rng.uniform(1.8, 3.8)
        grad = (yy * 1.0 - 0.5)

    # log2 luminance field
    log_l = log_mean + contrast * ((base - 0.5) * 2.0) + 0.6 * (detail - 0.5) + grad * contrast * 0.5

    # Light sources: small very bright blobs
    n_lights = rng.integers(0, 5) if kind != "specular" else rng.integers(3, 10)
    for _ in range(int(n_lights)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rad = rng.uniform(1.5, 6.0)
        peak = rng.uniform(6.0, 16.0)   # log2, up to 2^16 * 100 nits is way past any panel
        gx = np.exp(-((np.arange(W)[None, :] - cx) ** 2) / (2 * rad * rad))
        gy = np.exp(-((np.arange(H)[:, None] - cy) ** 2) / (2 * rad * rad))
        blob = gx * gy
        log_l = np.maximum(log_l, log_l + np.log2(np.maximum(blob, 1e-4)) * 0 + (peak - log_l) * np.clip(blob * 2.2, 0, 1))

    # Occluders: dark shapes
    n_occ = rng.integers(0, 4)
    for _ in range(int(n_occ)):
        x0 = rng.integers(0, W - 8); x1 = min(W, x0 + rng.integers(8, W // 2))
        y0 = rng.integers(0, H - 8); y1 = min(H, y0 + rng.integers(6, H // 2))
        log_l[y0:y1, x0:x1] -= rng.uniform(2.0, 7.0)

    # Chromatic variation: tint each channel slightly differently so the network
    # learns that hue must be preserved rather than ignored.
    tint = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
    tint /= tint.mean()

    lum = np.exp2(np.clip(log_l, -20.0, 18.0)).astype(np.float32)
    hdr = lum[..., None] * tint[None, None, :]
    hdr = np.maximum(hdr, 0.0).astype(np.float32)

    m = np.max(hdr, axis=-1)
    stats = {
        "mean_lum": float(np.mean(hdr * np.array([0.2126, 0.7152, 0.0722], np.float32))),
        "p9999_lum": float(np.percentile(m, 99.99)),
        "kind": str(kind),
    }
    return hdr, stats


def sample_display(rng):
    """One plausible display capability sample. Covers SDR, OLED, LCD, ABL-limited."""
    is_hdr = rng.random() < 0.72

    if not is_hdr:
        return {
            "peak_nits": rng.choice([80.0, 100.0, 250.0, 400.0]).item(),
            "fullframe_nits": 0.0,   # filled below
            "min_nits": rng.choice([0.02, 0.05, 0.10, 0.15]).item(),
            "paper_nits": rng.choice([80.0, 100.0, 120.0]).item(),
            "is_hdr": 0.0,
            "outdev": 0,             # SDR gamma
            "gamut": int(rng.integers(0, 2)),  # sRGB or P3
        }

    peak = rng.choice([400.0, 600.0, 1000.0, 1400.0, 2000.0, 4000.0]).item()
    abl = rng.choice([0.30, 0.45, 0.60, 0.75, 1.0]).item()
    outdev = 3 if rng.random() < 0.8 else 4      # PQ mostly, some scRGB
    return {
        "peak_nits": peak,
        "fullframe_nits": peak * abl,
        "min_nits": rng.choice([0.0005, 0.002, 0.005, 0.05]).item(),
        "paper_nits": rng.choice([100.0, 160.0, 203.0, 300.0]).item(),
        "is_hdr": 1.0,
        "outdev": outdev,
        "gamut": int(rng.integers(0, 3)),   # sRGB / P3 / Rec2020
    }
