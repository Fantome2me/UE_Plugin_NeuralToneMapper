"""Local Laplacian Filter parameter labels (learned, per tile).

This is the second half of the LLF-LUT paper that v1.0.1 did NOT implement: instead of
three hand-tuned band strengths (0.60 / 0.25 / 0.10), the network predicts a 4x4 grid of
per-band strengths, and the engine applies them as a Durand-style base/detail split.

Why the label is a least-squares fit and not a fixed formula
-----------------------------------------------------------
The teacher is a real local tone mapping, not a constant:

    ref_log = curve(base) + detail            # compress the BASE only, keep detail

Ideal correction (in display-referred log2 stops) is therefore

    target = ref_log - curve(logY)

The engine can only produce a correction of the form

    corr = s_base * (curve(G3) - curve(G0))     # nonlinear base compression term
         + s_mid  * (G1 - G3)                   # coarse band
         + s_high * (G0 - G1)                   # fine band

so per tile we solve the 3x3 linear least squares for (s_base, s_mid, s_high). That is
exactly "the LLF parameter map that best reproduces the teacher", and it is closed form,
so label generation stays as fast as the rest of the dataset pipeline.

The three regressors are deliberately NOT collinear: (G0-G1) and (G1-G3) are disjoint
frequency bands, and (curve(G3)-curve(G0)) is the nonlinearity of the curve across the
same span. Using (G0-G3) for the middle term instead would be an exact linear
combination of the other two and the system would be singular.
"""

import numpy as np

LOCAL_TILES = 4
# 4 bands per tile:
#   s_base : nonlinear base compression      (curve(base) - curve(logY))
#   s_mid  : coarse detail band              (G1 - G3)
#   s_high : fine detail band                (G0 - G1)
#   s_low  : LOCAL EXPOSURE, the DC term     (G3 - frame mean)
#
# The fourth one exists because the first three are all band DIFFERENCES: they are
# zero-mean locally, so no combination of them can lift a uniformly dark region. That
# is why NTM 1.0.2 "did nothing in the shadows": its labels said "restore the detail the
# curve ate", and a compressive SDR curve only eats detail in the highlights. A local
# exposure term is what makes a dark corner visibly open up.
LOCAL_BANDS = 4
LOCAL_N = LOCAL_TILES * LOCAL_TILES * LOCAL_BANDS     # 64

# How much of the local-exposure term the teacher applies. 0.35 stops of lift per stop
# a region sits below the frame mean, saturated at EXPOSURE_CLAMP stops.
TEACHER_EXPOSURE_GAIN = 0.35
EXPOSURE_CLAMP = 3.0

# The curve is evaluated on the same knot grid the engine uses.
KNOT_X0 = -10.0
KNOT_X1 = 4.0
KNOT_N = 8

# Clamp for the fitted strengths. Beyond this the correction is large enough to produce
# visible halos, so the label is saturated rather than letting one extreme tile dominate.
S_MIN, S_MAX = -1.5, 1.5


def _down2(x):
    """2x box downsample, matching the engine's Downsample pass."""
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:h, :w]
    return 0.25 * (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2])


def _up_to(x, shape):
    """Bilinear upsample to `shape` (matches a bilinear texture fetch closely enough)."""
    h, w = shape
    ys = (np.arange(h, dtype=np.float32) + 0.5) * (x.shape[0] / h) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * (x.shape[1] / w) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, x.shape[0] - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, x.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, x.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, x.shape[1] - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    return (x[np.ix_(y0, x0)] * (1 - fy) * (1 - fx)
            + x[np.ix_(y0, x1)] * (1 - fy) * fx
            + x[np.ix_(y1, x0)] * fy * (1 - fx)
            + x[np.ix_(y1, x1)] * fy * fx)


def eval_curve(log_x, knots):
    """Piecewise-linear interpolation through the 8 knots, same as the engine."""
    t = (log_x - KNOT_X0) / (KNOT_X1 - KNOT_X0) * (KNOT_N - 1)
    t = np.clip(t, 0.0, KNOT_N - 1 - 1e-6)
    i0 = np.floor(t).astype(np.int32)
    f = (t - i0).astype(np.float32)
    k0 = knots[i0]
    k1 = knots[np.minimum(i0 + 1, KNOT_N - 1)]
    return (k0 + (k1 - k0) * f).astype(np.float32)


def pyramid(lo, levels=4):
    """G0..G3 (1/2, 1/4, 1/8, 1/16) of the log2-luminance image."""
    out = []
    cur = lo
    for _ in range(levels):
        cur = _down2(cur)
        out.append(cur)
    return out


def derive_local_params(hdr, knots, tiles=LOCAL_TILES):
    """
    hdr   : (H,W,3) linear, 1.0 == 100 nits
    knots : (8,) log2 output of the tone curve
    return: (LOCAL_N,) float32, tile-major [tile][band]
    """
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    logy = np.log2(np.maximum(lum, 1e-7)).astype(np.float32)
    H, W = logy.shape

    G0, G1, G2, G3 = pyramid(logy, 4)

    G0u = _up_to(G0, (H, W))
    G1u = _up_to(G1, (H, W))
    G3u = _up_to(G3, (H, W))

    c0 = eval_curve(G0u, knots)
    c3 = eval_curve(G3u, knots)

    # Teacher: compress the base, keep the detail (Durand), PLUS open up the regions
    # that sit below the frame mean. Without the third term the reference is detail
    # restoration only, and detail restoration is intrinsically a highlight operation
    # (its magnitude is proportional to 1 - curve_slope, which is largest where the
    # curve compresses hardest).
    mean_log = float(np.log2(max(float(np.exp2(logy).mean()), 1e-7)))
    detail = logy - G3u
    local_exposure = TEACHER_EXPOSURE_GAIN * np.clip(mean_log - G3u, -EXPOSURE_CLAMP, EXPOSURE_CLAMP)
    ref_log = c3 + detail + local_exposure

    target = (ref_log - eval_curve(logy, knots)).astype(np.float32)

    T1 = (c3 - c0).astype(np.float32)          # nonlinear base compression
    T2 = (G1u - G3u).astype(np.float32)        # coarse band
    T3 = (G0u - G1u).astype(np.float32)        # fine band
    T4 = np.clip(G3u - mean_log, -EXPOSURE_CLAMP, EXPOSURE_CLAMP).astype(np.float32)

    th = max(1, H // tiles)
    tw = max(1, W // tiles)

    out = np.zeros((tiles, tiles, LOCAL_BANDS), dtype=np.float32)
    reg = 1e-3        # ridge term keeps tiles with a flat band from blowing up

    for ty in range(tiles):
        for tx in range(tiles):
            sl = (slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw))
            a = [T1[sl].ravel(), T2[sl].ravel(), T3[sl].ravel(), T4[sl].ravel()]
            y = target[sl].ravel()
            n = y.size
            if n < 8:
                continue
            A = np.stack(a, axis=1)                       # [n,4]
            M = A.T @ A
            M += np.eye(4, dtype=np.float32) * (reg * max(np.trace(M) / 4.0, 1e-6))
            b = A.T @ y
            try:
                s = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                s = np.zeros(LOCAL_BANDS, dtype=np.float32)
            out[ty, tx] = np.clip(s, S_MIN, S_MAX)

    # Normalise to [-1,1] so the network head can use a plain tanh.
    return (out / S_MAX).astype(np.float32).ravel()


def derive_local_params_from_reference(hdr, reference_rgb, knots, tiles=LOCAL_TILES):
    """Fit the actual runtime local correction to a paired display-linear reference."""
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    ref_lum = reference_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    logy = np.log2(np.maximum(lum, 1e-7)).astype(np.float32)
    ref_log = np.log2(np.maximum(ref_lum, 1e-5)).astype(np.float32)
    H, W = logy.shape

    G0, G1, _, G3 = pyramid(logy, 4)
    G0u, G1u, G3u = (_up_to(g, (H, W)) for g in (G0, G1, G3))
    mean_log = float(logy.mean())

    target = ref_log - eval_curve(logy, knots)
    detail_mix, strength_scale = 0.25, 1.5
    terms = [
        np.zeros_like(logy),
        detail_mix * strength_scale * (G1u - G3u),
        detail_mix * strength_scale * (G0u - G1u),
    ]
    max_shadow_stops = np.clip(mean_log - G1u, 0.0, EXPOSURE_CLAMP) * strength_scale * 0.5 * 3.0
    terms.append(eval_curve(logy + max_shadow_stops, knots) - eval_curve(logy, knots))
    out = np.zeros((tiles, tiles, LOCAL_BANDS), dtype=np.float32)
    th, tw = max(1, H // tiles), max(1, W // tiles)
    for ty in range(tiles):
        for tx in range(tiles):
            sl = (slice(ty * th, (ty + 1) * th), slice(tx * tw, (tx + 1) * tw))
            A = np.stack([term[sl].ravel() for term in terms], axis=1)
            y = target[sl].ravel()
            if y.size < 8:
                continue
            M = A.T @ A
            M += np.eye(LOCAL_BANDS, dtype=np.float32) * (1e-3 * max(np.trace(M) / LOCAL_BANDS, 1e-6))
            try:
                fit = np.linalg.solve(M, A.T @ y)
                fit[0] = 0.0
                fit[1:3] = np.clip(fit[1:3], -1.0, 1.0)
                fit[3] = -np.clip(fit[3], 0.0, 1.0)
                out[ty, tx] = fit
            except np.linalg.LinAlgError:
                pass
    return (out / S_MAX).astype(np.float32).ravel()


def edge_channel(hdr):
    """
    Sobel magnitude of the 1/2-res log2 luminance, in [0,1].

    This is the "conditioned on a Canny edge map" input from the paper: the network sees
    where the strong edges are, which is what tells it how aggressively the local
    filter may act without creating halos. Computed on the same 1/2-res level the
    engine's G0 provides, so the training and inference inputs agree.
    """
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    half = _down2(np.log2(np.maximum(lum, 1e-7)).astype(np.float32))

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(half, 1, mode="edge")
    win = sliding_window_view(p, (3, 3))
    gx = (win * kx).sum(axis=(2, 3))
    gy = (win * ky).sum(axis=(2, 3))
    mag = np.sqrt(gx * gx + gy * gy)

    # 3 stops of log2 difference across two pixels is already a very strong edge.
    return np.clip(mag / 3.0, 0.0, 1.0).astype(np.float32)
