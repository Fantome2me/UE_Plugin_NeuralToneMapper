"""
Packing of (HDR, stats, display) into the exact tensor the UE plugin consumes.

THIS FILE IS THE CONTRACT. If you change C or the channel order here you must
change NTMOCurvePolicyNeural.cpp / NTMOLutBake.usf to match.

Network input : float32 [C=16, H=72, W=128]
   0..2   PQ encoded RGB (LinearToST2084(x * 100)), same shaper as the plugin LUT
   3      log2(mean luminance) / 16
   4      log2(p99.99 luminance) / 16
   5      peak_nits / 4000
   6      fullframe_nits / 4000        (== peak for SDR)
   7      log10(min_nits) / 4          ([-4,0] -> [-1,0])
   8      paper_white_nits / 300
   9      is_hdr
   10..12 output device one-hot: SDR / PQ / scRGB
   13     gamut id / 2
   14     brightness bias (stops) / 3
   15     intensity

Network output: float32 [C=12]
   0..7   curve knots, log2 of the OUTPUT linear value at input nodes
          x_i = -10 + 2*i  (log2 input; 1.0 == 100 nits)
   8      saturation       [0.5, 2.0]
   9      gamut compress   [0, 1]
   10     highlight desat  [0, 1]
   11     black lift       [0, 0.02]

ON-DISK FORMAT (why it is not the tensor above):
    A full [16,72,128] fp16 sample is 288 KB. 1M samples would be 288 GB, which
    blows past a 300 GB data disk. Only channels 0..2 actually vary per pixel -
    everything else is a per-sample scalar broadcast over the image. So we store:

        rgb_pq : uint8  [N, 3, 72, 128]   27.6 KB/sample
        ctx    : float32[N, 13]          channels 3..15, post-normalisation
        y      : float32[N, 12]

    ~27.7 KB/sample -> 1M samples = 28 GB. `assemble_tensor` rebuilds the exact
    [16,72,128] tensor at load time, so training sees the same thing the plugin
    will. PQ is perceptually uniform, so 8 bit costs the curve regression
    essentially nothing (we regress 8 global numbers, not pixels).
"""

import numpy as np

# Contract v2 (NTM 1.0.2). Changing either constant requires the matching change in
# NTMOCurvePolicyNeural.h (NTMOInput::Channels) and NTMOFeatures.usf.
C_IMG = 4           # PQ rgb (3) + edge magnitude (1)
C_IN = 49           # 4 image + 45 ctx (13 display/scene + 16 tiles + 16 histogram)
C_OUT = 72          # 8 curve knots + 64 local Laplacian strengths
C_CTX = 45          # channels 4..48
LOCAL_N = 64        # 4x4 tiles x 4 bands (s_base, s_mid, s_high, s_low)
W, H = 128, 72

PQ_M1 = 0.1593017578125
PQ_M2 = 78.84375
PQ_C1 = 0.8359375
PQ_C2 = 18.8515625
PQ_C3 = 18.6875


def linear_to_pq(nits_linear):
    y = np.clip(nits_linear, 0.0, 1.0)
    ym1 = np.power(y, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * ym1) / (1.0 + PQ_C3 * ym1), PQ_M2)


def pack_sample(hdr, stats, disp):
    """hdr (H,W,3) linear 1.0==100nits -> (img_u8 [4,H,W], ctx float32 [45]).

    Channel 3 is the edge magnitude. It is the paper's "conditioned on an edge map"
    input: it tells the network where the strong edges are, which is exactly what
    decides how hard the local filter may push before halos appear.
    """
    pq = linear_to_pq(hdr * 100.0)
    rgb_u8 = np.clip(pq * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Edge at 1/2 res (same level the engine's G0 provides), upsampled back so it can
    # ride along as a 4th channel.
    from ntmo_local import edge_channel, _up_to
    e = edge_channel(hdr)
    e_full = _up_to(e, rgb_u8.shape[:2])
    img_u8 = np.concatenate(
        [rgb_u8, np.clip(e_full * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, :, None]], axis=2
    ).transpose(2, 0, 1)

    log2_mean = np.log2(max(stats["mean_lum"], 1e-8))
    log2_p9999 = np.log2(max(stats["p9999_lum"], 1e-8))

    ctx = np.empty(C_CTX, dtype=np.float32)
    ctx[0] = log2_mean / 16.0
    ctx[1] = log2_p9999 / 16.0
    ctx[2] = min(disp["peak_nits"] / 4000.0, 1.0)
    ctx[3] = min(disp["fullframe_nits"] / 4000.0, 1.0)
    ctx[4] = np.clip(np.log10(max(disp["min_nits"], 1e-4)) / 4.0, -1.0, 0.0)
    ctx[5] = min(disp["paper_nits"] / 300.0, 1.5)
    ctx[6] = disp["is_hdr"]

    outdev = int(disp["outdev"])
    ctx[7] = 1.0 if outdev <= 2 else 0.0        # SDR gamma
    ctx[8] = 1.0 if outdev in (3, 4) else 0.0   # PQ
    ctx[9] = 1.0 if outdev in (5, 6) else 0.0   # scRGB

    ctx[10] = disp["gamut"] / 2.0
    ctx[11] = disp.get("bias", 0.0) / 3.0
    ctx[12] = disp.get("intensity", 1.0)

    # --- 4x4 tile luminance grid (ctx[13:29]) -------------------------------
    # The image branch aggregate-pools to a single descriptor, so without these
    # tiles the network cannot tell "uniformly dark" from "dark room with a bright
    # sign", which are exactly the two scenes that need different black points.
    # Measured without tiles: corr(pred knot0, true knot0) stalled at 0.36 no
    # matter how long training ran. These 16 numbers encode the layout cheaply.
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    H, W = lum.shape
    th, tw = max(1, H // 4), max(1, W // 4)
    for ty in range(4):
        for tx in range(4):
            tile = lum[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            m = float(tile.mean()) if tile.size else 0.0
            ctx[13 + ty * 4 + tx] = np.clip(np.log2(max(m, 1e-8)) / 16.0, -1.0, 1.0)

    # --- 16-bin log2 luminance histogram (ctx[29:45]) -----------------------
    # Tiles alone were not enough: they describe WHERE brightness sits but not HOW
    # MUCH of the frame is bright, which is what separates "specular" (a tiny blown
    # highlight) from "marquee" (a large lit sign). Measured on the 4x4 grid alone,
    # corr(tile, true knot0) was only -0.08, and a least-squares fit over all inputs
    # capped achievable L1 at ~1.0 -- so the missing information, not the network,
    # was the binding constraint.
    hist_edges = np.linspace(-14.0, 8.0, 17)
    lg = np.log2(np.maximum(lum, 1e-8)).ravel()
    counts, _ = np.histogram(lg, bins=hist_edges)
    ctx[29:45] = (counts.astype(np.float32) / max(lg.size, 1))

    return img_u8, ctx


def make_label(knots, local=None):
    """
    knots : 8 floats (log2 display output)
    local : 64 floats in [-1,1], tile-major (s_base, s_mid, s_high, s_low) per 4x4 tile

    -> (72,) float32.  [0:8] curve, [8:72] local Laplacian strengths.

    The four "look" values the previous contract carried (saturation, gamut, highlight
    desaturation, black lift) are gone: the engine now reads them from project settings,
    because keeping them as network outputs meant the artist-facing sliders did nothing
    while the neural path was active.
    """
    y = np.zeros(C_OUT, dtype=np.float32)
    y[0:8] = knots
    if local is not None:
        y[8:8 + LOCAL_N] = np.asarray(local, dtype=np.float32).ravel()[:LOCAL_N]
    return y


# ---------------------------------------------------------------------------
# Batch assembly -- used by train.py. Torch version to avoid a host round trip.
# ---------------------------------------------------------------------------
def assemble_tensor_torch(img_u8, ctx):
    """
    img_u8 : uint8 tensor [B,4,72,128]   ctx : float32 tensor [B,45]
    returns float32 tensor [B,49,72,128]
    """
    import torch

    rgb = img_u8.float().div_(255.0)
    b = rgb.shape[0]
    parts = [rgb]
    for i in range(C_CTX):
        parts.append(ctx[:, i].view(b, 1, 1, 1).expand(b, 1, H, W))
    return torch.cat(parts, dim=1)
