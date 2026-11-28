"""
NTMO teacher operators -- pure numpy, no torch, no downloads.

Every operator here is implemented from constants verified in Unreal Engine 5.8
source, so the labels are reproducible and auditable:

  * Hable / Uncharted 2 filmic  -> Engine/Shaders/Private/TonemapCommon.ush:619-640
        A=0.15 B=0.50 C=0.10 D=0.20 E=0.02 F=0.30, W=11.2, ExposureBias=2.0
  * UE FilmToneMap (4 official presets) -> TonemapCommon.ush:110-225 + :73-103
        Uncharted : Slope 0.63 Toe 0.55 Shoulder 0.47 Black 0.00 White 0.01
        HP        : Slope 0.65 Toe 0.63 Shoulder 0.45 Black 0.00 White 0.00
        Legacy    : Slope 0.98 Toe 0.30 Shoulder 0.22 Black 0.00 White 0.025
        ACES      : Slope 0.91 Toe 0.53 Shoulder 0.23 Black 0.00 White 0.035
  * Reinhard extended, pure linear clip/gamma.

All operators take a LINEAR value where 1.0 == 100 nits (Unreal's
LinearToNitsScale convention) and return a display-referred linear value in the
same 1.0 == 100 nits domain, already limited to the panel peak.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LINEAR_TO_NITS = 100.0

UE_FILMIC_PRESETS = {
    "uncharted": dict(slope=0.63, toe=0.55, shoulder=0.47, black=0.00, white=0.010),
    "hp":        dict(slope=0.65, toe=0.63, shoulder=0.45, black=0.00, white=0.000),
    "legacy":    dict(slope=0.98, toe=0.30, shoulder=0.22, black=0.00, white=0.025),
    "aces":      dict(slope=0.91, toe=0.53, shoulder=0.23, black=0.00, white=0.035),
}

HABLE = dict(A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30, W=11.2, bias=2.0)


# ---------------------------------------------------------------------------
# Individual operators. All are monotone, defined for x >= 0.
# ---------------------------------------------------------------------------
def op_clip(x):
    """Reference baseline: hard clip. Useful as the 'do nothing' teacher."""
    return np.clip(x, 0.0, 1.0)


def op_reinhard(x, white=4.0):
    """Extended Reinhard. white scales where things saturate."""
    white = max(float(white), 1e-3)
    return x * (1.0 + x / (white * white)) / (1.0 + x)


def op_hable(x):
    """Hable / Uncharted 2 filmic, exact UE constants."""
    A, B, C, D, E, F = HABLE["A"], HABLE["B"], HABLE["C"], HABLE["D"], HABLE["E"], HABLE["F"]
    W, bias = HABLE["W"], HABLE["bias"]

    def curve(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    return curve(np.maximum(x, 0.0) * bias) / curve(np.float32(W))


def _ue_filmic_curve(log10_x, slope, toe, shoulder, black_clip, white_clip):
    """Direct port of UE's FilmToneMap() core (log10 domain)."""
    toe_scale = 1.0 + black_clip - toe
    shoulder_scale = 1.0 + white_clip - shoulder

    in_match = 0.18
    out_match = 0.18

    if toe > 0.8:
        toe_match = (1.0 - toe - out_match) / slope + np.log10(in_match)
    else:
        bt = (out_match + black_clip) / toe_scale - 1.0
        toe_match = np.log10(in_match) - 0.5 * np.log((1.0 + bt) / (1.0 - bt)) * (toe_scale / slope)

    straight_match = (1.0 - toe) / slope - toe_match
    shoulder_match = shoulder / slope - straight_match

    straight = slope * (log10_x + straight_match)
    toe_c = (-black_clip) + (2.0 * toe_scale) / (1.0 + np.exp((-2.0 * slope / toe_scale) * (log10_x - toe_match)))
    sh_c = (1.0 + white_clip) - (2.0 * shoulder_scale) / (1.0 + np.exp((2.0 * slope / shoulder_scale) * (log10_x - shoulder_match)))

    t = np.clip((log10_x - toe_match) / (shoulder_match - toe_match), 0.0, 1.0)
    t = np.where(shoulder_match < toe_match, 1.0 - t, t)
    t = (3.0 - 2.0 * t) * t * t

    toe_part = np.where(log10_x < toe_match, toe_c, straight)
    sh_part = np.where(log10_x > shoulder_match, sh_c, straight)
    return toe_part * (1.0 - t) + sh_part * t


def op_ue_filmic(x, preset_name="uncharted"):
    """UE's built-in filmic tonemapper, one of the four documented presets."""
    p = UE_FILMIC_PRESETS[preset_name]
    xx = np.maximum(x, 1e-8)
    y = _ue_filmic_curve(np.log10(xx), p["slope"], p["toe"], p["shoulder"], p["black"], p["white"])
    return np.clip(y, 0.0, 1.0)


def op_analytic_policy(x, log_peak, log_floor=-10.0, knee_shift=0.0, toe_lift=0.20):
    """
    The plugin's own analytic policy, in log2 space. Mirrors NTMOPolicy.cpp so
    the network can be validated against the shipped fallback.
    """
    xx = np.maximum(x, 1e-8)
    log_x = np.log2(xx)

    knee = log_peak - 0.5 - knee_shift
    span = max(log_peak - knee, 0.5)
    toe_x = -4.0

    y = np.where(
        log_x <= toe_x,
        log_x + toe_lift * np.clip((toe_x - log_x) / 6.0, 0.0, 1.0) ** 2,
        np.where(log_x <= knee, log_x, knee + span * (1.0 - np.exp(-(log_x - knee) / span))),
    )
    y = np.clip(y, log_floor + 0.01, log_peak)
    return np.exp2(y)


# ---------------------------------------------------------------------------
# Registry: teacher -> callable(hdr_linear, ctx) -> display linear
# ---------------------------------------------------------------------------
def _finalize(y_norm, ctx, mid_grey_norm=0.18):
    """Map labels to display linear, keeping HDR diffuse gray independent of peak."""
    peak_linear = max(ctx["peak_nits"], 50.0) / LINEAR_TO_NITS
    floor_linear = max(ctx["min_nits"], 1e-5) / LINEAR_TO_NITS
    y = np.clip(np.asarray(y_norm, dtype=np.float64) * peak_linear,
                floor_linear, peak_linear)
    if not ctx.get("is_hdr", 0):
        return y

    target = np.clip(ctx.get("mid_grey_nits", 15.0) / LINEAR_TO_NITS,
                     floor_linear, peak_linear)
    raw_mid = np.clip(float(mid_grey_norm) * peak_linear, floor_linear, peak_linear)
    log_floor, log_peak = np.log2(floor_linear), np.log2(peak_linear)
    log_raw_mid, log_target = np.log2(raw_mid), np.log2(target)
    log_y = np.log2(np.maximum(y, floor_linear))
    below = log_y <= log_raw_mid
    t_low = np.clip((log_y - log_floor) / max(log_raw_mid - log_floor, 1e-8), 0.0, 1.0)
    t_high = np.clip((log_y - log_raw_mid) / max(log_peak - log_raw_mid, 1e-8), 0.0, 1.0)
    anchored = np.where(below,
                        log_floor + t_low * (log_target - log_floor),
                        log_target + t_high * (log_peak - log_target))
    return np.clip(np.exp2(anchored), floor_linear, peak_linear)


def _teacher_clip(x, ctx):
    return _finalize(op_clip(x), ctx, op_clip(0.18))


def _teacher_reinhard(x, ctx):
    white = ctx.get("reinhard_white", 4.0)
    return _finalize(op_reinhard(x, white=white), ctx, op_reinhard(0.18, white=white))


def _teacher_hable(x, ctx):
    return _finalize(op_hable(x), ctx, op_hable(0.18))


def _make_ue_filmic(name):
    def _f(x, ctx):
        return _finalize(op_ue_filmic(x, name), ctx, op_ue_filmic(0.18, name))
    _f.__name__ = f"teacher_ue_filmic_{name}"
    return _f


def _teacher_analytic(x, ctx):
    log_peak = np.log2(max(ctx["peak_nits"], 50.0) / LINEAR_TO_NITS)
    # ABL makes the shoulder start earlier, exactly like the plugin.
    abl = 1.0 - ctx["fullframe_nits"] / max(ctx["peak_nits"], 1.0)
    knee_shift = 2.5 * float(np.clip(abl, 0.0, 1.0)) * ctx.get("abl_awareness", 1.0)
    y = op_analytic_policy(x, log_peak, knee_shift=knee_shift)
    raw_mid = float(op_analytic_policy(0.18, log_peak, knee_shift=knee_shift))
    peak_linear = max(ctx["peak_nits"], 50.0) / LINEAR_TO_NITS
    return _finalize(y / peak_linear, ctx, raw_mid / peak_linear)


TEACHERS = {
    "clip": _teacher_clip,
    "reinhard": _teacher_reinhard,
    "hable": _teacher_hable,
    "ue_uncharted": _make_ue_filmic("uncharted"),
    "ue_hp": _make_ue_filmic("hp"),
    "ue_legacy": _make_ue_filmic("legacy"),
    "ue_aces": _make_ue_filmic("aces"),
    "analytic": _teacher_analytic,
}

# Sampling weights: the analytic policy and Hable get the most mass because they
# are the closest to what a game actually wants. clip is kept for regression.
TEACHER_WEIGHTS = {
    "clip": 0.04,
    "reinhard": 0.10,
    "hable": 0.20,
    "ue_uncharted": 0.14,
    "ue_hp": 0.10,
    "ue_legacy": 0.08,
    "ue_aces": 0.14,
    "analytic": 0.20,
}


# ---------------------------------------------------------------------------
# Label extraction
#
# The teachers below are pure functions of luminance, so the 8 knots can be
# evaluated DIRECTLY on the knot grid instead of fitted through image pixels.
# That is exact (no sampling error, no empty bins, guaranteed smooth), ~50x
# faster, and it means the label depends only on (display, scene statistics) -
# which is precisely the mapping we want the network to learn.
# ---------------------------------------------------------------------------
KNOT_X0, KNOT_X1, KNOT_N = -10.0, 4.0, 8
KNOT_STEP = (KNOT_X1 - KNOT_X0) / (KNOT_N - 1)   # 2.0
KNOT_LOG_MIN, KNOT_LOG_MAX = -14.0, 8.0


def knots_dense(teacher_fn, ctx):
    """
    Evaluate a teacher at the 8 knot inputs x_i = -10 + 2i (log2, 1.0 == 100 nits)
    and return the resulting 8 output knots in log2.

    Always monotone: every teacher here is monotone and _finalize() is monotone.
    """
    xs = KNOT_X0 + KNOT_STEP * np.arange(KNOT_N, dtype=np.float64)
    lin = np.exp2(xs).astype(np.float32)

    # Neutral grey probe: one row per knot, 3 channels.
    probe = np.repeat(lin[:, None], 3, axis=1)[None, :, :]
    out = teacher_fn(probe, ctx)
    m_out = np.max(out, axis=-1)[0]

    knots = np.log2(np.maximum(m_out, 1e-12)).astype(np.float32)
    knots = np.clip(knots, KNOT_LOG_MIN, KNOT_LOG_MAX)

    # Numerical guard: force strict monotonicity.
    for i in range(1, KNOT_N):
        knots[i] = max(knots[i], knots[i - 1] + 1e-4)
    return knots


def fit_knots_from_pixels(hdr, teacher_fn, ctx, min_count=32):
    """
    Pixel-driven variant, kept for real EXR input (where a teacher may not be a
    pure luminance function). Not used for the synthetic dataset.
    """
    m_in = np.max(hdr, axis=-1)
    out = teacher_fn(hdr, ctx)
    m_out = np.max(out, axis=-1)

    valid = m_in > 1e-6
    if not np.any(valid):
        return None

    log_in = np.log2(m_in[valid])
    log_out = np.log2(np.maximum(m_out[valid], 1e-8))

    idx = np.clip(((log_in - KNOT_X0) / KNOT_STEP).astype(np.int32), 0, KNOT_N - 1)
    sums = np.bincount(idx, weights=log_out, minlength=KNOT_N)
    counts = np.bincount(idx, minlength=KNOT_N).astype(np.float64)

    knots = np.full(KNOT_N, np.nan, dtype=np.float64)
    ok = counts >= min_count
    knots[ok] = sums[ok] / counts[ok]
    if not np.any(ok):
        return None

    xi = np.arange(KNOT_N)
    knots = np.interp(xi, xi[ok], knots[ok])
    return np.clip(knots, KNOT_LOG_MIN, KNOT_LOG_MAX).astype(np.float32)



def knots_to_curve(knots, log_floor=-10.0):
    """Monotone cubic evaluation matching NTMOCommon.ush (for validation)."""
    x = np.linspace(KNOT_X0 - 2.0, KNOT_X1 + 2.0, 512)
    k = np.asarray(knots, dtype=np.float64)

    t = np.clip((x - KNOT_X0) / KNOT_STEP, 0.0, KNOT_N - 1)
    i = np.minimum(np.floor(t).astype(int), KNOT_N - 2)
    u = t - i

    y0, y1 = k[i], k[i + 1]
    ym = np.where(i > 0, k[np.maximum(i - 1, 0)], y0)
    yp = np.where(i + 2 < KNOT_N, k[np.minimum(i + 2, KNOT_N - 1)], y1)

    m0 = (y1 - ym) / (2.0 * KNOT_STEP)
    m1 = (yp - y0) / (2.0 * KNOT_STEP)

    delta = (y1 - y0) / KNOT_STEP
    flat = np.abs(delta) < 1e-6
    a = np.where(flat, 0.0, m0 / np.where(flat, 1.0, delta))
    b = np.where(flat, 0.0, m1 / np.where(flat, 1.0, delta))
    s = a * a + b * b
    scale = np.where(s > 9.0, 3.0 / np.sqrt(np.maximum(s, 1e-12)), 1.0)
    m0 = np.where(flat, 0.0, np.where(s > 9.0, scale * a * delta, m0))
    m1 = np.where(flat, 0.0, np.where(s > 9.0, scale * b * delta, m1))

    u2, u3 = u * u, u * u * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2

    y = h00 * y0 + h10 * KNOT_STEP * m0 + h01 * y1 + h11 * KNOT_STEP * m1
    return x, y
