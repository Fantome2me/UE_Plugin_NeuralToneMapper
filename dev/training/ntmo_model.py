"""
NTMO curve-prediction networks -- deploy-safe by construction.

Two sizes:
    S  (~40 K params)  -> weak GPUs / GTX 1060 class
    M  (~250 K params) -> mainstream (RTX 3060+)

WHY THIS MODEL LOOKS THE WAY IT DOES
------------------------------------
The target inference backend is UE 5.8's NNERuntimeRDGHlsl. Its operator set was
verified by reading
    Engine/Plugins/Experimental/NNERuntimeRDG/Shaders/Private/NNEHlslShaders/

    supported: Conv, ConvTranspose, ConvMatmul, Gemm, MatMul,
               BatchNormalization, InstanceNormalization, LayerNormalization,
               Pool, Resize, Upsample, Pad, Slice, Split, Transpose,
               Gather, GatherElements, ScatterND, Softmax, Reduce, CumSum,
               Cast, ElementWise unary/binary/variadic
    NOT supported: attention, and Concat / GroupNorm are not on the list either.

So this model deliberately avoids:
  * GroupNorm        -> BatchNorm2d instead (folds into Conv at export)
  * Concat           -> project both branches to the same width and ADD
  * Softplus         -> sigmoid-based positive step (Sigmoid is ElementWiseUnary)
  * Clip             -> torch.maximum / minimum (ElementWiseBinary)
  * AdaptiveAvgPool  -> reduce-mean over H,W (Reduce)

Result: every op below maps onto the supported set, so the exported ONNX should
load without a translation layer. export_onnx.py verifies this.

Output contract (see ntmo_pack.py):
    [B, 8] curve knots in log2, strictly increasing
    [B, 4] saturation, gamut compress, highlight desat, black lift
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel counts come from the data contract, never from a local copy. A duplicated
# `C_IN = 16` here silently desynced from ntmo_pack after the tile features landed
# and produced "mat1 and mat2 shapes cannot be multiplied (4x29 and 13x128)".
from ntmo_pack import C_IN, C_OUT, C_IMG, LOCAL_N

KNOT_N = 8

KNOT_SPACING_MAX = 4.0     # max log2 rise per knot segment (~2.0 == identity)
KNOT_STEP_MIN = 0.05
KNOT_LOG_MIN = -14.0
KNOT_LOG_MAX = 8.0


class MonotoneCurveHead(nn.Module):
    """
    raw [B,8] -> strictly increasing knots [B,8], bounded to [log_min, log_max].

    Uses normalised softmax increments so the curve is monotone AND bounded
    *by construction* - no clipping, which would flatten ties at the limits:

        knot0 = kmin + (kmax - kmin) * sigmoid(raw_base)
        w     = softmax(raw_step)                      sum(w) == 1, w > 0
        span  = (kmax - knot0) * sigmoid(raw_span)     how far the curve reaches
        knot_i = knot0 + span * cumsum(w)_i

    The explicit span matters: without it the final knot is always pinned to kmax,
    which would forbid the network from ever choosing a lower highlight ceiling
    (exactly the decision that matters on a dim panel).

    At the shipped init we get knot0 ~= -8.1 and knots ending near +5.9, i.e. very
    close to identity. Softmax / Sigmoid / CumSum are all on the
    NNERuntimeRDGHlsl supported list, so this exports cleanly.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("kmin", torch.tensor(KNOT_LOG_MIN))
        self.register_buffer("kmax", torch.tensor(KNOT_LOG_MAX))

        # Per-knot mean / std measured over the generated dataset. Using these as
        # the head's origin is what makes the head trainable:
        #
        #   old: base = -14 + 22*sigmoid(z)   ->  d(base)/dz ~= 5.5
        #   new: base = mean0 + 2*std0*tanh(z) ->  d(base)/dz ~= 1.7
        #
        # With the old form a single epoch of Adam updates could translate the
        # whole curve by ~4 log2 units, so val swung between 0.88 and 11.7 while
        # train decreased smoothly. The tanh form is 3x less sensitive AND starts
        # the network on the dataset's mean curve instead of a random one.
        self.register_buffer("kmean", torch.tensor([
            -9.13, -7.07, -4.81, -2.51, -0.37, 1.23, 2.17, 2.50]))
        self.register_buffer("kstd", torch.tensor([
            1.69, 1.72, 1.67, 1.69, 1.76, 1.62, 1.49, 1.56]))

        # [7,8] prefix-sum matrix: (w @ M)[j] == sum_{k < j} w[k], so column 0 is 0
        # and column j is cumsum(w)[j-1]. MatMul instead of Pad/Concat because Pad
        # lowers to Concat in ONNX export and the RDG HLSL runtime has no Concat.
        M = torch.zeros(7, 8)
        for k in range(7):
            for j in range(k + 1, 8):
                M[k, j] = 1.0
        self.register_buffer("cum_matrix", M)

    def forward(self, raw):
        # 8 offset logits + 1 span logit.
        base = self.kmean[0] + 2.0 * self.kstd[0] * torch.tanh(raw[:, :1])   # [B,1]

        # Softmax with a floor. A pure softmax saturates to one-hot under large
        # logits, which lets consecutive knots coincide and silently breaks strict
        # monotonicity. Flooring every increment guarantees knots[i] < knots[i+1].
        eps = 2e-3
        w = (1.0 - 7.0 * eps) * torch.softmax(raw[:, 1:8], dim=1) + eps        # [B,7]

        # The span is floored as well: with a bare sigmoid the span can approach 0
        # for extreme logits, which collapses every knot onto `base` and reintroduces
        # a degenerate (flat) curve no matter how large the increment floor is.
        full_span = (self.kmean[7] - self.kmean[0]) \
            + 2.0 * (self.kstd[0] + self.kstd[7])
        span = full_span * (0.05 + 0.95 * torch.sigmoid(raw[:, 8:9]))         # [B,1]

        cum_p = w @ self.cum_matrix                                          # [B,8], col 0 == 0
        knots = base + span * cum_p
        return knots


class ImageBranch(nn.Module):
    """C_IMG x 72 x 128 (PQ rgb + edge) -> descriptor [B, out_dim]."""

    def __init__(self, width=32, depth=4, out_dim=128):
        super().__init__()
        blocks = []
        cin = C_IMG
        w = width
        for _ in range(depth):
            blocks.append(nn.Conv2d(cin, w, 3, stride=2, padding=1, bias=False))
            # momentum=None -> cumulative moving average instead of EMA(0.1).
            # Why it matters: with EMA the running stats could be dragged around by
            # one atypical epoch, and because validation runs in eval() (running
            # stats) while training runs in train() (batch stats), the two diverged
            # wildly -- val read 5.2 or 11.7 at epochs whose train loss was the best
            # of the run. A cumulative average makes the eval-time stats stable.
            blocks.append(nn.BatchNorm2d(w, momentum=None))
            blocks.append(nn.ReLU(inplace=True))
            cin = w
            w = min(w * 2, width * 4)

        self.features = nn.Sequential(*blocks)
        # Explicit ReduceMean over H,W: exports as Reduce, not GlobalAveragePool.
        self.proj = nn.Conv2d(cin, out_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim, momentum=None)

    def forward(self, x):
        f = self.features(x)
        f = self.proj(f)
        f = self.bn(f)
        return f.mean(dim=(2, 3))            # [B, out_dim]


class CurveNet(nn.Module):
    def __init__(self, width=32, depth=4, dim=128):
        super().__init__()
        # 4 image channels: PQ rgb + edge magnitude. The edge map is the paper's
        # "conditioned on a Canny edge map" input -- it is what tells the local filter
        # how hard it may push before halos appear.
        self.image_ch = C_IMG
        ctx_ch = C_IN - self.image_ch

        self.image_branch = ImageBranch(width, depth, dim)

        self.ctx_branch = nn.Sequential(
            nn.Linear(ctx_ch, 128), nn.ReLU(inplace=True),
            nn.Linear(128, dim), nn.ReLU(inplace=True),
        )

        # Fusion by ADD: avoids Concat, which the RDG HLSL runtime does not expose.
        # Dropout is light on purpose: identity at inference, exports to nothing.
        self.trunk = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.05),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.05),
        )

        self.head_knots = nn.Linear(128, KNOT_N + 1)   # 1 base + 7 steps + 1 span
        self.norm = nn.LayerNorm(128)                  # keeps the curve head out of tanh saturation

        # Direct linear path from the display context to the curve parameters.
        #
        # Without it the model collapsed to a CONSTANT curve: prediction std was
        # 0.001 against a target std of 1.7, and corr(pred knot0, true knot0) was
        # 0.017. Two things conspired -- LayerNorm renormalises the trunk output,
        # and a small head init means the trunk's contribution to `raw` is ~0.01,
        # so tanh(raw) is basically zero and the head emits the dataset mean for
        # every input. The display parameters (panel peak, full-frame nits, ABL
        # ratio) are strong, low-dimensional predictors of the curve, so giving
        # them their own linear term fixes the conditioning and gives training a
        # short gradient path.
        self.ctx_head = nn.Linear(ctx_ch, KNOT_N + 1)

        self.curve = MonotoneCurveHead()

        # ---- local Laplacian strengths (v1.0.2) --------------------------------
        # 4x4 tiles x 3 bands (s_base, s_mid, s_high). This is the learned LLF
        # parameter map: it replaces the three hand-tuned constants (0.60/0.25/0.10)
        # that v1.0.1 used, so the local operator now adapts per tile to the image.
        self.head_local = nn.Linear(128, LOCAL_N)
        self.ctx_local = nn.Linear(ctx_ch, LOCAL_N)

        # Placement matrices. The graph must emit ONE tensor of 56 values, but Concat
        # is not executable by NNERuntimeRDGHlsl (and Pad lowers to Concat). MatMul by
        # a constant 0/1 matrix is supported and does the same job:
        #     out = knots @ place_knots + local @ place_local
        pk = torch.zeros(8, C_OUT)
        for i in range(8):
            pk[i, i] = 1.0
        pl = torch.zeros(LOCAL_N, C_OUT)
        for j in range(LOCAL_N):
            pl[j, 8 + j] = 1.0
        self.register_buffer("place_knots", pk)
        self.register_buffer("place_local", pl)

        # Initialisation: the untrained model must emit the DATASET MEAN curve while
        # still being able to condition on its input. tanh(0)=0 -> base = mean and
        # sigmoid(0.585)=0.642 -> span = mean range.
        nn.init.normal_(self.head_knots.weight, std=0.05)
        nn.init.normal_(self.ctx_head.weight, std=0.02)
        # Local head starts near zero (tanh(0)=0 -> no local correction), so an
        # untrained model degrades to the v1.0.1 behaviour instead of distorting.
        nn.init.normal_(self.head_local.weight, std=0.01)
        nn.init.zeros_(self.head_local.bias)
        nn.init.normal_(self.ctx_local.weight, std=0.01)
        nn.init.zeros_(self.ctx_local.bias)
        with torch.no_grad():
            self.head_knots.bias.zero_()
            self.head_knots.bias[0] = 0.0
            self.head_knots.bias[8] = 0.0     # span lands on the dataset mean range
            self.ctx_head.bias.zero_()

    def forward(self, x):                                 # [B,16,72,128]
        img = x[:, :self.image_ch]                        # [B,3,72,128]
        # Channels 3..15 are per-sample scalars broadcast over H,W by
        # assemble_tensor_torch, so a single spatial position recovers them.
        ctx = x[:, self.image_ch:, 0, 0]                  # [B,13]

        g = self.image_branch(img)
        c = self.ctx_branch(ctx)

        # LayerNorm before the head is NOT cosmetic -- without it the head dies.
        #
        # `h` is the output of ReLU layers, so it is strictly positive and its
        # magnitude drifts upward during training. The curve head reads it through
        # tanh/sigmoid, so a large |h| saturates the activation, the gradient there
        # becomes ~0, and the head is then pinned to a CONSTANT curve forever.
        # Measured symptom: prediction std across 10000 validation samples was
        # 0.001 per knot (i.e. one fixed curve for every image) and the error was
        # a flat 5.43 in every brightness/panel bucket. Normalising `h` keeps the
        # head's inputs O(1) so the curve stays responsive.
        h = self.norm(self.trunk(g + c))

        # Trunk (content) + direct display-context term. The additive ctx path is
        # what keeps the output from collapsing to a constant curve.
        raw = self.head_knots(h) + self.ctx_head(ctx)     # [B,9]
        knots = self.curve(raw)                           # [B,8]

        # ---- local strengths ---------------------------------------------------
        raw_local = self.head_local(h) + self.ctx_local(ctx)     # [B,48]
        local = torch.tanh(raw_local)                            # [-1,1]

        # Pack into the single [B,56] output the graph must produce.
        return knots @ self.place_knots + local @ self.place_local


def build_model(size="M"):
    if size == "S":
        return CurveNet(width=20, depth=3, dim=96)
    if size == "M":
        return CurveNet(width=32, depth=4, dim=128)
    if size == "L":
        return CurveNet(width=48, depth=5, dim=160)
    raise ValueError(f"unknown size {size}")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for s in ("S", "M", "L"):
        m = build_model(s).eval()
        n = count_params(m)
        x = torch.zeros(2, C_IN, 72, 128)
        with torch.no_grad():
            y = m(x)
        print(f"{s}: {n/1e3:.1f}K params  out={tuple(y.shape)}  "
              f"monotone={bool((y[:,1:8] > y[:,:7]).all())}  "
              f"knots0={[round(v,2) for v in y[0,:8].tolist()]}")
