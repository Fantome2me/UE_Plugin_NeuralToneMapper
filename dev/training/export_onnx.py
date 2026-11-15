"""
Export a trained NTMO checkpoint to ONNX and validate it against the operator
set that UE 5.8's NNERuntimeRDGHlsl actually implements.

    python export_onnx.py --ckpt /data/ntmo/run/best.pt --out /data/ntmo/run

Produces:
    ntmo_curve.onnx      (opset 17, fixed shape [1,16,72,128])
    ntmo_curve_report.json
"""

import argparse
import json
import os

import numpy as np
import torch

from ntmo_model import C_IN, C_OUT, build_model, count_params
from ntmo_pack import H, W

# Verified against Engine/Plugins/Experimental/NNERuntimeRDG/Shaders/Private/NNEHlslShaders/
SUPPORTED_OPS = {
    "Conv", "ConvTranspose", "Gemm", "MatMul",
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "Pool",
    "Resize", "Upsample", "Pad", "Slice", "Split", "Transpose",
    "Gather", "GatherElements", "ScatterND", "Softmax", "ReduceMean", "ReduceSum",
    "ReduceMax", "ReduceMin", "CumSum", "Cast", "Concat",
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Softplus", "Clip", "Abs", "Exp", "Log", "Neg",
    "Add", "Sub", "Mul", "Div", "Min", "Max", "Pow", "Sqrt",
    "Identity", "Constant", "ConstantOfShape", "Shape", "Reshape", "Flatten", "Squeeze",
    "Unsqueeze", "Expand", "Tile", "Where", "Erf",
}

# Ops we know the RDG HLSL runtime does NOT expose. Keep this list loud.
KNOWN_UNSUPPORTED = {"Concat", "GroupNormalization", "LSTM", "GRU", "Attention",
                     "MultiHeadAttention", "Einsum", "NonMaxSuppression"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    size = ck.get("size", "M")

    model = build_model(size)
    # strict=False: newer checkpoints may lack deterministic constant buffers that
    # were added later (e.g. the prefix-sum matrix). Missing non-learnable buffers
    # are reconstructed in __init__ and are mathematically identical.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"[export] note: buffers/count missing from checkpoint: {missing}")
    if unexpected:
        print(f"[export] note: unused checkpoint entries: {unexpected}")
    model.eval()

    n_params = count_params(model)
    print(f"[export] size={size} params={n_params/1e3:.1f}K "
          f"epoch={ck.get('epoch')} best_val={ck.get('best')}")

    dummy = torch.zeros(1, C_IN, H, W)

    onnx_path = os.path.join(args.out, "ntmo_curve.onnx")
    # dynamo=False -> legacy TorchScript exporter. torch >= 2.9 defaults to the new
    # dynamo exporter, which imports `onnxscript` and emits a different (decomposed)
    # op set. The legacy path is what produced the op list this project validated
    # against NNERuntimeRDGHlsl, so pin it explicitly instead of inheriting a default.
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["scene"], output_names=["curve"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,           # fixed shape: NNERuntimeRDG prefers this
        dynamo=False,
    )
    print(f"[export] wrote {onnx_path}")

    # ---- validate --------------------------------------------------------
    import onnx

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)

    ops = sorted({n.op_type for n in m.graph.node})
    unknown = [o for o in ops if o not in SUPPORTED_OPS]
    known_bad = [o for o in ops if o in KNOWN_UNSUPPORTED]

    # numeric check against pytorch
    ok_numeric = True
    max_err = 0.0
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        for _ in range(4):
            x = rng.random((1, C_IN, H, W), dtype=np.float32)
            yo = sess.run(None, {"scene": x})[0]
            with torch.no_grad():
                yt = model(torch.from_numpy(x)).numpy()
            max_err = max(max_err, float(np.abs(yo - yt).max()))
        ok_numeric = max_err < 1e-3
        mono = True
        for k in range(8):
            xk = np.random.default_rng(k + 1).random((1, C_IN, H, W), dtype=np.float32)
            yk = sess.run(None, {"scene": xk})[0]
            if not bool((np.diff(yk[:, :8], axis=1) > 0).all()):
                mono = False
                break
        # Report the REAL output arity instead of the dataset label width.
        actual_out = [int(d) for d in yk.shape]
    except ImportError:
        mono = None
        actual_out = None
        print("[export] onnxruntime not installed, skipping numeric check")

    report = {
        "onnx": os.path.abspath(onnx_path),
        "opset": args.opset,
        "size": size,
        "params": n_params,
        "input_shape": [1, C_IN, H, W],
        "output_shape": actual_out if actual_out is not None else [1, C_OUT],
        "ops": ops,
        "unexpected_ops": unknown,
        "known_unsupported_ops": known_bad,
        "numeric_max_err": max_err,
        "numeric_ok": ok_numeric,
        "monotone_outputs": mono,
        "deployable": (len(unknown) == 0 and len(known_bad) == 0 and ok_numeric),
    }

    rp = os.path.join(args.out, "ntmo_curve_report.json")
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[export] ops: {ops}")
    if unknown:
        print(f"[export] !! ops not in the verified RDG set: {unknown}")
    if known_bad:
        print(f"[export] !! ops the RDG runtime does not expose: {known_bad}")
    print(f"[export] numeric max err = {max_err:.3e}  monotone={mono}")
    print(f"[export] DEPLOYABLE = {report['deployable']}")
    print(f"[export] report -> {rp}")

    if not report["deployable"]:
        print("[export] WARNING: this graph may not load on NNERuntimeRDGHlsl. "
              "Consider --ops 16, or switch the curve source to Analytic.")


if __name__ == "__main__":
    main()
