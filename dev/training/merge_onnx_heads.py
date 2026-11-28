"""Merge a frozen shipped curve head with a pair-trained local head."""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rename_graph(graph, prefix, preserved_inputs):
    names = set()
    for node in graph.node:
        names.update(n for n in node.input if n)
        names.update(n for n in node.output if n)
    names.update(x.name for x in graph.initializer)
    names.update(x.name for x in graph.value_info)
    names.update(x.name for x in graph.output)
    names.update(x.name for x in graph.input)
    mapping = {n: n if n in preserved_inputs else prefix + n for n in names}
    for node in graph.node:
        node.name = prefix + node.name
        node.input[:] = [mapping.get(n, n) for n in node.input]
        node.output[:] = [mapping.get(n, n) for n in node.output]
    for value in list(graph.initializer) + list(graph.value_info) + list(graph.output):
        value.name = mapping.get(value.name, value.name)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="shipped model whose global curve is retained")
    ap.add_argument("--local", required=True, help="trained model supplying local parameters")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = onnx.load(args.base)
    local = onnx.load(args.local)
    if len(base.graph.input) != 1 or len(local.graph.input) != 1:
        raise ValueError("expected one input per model")
    if base.graph.input[0].name != local.graph.input[0].name:
        raise ValueError("model input names differ")
    if base.graph.input[0].type.tensor_type.shape != local.graph.input[0].type.tensor_type.shape:
        raise ValueError("model input shapes differ")
    if len(base.graph.output) != 1 or len(local.graph.output) != 1:
        raise ValueError("expected one output per model")

    input_name = base.graph.input[0].name
    bmap = rename_graph(base.graph, "base_", {input_name})
    lmap = rename_graph(local.graph, "local_", {input_name})
    base_out = base.graph.output[0].name
    local_out = local.graph.output[0].name

    graph = helper.make_graph(
        list(base.graph.node) + list(local.graph.node),
        "NTMO_v13_curve_plus_pair_trained_local",
        [base.graph.input[0]],
        [helper.make_tensor_value_info("merged_output", TensorProto.FLOAT, [1, 72])],
        initializer=list(base.graph.initializer) + list(local.graph.initializer),
        value_info=list(base.graph.value_info) + list(local.graph.value_info),
    )
    graph.node.extend([
        helper.make_node("Gather", [base_out, "curve_indices"], ["base_knots"], axis=1),
        helper.make_node("MatMul", ["base_knots", "curve_place"], ["curve_placed"]),
        helper.make_node("Gather", [local_out, "local_indices"], ["local_values"], axis=1),
        helper.make_node("MatMul", ["local_values", "local_place"], ["local_placed"]),
        helper.make_node("Add", ["curve_placed", "local_placed"], ["merged_output"]),
    ])
    graph.initializer.extend([
        numpy_helper.from_array(np.arange(8, dtype=np.int64), "curve_indices"),
        numpy_helper.from_array(np.arange(8, 72, dtype=np.int64), "local_indices"),
        numpy_helper.from_array(np.eye(8, 72, dtype=np.float32), "curve_place"),
        numpy_helper.from_array(np.pad(np.eye(64, dtype=np.float32), ((0, 0), (8, 0))), "local_place"),
    ])
    opsets = {x.domain: x.version for x in base.opset_import}
    for op in local.opset_import:
        opsets[op.domain] = max(opsets.get(op.domain, 0), op.version)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(k, v) for k, v in opsets.items()])
    model.ir_version = max(base.ir_version, local.ir_version)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)

    import onnxruntime as ort
    a = ort.InferenceSession(args.base, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(args.local, providers=["CPUExecutionProvider"])
    c = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(117)
    max_curve, max_local = 0.0, 0.0
    for _ in range(8):
        x = rng.random((1, 49, 72, 128), dtype=np.float32)
        ya = a.run(None, {a.get_inputs()[0].name: x})[0]
        yb = b.run(None, {b.get_inputs()[0].name: x})[0]
        yc = c.run(None, {c.get_inputs()[0].name: x})[0]
        max_curve = max(max_curve, float(np.max(np.abs(yc[:, :8] - ya[:, :8]))))
        max_local = max(max_local, float(np.max(np.abs(yc[:, 8:] - yb[:, 8:]))))
    print({"max_curve_diff_from_base": max_curve, "max_local_diff_from_trained": max_local,
           "output_shape": c.get_outputs()[0].shape, "output_bytes": __import__("os").path.getsize(args.out)})
    if max_curve > 1e-6 or max_local > 1e-5:
        raise SystemExit("head merge parity failed")


if __name__ == "__main__":
    main()
