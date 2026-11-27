"""Profile the Ballé factorized encoder ONNX at a fixed input size.

The public encoder ONNX builds its GDN 1x1-conv weights through Shape/Gather/
Reshape, so vanilla ONNX shape inference stops at the first GDN block.  This
script first freezes those data-dependent Reshape shapes with ONNX Runtime
(the same trick as test_acetone.py), then infers shapes and reports:

  * parameter count and size
  * MACs per operator / per layer
  * the largest activation tensors (fp32)

Usage:
    python tools/profile_encoder.py
    python tools/profile_encoder.py --input 1,3,64,64 --top 20
    python tools/profile_encoder.py --input 1,3,256,256 --json analysis/p256.json

Needs onnx, numpy, onnxruntime (use E:\\spaceviv\\.venv-acetone on Windows).
"""

from __future__ import annotations

import argparse
import collections
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.utils import extract_model


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "Model" / "bmshj2018_factorized_encoder.onnx"


def tensor_shape(value_info) -> list[int] | None:
    tt = value_info.type.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            return None
    return dims


def numel(shape: list[int] | None) -> int | None:
    if shape is None:
        return None
    n = 1
    for d in shape:
        n *= d
    return n


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
    return default


def prune_unused(graph: onnx.GraphProto) -> None:
    nodes = list(graph.node)
    producers = {out: i for i, n in enumerate(nodes) for out in n.output}
    needed: set[int] = set()
    stack = [o.name for o in graph.output]
    while stack:
        idx = producers.get(stack.pop())
        if idx is None or idx in needed:
            continue
        needed.add(idx)
        stack.extend(nodes[idx].input)
    kept = [n for i, n in enumerate(nodes) if i in needed]
    del graph.node[:]
    graph.node.extend(kept)
    used = {name for n in graph.node for name in n.input}
    kept_init = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)


def freeze_reshape_shapes(model: onnx.ModelProto, input_shape: list[int]) -> None:
    """Evaluate Shape/Gather/Concat-produced Reshape targets and freeze them."""
    import onnxruntime as ort

    graph = model.graph
    init_names = {i.name for i in graph.initializer}
    reshape_nodes = [n for n in graph.node
                     if n.op_type == "Reshape" and n.input[1] not in init_names]
    if not reshape_nodes:
        return

    shape_inputs = [n.input[1] for n in reshape_nodes]
    augmented = onnx.ModelProto()
    augmented.CopyFrom(model)
    existing = {o.name for o in augmented.graph.output}
    for name in shape_inputs:
        if name not in existing:
            augmented.graph.output.append(helper.make_empty_tensor_value_info(name))

    sess = ort.InferenceSession(augmented.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    values = sess.run(shape_inputs,
                      {input_name: np.zeros(input_shape, dtype=np.float32)})
    for index, (node, value) in enumerate(zip(reshape_nodes, values, strict=True)):
        name = f"static_reshape_shape_{index}"
        node.input[1] = name
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))


def prepare(model_path: Path, input_shape: list[int], out_dir: Path) -> onnx.ModelProto:
    source = onnx.load(str(model_path))
    output_names = [o.name for o in source.graph.output]
    encoder_output = "x_out" if "x_out" in output_names else output_names[0]

    trimmed = out_dir / "_encoder_trimmed.onnx"
    extract_model(str(model_path), str(trimmed),
                  input_names=[source.graph.input[0].name],
                  output_names=[encoder_output])
    model = onnx.load(str(trimmed))
    freeze_reshape_shapes(model, input_shape)
    prune_unused(model.graph)
    onnx.checker.check_model(model)

    graph = model.graph
    graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    del graph.input[0].type.tensor_type.shape.dim[:]
    for d in input_shape:
        graph.input[0].type.tensor_type.shape.dim.add().dim_value = d
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--input", default="1,3,64,64",
                    help="NCHW input shape (default 1,3,64,64)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    in_shape = [int(x) for x in args.input.split(",")]
    with tempfile.TemporaryDirectory() as td:
        model = prepare(args.model, in_shape, Path(td))
    inferred = shape_inference.infer_shapes(model, strict_mode=False,
                                            data_prop=True)

    shapes: dict[str, list[int] | None] = {}
    for coll in (inferred.graph.input, inferred.graph.value_info,
                 inferred.graph.output):
        for vi in coll:
            shapes[vi.name] = tensor_shape(vi)

    init = {i.name: i for i in model.graph.initializer}
    init_elems = {name: int(np.prod(i.dims)) for name, i in init.items()}
    dtype_bytes = {
        name: np.dtype(onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).itemsize
        for name, i in init.items()
    }
    init_bytes = sum(init_elems[n] * dtype_bytes[n] for n in init)

    rows = []
    by_op_macs = collections.Counter()
    by_op_params = collections.Counter()
    by_op_count = collections.Counter()
    activations: list[tuple[int, str, list[int]]] = []

    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        out_shape = shapes.get(node.output[0]) if node.output else None
        params = sum(init_elems[n] for n in node.input if n in init)
        macs = 0

        if op == "Conv" and out_shape is not None:
            w_shape = shapes.get(node.input[1])
            if w_shape is None and node.input[1] in init:
                w_shape = list(init[node.input[1]].dims)
            if w_shape is not None:
                group = attr(node, "group", 1)
                n = out_shape[0]
                spatial = int(np.prod(out_shape[2:])) if len(out_shape) > 2 else 1
                c_in = w_shape[1] * group
                kernel = int(np.prod(w_shape[2:])) if len(w_shape) > 2 else 1
                macs = n * out_shape[1] * spatial * (c_in // group) * kernel
        elif op in ("Gemm", "MatMul") and out_shape is not None:
            a_shape = shapes.get(node.input[0])
            if a_shape is None and node.input[0] in init:
                a_shape = list(init[node.input[0]].dims)
            if a_shape is not None:
                macs = int(np.prod(out_shape)) * a_shape[-1]

        by_op_count[op] += 1
        by_op_macs[op] += macs
        by_op_params[op] += params
        if out_shape is not None:
            elem = numel(out_shape)
            if elem is not None:
                activations.append((elem, node.output[0], out_shape))

        rows.append({"idx": idx, "op": op,
                     "name": node.name or node.output[0],
                     "output": node.output[0], "out_shape": out_shape,
                     "params": params, "macs": macs})

    total_macs = sum(by_op_macs.values())
    total_params = sum(init_elems.values())
    unknown = sum(1 for r in rows if r["out_shape"] is None)

    # --- workspace with perfect buffer reuse (liveness allocator) ---
    nodes = model.graph.node
    last_use: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for name in node.input:
            if name not in init:
                last_use[name] = i
    live: set[str] = set()
    peak_bytes = 0
    peak_step = -1
    for i, node in enumerate(nodes):
        live.update(o for o in node.output if o not in init)
        for name in list(live):
            if last_use.get(name, len(nodes)) < i:
                live.discard(name)
        size = sum(4 * (numel(shapes.get(n)) or 0) for n in live)
        if size > peak_bytes:
            peak_bytes = size
            peak_step = i
    graph_out_bytes = sum(4 * (numel(shapes.get(o.name)) or 0)
                          for o in model.graph.output)

    print(f"model      : {args.model}")
    print(f"input      : {in_shape}")
    print(f"nodes      : {len(model.graph.node)}  (unknown-shape: {unknown})")
    print(f"params     : {total_params:,}  ({init_bytes/1e6:.2f} MB fp32)")
    print(f"MACs       : {total_macs/1e6:.2f} M  "
          f"({total_macs/1e9:.3f} G) at {in_shape[2]}x{in_shape[3]}")
    print(f"workspace  : {peak_bytes/1e6:.2f} MB fp32 peak with perfect reuse "
          f"(node {peak_step});  {peak_bytes/4/1e3:.0f} KB if int8")
    print()
    print("By operator:")
    for op in sorted(by_op_macs, key=lambda o: by_op_macs[o], reverse=True):
        if by_op_macs[op] == 0 and by_op_params[op] == 0:
            continue
        share = 100.0 * by_op_macs[op] / total_macs if total_macs else 0.0
        print(f"  {op:12s} count={by_op_count[op]:3d}  "
              f"MACs={by_op_macs[op]/1e6:9.2f} M ({share:5.1f}%)  "
              f"params={by_op_params[op]:,}")
    print()
    print(f"Top {args.top} operators by MACs:")
    for r in sorted(rows, key=lambda r: r["macs"], reverse=True)[:args.top]:
        shape = "?" if r["out_shape"] is None else "x".join(map(str, r["out_shape"]))
        print(f"  [{r['idx']:3d}] {r['op']:9s} {r['name'][:46]:46s} "
              f"{shape:18s} MACs={r['macs']/1e6:8.2f} M params={r['params']:,}")
    print()
    print("Largest activation tensors (fp32, >16 KB):")
    for elem, name, shape in sorted(activations, reverse=True):
        if elem * 4 < 16 * 1024:
            break
        print(f"  {elem*4/1024:8.1f} KB  {name[:64]:64s} {shape}")

    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "model": str(args.model), "input": in_shape,
            "total_params": total_params, "total_macs": total_macs,
            "by_op": {op: {"count": by_op_count[op],
                           "macs": by_op_macs[op],
                           "params": by_op_params[op]} for op in by_op_count},
            "nodes": rows,
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
