"""Make a static-shape, encoder-only ONNX for PTQ / profiling.

The public bmshj2018 encoder computes its GDN 1x1-conv weight shapes at runtime
(Shape/Gather/Concat/Reshape).  For a fixed input size those values are
constant, so:

  1. extract the subgraph that produces ``x_out`` (drop the entropy helpers),
  2. evaluate every dynamic Reshape target with ONNX Runtime and freeze it,
  3. prune nodes/initializers that no longer feed x_out.

Usage:
    python tools/prepare_encoder_onnx.py --size 64
    python tools/prepare_encoder_onnx.py --size 64 --out analysis/encoder_static_64.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import extract_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "Model" / "bmshj2018_factorized_encoder.onnx"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=64, help="square input size")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    input_shape = [1, 3, args.size, args.size]
    out = args.out or (ROOT / "analysis" / f"encoder_static_{args.size}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    source = onnx.load(str(args.model))
    output_names = [o.name for o in source.graph.output]
    encoder_output = "x_out" if "x_out" in output_names else output_names[0]
    tmp = out.with_name(out.stem + "_trimmed.onnx")

    extract_model(str(args.model), str(tmp),
                  input_names=[source.graph.input[0].name],
                  output_names=[encoder_output])
    model = onnx.load(str(tmp))
    freeze_reshape_shapes(model, input_shape)
    prune_unused(model.graph)
    onnx.checker.check_model(model)

    graph = model.graph
    graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    del graph.input[0].type.tensor_type.shape.dim[:]
    for d in input_shape:
        graph.input[0].type.tensor_type.shape.dim.add().dim_value = d

    onnx.save(model, str(out))
    tmp.unlink(missing_ok=True)

    # verify against the original model on a deterministic input
    import onnxruntime as ort
    x = (np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
         % 255.0 / 255.0)
    orig = ort.InferenceSession(str(args.model),
                                providers=["CPUExecutionProvider"])
    stat = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y0 = orig.run([encoder_output], {orig.get_inputs()[0].name: x})[0]
    y1 = stat.run(None, {stat.get_inputs()[0].name: x})[0]
    print(f"wrote {out}")
    print(f"  nodes {len(model.graph.node)}, initializers {len(model.graph.initializer)}")
    print(f"  x_out {y1.shape}, max |static - original| = {np.abs(y0-y1).max():.3e}")


if __name__ == "__main__":
    main()
