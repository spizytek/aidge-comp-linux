"""Build a clean encoder ONNX: 4x Conv2D + 3x GDN unit operator.

The GDN node is not a native ONNX op.  AIDGE's generic importer turns unknown
ops into ``GenericOperator`` nodes and copies their attributes, so the graph is
just:

    input -> conv0 -> GDN -> conv2 -> GDN -> conv4 -> GDN -> conv6 -> x_out

No Pow/Sqrt/Div/Add/Mul chains.  This is the graph we quantize (convs int8,
GDN float) and export to C.

Usage:
    python tools/build_gdn_unit_onnx.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from compressai.zoo import bmshj2018_factorized
from compressai.layers import GDN

ROOT = Path(__file__).resolve().parent.parent


def eff_params(gdn: GDN):
    with torch.no_grad():
        beta = gdn.beta_reparam(gdn.beta).reshape(-1).numpy().astype(np.float32)
        gamma = gdn.gamma_reparam(gdn.gamma).numpy().astype(np.float32)
    return gamma, beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis" / "encoder_gdn_unit_64.onnx")
    args = ap.parse_args()

    model = bmshj2018_factorized(quality=args.quality, pretrained=True).eval()
    g_a = model.g_a

    conv_nodes = []
    inits = []
    gdn_params = {}
    prev = "input"
    names = ["c0", "c2", "c4", "x_out"]
    conv_idx = [0, 2, 4, 6]
    gdn_idx = [1, 3, 5]
    gdn_names = ["g1", "g3", "g5"]

    # conv0 -> gdn1 -> conv2 -> gdn3 -> conv4 -> gdn5 -> conv6
    for k, (ci, cname) in enumerate(zip(conv_idx, names)):
        conv = g_a[ci]
        w = conv.weight.detach().numpy().astype(np.float32)
        b = conv.bias.detach().numpy().astype(np.float32)
        inits.append(numpy_helper.from_array(w, f"{cname}_w"))
        inits.append(numpy_helper.from_array(b, f"{cname}_b"))
        out = cname
        conv_nodes.append(helper.make_node(
            "Conv", [prev, f"{cname}_w", f"{cname}_b"], [out],
            name=f"conv_{ci}",
            kernel_shape=[5, 5], strides=[2, 2], pads=[2, 2, 2, 2],
        ))
        prev = out

        if k < len(gdn_idx):
            gdn = g_a[gdn_idx[k]]
            gamma, beta = eff_params(gdn)
            c = beta.shape[0]
            gname = gdn_names[k]
            # Proper GDN unit operator: a single custom ONNX node.  AIDGE's
            # generic importer creates a GenericOperator("GDN") and copies the
            # gamma/beta/channels attributes.
            pname = f"gdn_{gdn_idx[k]}"
            gdn_params[f"gamma_{k}"] = gamma
            gdn_params[f"beta_{k}"] = beta
            conv_nodes.append(helper.make_node(
                "GDN", [prev], [gname], name=pname,
                channels=int(c),
                inverse=0,
                gamma=gamma.reshape(-1).tolist(),
                beta=beta.tolist(),
            ))
            prev = gname

    graph = helper.make_graph(
        conv_nodes,
        "balle_encoder_gdn_unit",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 3, args.size, args.size])],
        [helper.make_tensor_value_info("x_out", TensorProto.FLOAT,
                                       [1, 320, args.size // 16, args.size // 16])],
        initializer=inits,
    )
    m = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=7,
    )
    try:
        onnx.checker.check_model(m)
    except Exception as exc:  # custom GDN op is not in the ONNX schema
        print("onnx checker skipped:", exc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.2f} MB)")
    print(f"nodes: {len(graph.node)}  initializers: {len(inits)}")
    np.savez(args.out.with_name("gdn_unit_params.npz"), **gdn_params)
    print("saved GDN params ->", args.out.with_name("gdn_unit_params.npz"))

    # save a sample + exact reference
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, args.size, args.size), dtype=np.float32)
    with torch.no_grad():
        y = model.g_a(torch.from_numpy(x)).numpy()
    np.save(args.out.with_name("_unit_sample.npy"), x)
    np.save(args.out.with_name("_unit_exact.npy"), y)
    print("sample latent", y.shape)


if __name__ == "__main__":
    main()
