"""Minimal GDN unit-operator smoke test — the "import and use" path.

Run from the repo root:

    python examples/use_gdn.py

It builds a tiny ONNX graph containing a custom ``GDN`` node, imports it with
AIDGE's generic ONNX importer, installs the GDN CPU implementation, runs one
inference, and compares with NumPy.

Needs only aidge-core + aidge-backend-cpu + aidge-onnx + onnx; no
torch/compressai/aidge_export_c.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aidge_core  # noqa: E402
import aidge_backend_cpu  # noqa: E402,F401
import aidge_onnx  # noqa: E402

from aidge_ext_c import install_gdn_cpu  # noqa: E402


def make_demo_onnx(path: Path, gamma: np.ndarray, beta: np.ndarray) -> None:
    c = int(beta.shape[0])
    node = helper.make_node(
        "GDN", ["input"], ["output"], name="gdn_demo",
        channels=c, inverse=0,
        gamma=gamma.reshape(-1).tolist(), beta=beta.tolist(),
    )
    graph = helper.make_graph(
        [node], "gdn_demo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c, 4, 4])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)], ir_version=7)
    onnx.save(model, str(path))


def main() -> None:
    rng = np.random.default_rng(0)
    c = 8
    gamma = (rng.random((c, c), dtype=np.float32) * 0.2).astype(np.float32)
    beta = (rng.random(c, dtype=np.float32) + 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "gdn_demo.onnx"
        make_demo_onnx(model_path, gamma, beta)
        graph = aidge_onnx.load_onnx(str(model_path))

    install_gdn_cpu(graph)
    graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, c, 4, 4]])
    scheduler = aidge_core.SequentialScheduler(graph)

    x = rng.random((1, c, 4, 4)).astype(np.float32)
    t = aidge_core.Tensor(x)
    t.to_backend("cpu")
    t.set_data_format(aidge_core.dformat.nchw)
    t.to_dtype(aidge_core.dtype.float32)
    y = np.array(scheduler.forward(False, [t])[0])

    ref = x / np.sqrt(np.einsum("oc,nchw->nohw", gamma, x * x) +
                      beta.reshape(1, c, 1, 1))
    print("output", y.shape, "range", y.min(), y.max())
    print("max |aidge - numpy| =", float(np.abs(y - ref).max()))
    if not np.allclose(y, ref, atol=1e-5, rtol=1e-4):
        raise SystemExit("GDN output mismatch")
    print("GDN OK")


if __name__ == "__main__":
    main()
