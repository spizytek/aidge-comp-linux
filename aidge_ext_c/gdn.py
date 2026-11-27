"""AIDGE GDN / IGDN unit operator (runtime part).

Use this when you only need to build or import a graph containing GDN:

    from aidge_ext_c import register_gdn_cpu, make_gdn_node
    register_gdn_cpu()

    gdn = make_gdn_node(gamma, beta)          # gamma [C,C], beta [C]
    ...
    compile(...)

The graph node type is ``GDN`` with attributes ``channels``, ``inverse``,
``gamma`` and ``beta``.  AIDGE's generic ONNX importer creates exactly this
node when it reads an ONNX node with ``op_type="GDN"``; call
``install_gdn_cpu(graph)`` to attach the CPU implementation before compiling.

The C export backend is separate (``gdn_export_c.py``) so importing this
module does not require ``aidge_export_c``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aidge_core

ROOT = Path(__file__).resolve().parent.parent
EXT = Path(__file__).resolve().parent

_CPU_REGISTERED = False


class GDNImpl(aidge_core.OperatorImpl):
    """NumPy CPU implementation used for inference and PTQ calibration."""

    def __init__(self, op: aidge_core.Operator):
        aidge_core.OperatorImpl.__init__(self, op, "cpu")
        self.op = op

    def forward(self):
        op = self.op
        c = int(op.attr.channels)
        x = np.array(op.get_input(0)).astype(np.float32)
        gamma = np.array(op.attr.gamma, dtype=np.float32).reshape(c, c)
        beta = np.array(op.attr.beta, dtype=np.float32).reshape(1, c, 1, 1)
        norm = np.einsum("oc,nchw->nohw", gamma, x * x) + beta
        y = x / np.sqrt(norm)
        t = aidge_core.Tensor(y)
        t.to_backend("cpu")
        try:
            t.set_data_format(op.get_input(0).dformat)
        except Exception:
            pass
        op.set_output(0, t)


def register_gdn_cpu() -> None:
    """Register the CPU implementation for GenericOperator('GDN')."""
    global _CPU_REGISTERED
    if _CPU_REGISTERED:
        return
    aidge_core.register_GenericOperatorOp(
        ["cpu", "GDN"], lambda op, backend: GDNImpl(op))
    _CPU_REGISTERED = True


def make_gdn_node(gamma, beta, inverse: bool = False,
                  name: str = "gdn") -> aidge_core.Node:
    """Create a single GDN/IGDN node.

    gamma: (C, C) effective gamma (already non-negative)
    beta:  (C,)   effective beta  (already non-negative)
    """
    register_gdn_cpu()
    gamma = np.asarray(gamma, dtype=np.float32).reshape(-1)
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    c = int(beta.shape[0])
    if gamma.shape[0] != c * c:
        raise ValueError(
            f"gamma must contain {c*c} values, got {gamma.shape[0]}")

    node = aidge_core.GenericOperator(
        "GDN", 1, 0, 1, name=name)
    op = node.get_operator()
    op.attr.channels = c
    op.attr.inverse = bool(inverse)
    op.attr.gamma = [float(v) for v in gamma]
    op.attr.beta = [float(v) for v in beta]
    op.set_forward_dims(lambda dims: dims)  # shape-preserving
    return node


def install_gdn_cpu(graph) -> int:
    """Attach forward_dims + CPU backend to every GDN node in an imported graph."""
    register_gdn_cpu()
    count = 0
    for node in graph.get_nodes():
        if node.type() == "GDN":
            op = node.get_operator()
            op.set_forward_dims(lambda dims: dims)
            op.set_backend("cpu")
            count += 1
    return count
