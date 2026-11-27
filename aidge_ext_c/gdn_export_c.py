"""aidge_export_c backend for the GDN unit operator.

Optional: only import this when generating C.

    from aidge_ext_c import register_gdn_export
    register_gdn_export()

    from aidge_core.export_utils import export
    from aidge_core.export_utils.config import ExportConfig
    export(graph, ExportConfig(target="host_c", dtype="float32", ...), ...)

For int8, AIDGE's PTQ must know that GDN is not quantized.  Apply
``aidge_ext_c/patches/0001-aidge-quantization-gdn.md`` and rebuild
``aidge_quantization`` first; otherwise PTQ stops with
``makeStrategy(): no strategy registered for node type '... [GDN]'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aidge_core

try:
    from aidge_export_c import ExportLibC
    from aidge_export_c.node_export import ExportNodeC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aidge_export_c is required for C export. Install it first, or use "
        "the runtime-only aidge_ext_c.gdn module."
    ) from exc

EXT = Path(__file__).resolve().parent

_REGISTERED = False


class GdnC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        op = self.operator
        channels = int(op.attr.channels)
        gamma = np.array(op.attr.gamma, dtype=np.float32).reshape(-1)
        beta = np.array(op.attr.beta, dtype=np.float32).reshape(-1)

        self.attributes["channels"] = channels
        self.attributes["gamma"] = [float(v) for v in gamma]
        self.attributes["beta"] = [float(v) for v in beta]

        in_dtype = op.get_input(0).dtype
        self.attributes["is_int8"] = in_dtype == aidge_core.dtype.int8
        self.attributes["in_scale"] = 1.0
        if self.attributes["is_int8"] and self.get_quantizers():
            try:
                self.set_scaling_attributes()
                self.attributes["in_scale"] = float(
                    self.attributes.get("coef_value", 1.0))
            except Exception:
                pass

        self.config_template = str(EXT / "templates" / "gdn_config.jinja")
        self.forward_template = str(EXT / "templates" / "gdn_forward.jinja")
        self.include_list = ["kernels/c/gdn.h"]
        self.kernels_to_copy = []
        self.add_kernel_to_copy(str(EXT / "gdn.h"), "include/kernels/c")


def register_gdn_export() -> None:
    """Register the C exporter for GDN (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    ExportLibC.register_generic(
        "GDN",
        aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.float32)),
        aidge_core.ProdConso.in_place_model,
    )(GdnC)
    _REGISTERED = True
