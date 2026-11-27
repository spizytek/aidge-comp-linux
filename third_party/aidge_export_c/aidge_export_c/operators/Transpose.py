"""N-dimensional transpose C export node."""

import aidge_core

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


@ExportLibC.register(
    "Transpose",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class TransposeC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        nbdims = len(self.attributes["in_dims"][0])
        order = list(self.attributes["output_dims_order"])
        out_dims = [self.attributes["in_dims"][0][order[i]] for i in range(nbdims)]

        out_strides = [0] * nbdims
        out_strides[nbdims - 1] = 1
        for i in range(nbdims - 2, -1, -1):
            out_strides[i] = out_strides[i + 1] * out_dims[i + 1]

        self.attributes["out_strides"] = out_strides
        self.attributes["output_dims_order"] = order

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "transpose_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "transpose_forward.jinja"
        )
        self.include_list = ["<stddef.h>", "kernels/c/transpose.h"]
        self.kernels_to_copy = []
