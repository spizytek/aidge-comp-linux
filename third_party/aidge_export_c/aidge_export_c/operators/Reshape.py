"""Reshape (data-preserving copy) C export node."""

import aidge_core

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


@ExportLibC.register(
    "Reshape",
    aidge_core.ImplSpec(
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.default)
    ),
    aidge_core.ProdConso.in_place_model,
)
class ReshapeC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "reshape_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "reshape_forward.jinja"
        )
        self.include_list = ["<string.h>"]
        self.kernels_to_copy = []
