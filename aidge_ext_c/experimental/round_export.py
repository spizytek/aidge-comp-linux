"""Round/Clip C exporters needed by the expanded Quantizer/Dequantizer."""

from __future__ import annotations

from pathlib import Path

import aidge_core
from aidge_export_c import ExportLibC
from aidge_export_c.node_export import ExportNodeC

EXT = Path(__file__).resolve().parent


@ExportLibC.register(
    "Round",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class RoundC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.config_template = str(EXT / "templates" / "round_config.jinja")
        self.forward_template = str(EXT / "templates" / "round_forward.jinja")
        self.include_list = ["utils/c/typedefs.h"]
        self.kernels_to_copy = []


@ExportLibC.register(
    "Clip",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class ClipC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        op = self.operator
        self.attributes["clip_min"] = float(getattr(op.attr, "min", -3.4e38))
        self.attributes["clip_max"] = float(getattr(op.attr, "max", 3.4e38))
        self.config_template = str(EXT / "templates" / "clip_config.jinja")
        self.forward_template = str(EXT / "templates" / "clip_forward.jinja")
        self.include_list = ["utils/c/typedefs.h"]
        self.kernels_to_copy = []
