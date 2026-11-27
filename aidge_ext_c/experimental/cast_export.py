"""Minimal Cast C exporter (needed by the expanded Dequantizer micro-graph)."""

from __future__ import annotations

from pathlib import Path

import aidge_core
from aidge_export_c import ExportLibC
from aidge_export_c.node_export import ExportNodeC

EXT = Path(__file__).resolve().parent


@ExportLibC.register(
    "Cast",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class CastC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.config_template = str(EXT / "templates" / "cast_config.jinja")
        self.forward_template = str(EXT / "templates" / "cast_forward.jinja")
        self.include_list = ["utils/c/typedefs.h"]
        self.kernels_to_copy = []
