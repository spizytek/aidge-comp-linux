"""Standalone Quantizer C export node."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


@ExportLibC.register_metaop(
    "Quantizer",
    "Producer-1>Add#<-(Mul|BitShift#)<1-Producer; Add#->Clip#->Cast; "
    "(Clip#<*-Producer)*; BitShift#<-(Mul<1-Producer)?",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class QuantizerC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        output_tensor = self.operator.get_output(0)
        self.attributes["is_int8"] = output_tensor.dtype == aidge_core.dtype.int8
        self.attributes["shift_value"] = 0
        self.attributes["scaling_coef"] = 1
        self.attributes["activation"] = (
            "Rectifier" if get_node_from_metaop(node, "ReLU") else "Linear"
        )

        self.set_scaling_attributes()
        self.attributes["scaling_coef"] = self.attributes.get("coef_value", 1)

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "activation_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "activation_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/activation.h"]
        self.kernels_to_copy = []
