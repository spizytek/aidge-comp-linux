"""Standalone ReLU C export node."""

import aidge_core

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


@ExportLibC.register(
    "ReLU",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
    aidge_core.ProdConso.in_place_model,
)
class ReLUC(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        output_tensor = self.operator.get_output(0)
        if output_tensor.dtype not in (
            aidge_core.dtype.float32,
            aidge_core.dtype.int8,
        ):
            raise NotImplementedError(
                "aidge_export_c supports only float32 or int8 ReLU layers."
            )
        is_int8 = output_tensor.dtype == aidge_core.dtype.int8
        self.attributes["is_int8"] = is_int8
        self.attributes["shift_value"] = 0
        self.attributes["scaling_coef"] = 1
        if is_int8 and self.get_quantizers():
            self.set_scaling_attributes()
            self.attributes["scaling_coef"] = self.attributes.get("coef_value", 1)
        self.attributes["activation"] = "Rectifier"
        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "activation_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "activation_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/activation.h"]
        self.kernels_to_copy = []


@ExportLibC.register_metaop(
    "QReLU",
    "ReLU->Quantizer",
    aidge_core.ImplSpec(aidge_core.IOSpec(aidge_core.dtype.any)),
)
class QReLUC(ReLUC):
    pass
