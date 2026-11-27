"""Fully-connected and FC+activation C export nodes."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


_FC_DEFAULT_SPEC = aidge_core.ImplSpec(
    [
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.default),
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.any),
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any),
    ],
    [aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.any)],
)

_FC_NHWC_SPEC = aidge_core.ImplSpec(
    [
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.nhwc),
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.any),
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any),
    ],
    [aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.any)],
)


class _FCBase(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        input_tensor = self.operator.get_input(0)
        output_tensor = self.operator.get_output(0)
        fc_nodes = get_node_from_metaop(node, "FC")
        fc_operator = fc_nodes[0].get_operator() if fc_nodes else self.operator

        is_int8 = (
            input_tensor.dtype == aidge_core.dtype.int8
            and output_tensor.dtype == aidge_core.dtype.int8
        )
        is_float = (
            input_tensor.dtype == aidge_core.dtype.float32
            and output_tensor.dtype == aidge_core.dtype.float32
        )
        if not (is_int8 or is_float):
            raise NotImplementedError(
                "aidge_export_c supports only float32 or int8 FC/FCAct layers."
            )

        self.attributes["is_int8"] = is_int8
        self.attributes["shift_value"] = 0
        self.attributes["rescaling"] = "NoScaling"
        if is_int8:
            if self.get_quantizers():
                self.set_scaling_attributes()
                if self.attributes.get("shift_value", 0) != 0:
                    self.attributes["rescaling"] = "SingleShiftScaling"
                self.attributes["bias_default_type"] = "int32_t"
        elif get_node_from_metaop(node, "Quantizer"):
            raise NotImplementedError(
                "Quantized FCAct/QFC layers are not supported by aidge_export_c yet."
            )
        # QFC/Conv use SingleShiftScaling in Aidge even when a coef is present.
        self.attributes["scaling_coef"] = 1

        self.attributes["nb_channels"] = self.attributes["in_chan"][0] * aidge_core.get_nb_packs(
            input_tensor.dtype
        )
        self.attributes["nb_outputs"] = fc_operator.out_channels() * aidge_core.get_nb_packs(
            output_tensor.dtype
        )
        self.attributes["activation"] = "Linear"
        self.attributes["weights_size"] = (
            self.attributes["out_chan"][0]
            * self.attributes["in_chan"][0]
            * self.attributes["in_height"][0]
            * self.attributes["in_width"][0]
        )
        self.attributes["biases_size"] = self.attributes["out_chan"][0]

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "fullyconnected_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "fullyconnected_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/fullyconnected.h"]
        self.kernels_to_copy = []


@ExportLibC.register("FC", _FC_NHWC_SPEC)
@ExportLibC.register("FC", _FC_DEFAULT_SPEC)
class FCC(_FCBase):
    pass


@ExportLibC.register_metaop("FCAct", "FC->Quantizer?->ReLU", _FC_NHWC_SPEC)
@ExportLibC.register_metaop("FCAct", "FC->Quantizer?->ReLU", _FC_DEFAULT_SPEC)
class FCActC(_FCBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("QFC", "FC->Quantizer", _FC_NHWC_SPEC)
@ExportLibC.register_metaop("QFC", "FC->Quantizer", _FC_DEFAULT_SPEC)
class QFCC(_FCBase):
    pass
