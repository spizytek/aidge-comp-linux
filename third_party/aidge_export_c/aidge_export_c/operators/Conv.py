"""2D convolution (and Conv+ReLU) C export nodes."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


_CONV_NHWC_SPEC = aidge_core.ImplSpec(
    [
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.nhwc),
        aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.nhwc),
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any),
    ],
    [aidge_core.IOSpec(aidge_core.dtype.any_or_packed_last, aidge_core.dformat.nhwc)],
)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class _ConvBase(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        conv_nodes = get_node_from_metaop(node, "Conv2D")
        conv_operator = conv_nodes[0].get_operator() if conv_nodes else self.operator
        input_tensor = self.operator.get_input(0)
        output_tensor = self.operator.get_output(0)

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
                "aidge_export_c supports only float32 or int8 Conv2D layers."
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
                "Quantized Conv2D layers are not supported by aidge_export_c yet."
            )
        self.attributes["scaling_coef"] = 1

        self.attributes["nb_channels"] = conv_operator.in_channels() * aidge_core.get_nb_packs(
            input_tensor.dtype
        )
        self.attributes["nb_outputs"] = conv_operator.out_channels() * aidge_core.get_nb_packs(
            output_tensor.dtype
        )
        self.attributes["padding"] = [0, 0, 0, 0]
        self.attributes["activation"] = "Linear"
        self.attributes["kernel_dims"] = _pair(conv_operator.attr.kernel_dims)
        self.attributes["stride_dims"] = _pair(conv_operator.attr.stride_dims)
        self.attributes["dilation_dims"] = _pair(conv_operator.attr.dilation_dims)
        self.attributes["weights_size"] = (
            self.attributes["out_chan"][0]
            * self.attributes["in_chan"][0]
            * self.attributes["kernel_dims"][0]
            * (self.attributes["kernel_dims"][1] if len(self.attributes["kernel_dims"]) > 1 else 1)
        )
        self.attributes["biases_size"] = self.attributes["out_chan"][0]

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "convolution_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "convolution_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/convolution.h"]
        self.kernels_to_copy = []


@ExportLibC.register("Conv2D", _CONV_NHWC_SPEC)
class Conv2DC(_ConvBase):
    pass


@ExportLibC.register_metaop("ConvAct", "Conv2D->Quantizer?->ReLU", _CONV_NHWC_SPEC)
class ConvActC(_ConvBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop(
    ["PadConv", "PaddedConv2D"],
    "Pad->Conv2D->Quantizer?",
    _CONV_NHWC_SPEC,
)
class PadConvC(_ConvBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pad_nodes = get_node_from_metaop(node, "Pad")
        if pad_nodes:
            self.attributes["padding"] = list(pad_nodes[0].get_operator().attr.pads)


@ExportLibC.register_metaop(
    "PadConvAct",
    "Pad->Conv2D->Quantizer?->ReLU",
    _CONV_NHWC_SPEC,
)
class PadConvActC(PadConvC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("QConv", "Conv2D->Quantizer", _CONV_NHWC_SPEC)
class QConvC(_ConvBase):
    pass
