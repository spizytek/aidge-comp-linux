"""2D transposed convolution C export nodes (float32)."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


_CONV_TRANSPOSE_NHWC_SPEC = aidge_core.ImplSpec(
    [
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.nhwc),
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.nhwc),
        aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any),
    ],
    [aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.nhwc)],
)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class _ConvTransposeBase(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        nodes = get_node_from_metaop(node, "ConvTranspose2D")
        op = nodes[0].get_operator() if nodes else self.operator
        input_tensor = self.operator.get_input(0)
        output_tensor = self.operator.get_output(0)

        if (
            input_tensor.dtype != aidge_core.dtype.float32
            or output_tensor.dtype != aidge_core.dtype.float32
        ):
            raise NotImplementedError(
                "aidge_export_c currently supports only float32 ConvTranspose2D."
            )

        in_channels = op.in_channels()
        out_channels = op.out_channels()
        kernel_dims = _pair(op.attr.kernel_dims)
        in_dims = self.attributes["in_dims"][0]
        out_dims = self.attributes["out_dims"][0]

        def spatial_dims(dims, channels):
            if dims[-1] == channels:
                return dims[1], dims[2]
            if len(dims) == 4 and dims[1] == channels:
                return dims[2], dims[3]
            raise NotImplementedError(
                f"Cannot infer spatial dimensions from {dims} and {channels} channels."
            )

        in_h, in_w = spatial_dims(in_dims, in_channels)
        out_h, out_w = spatial_dims(out_dims, out_channels)

        self.attributes["nb_channels"] = in_channels
        self.attributes["nb_outputs"] = out_channels
        self.attributes["in_batch"] = [in_dims[0]]
        self.attributes["in_height"] = [in_h]
        self.attributes["in_width"] = [in_w]
        self.attributes["out_batch"] = [out_dims[0]]
        self.attributes["out_height"] = [out_h]
        self.attributes["out_width"] = [out_w]
        self.attributes["padding"] = [0, 0, 0, 0]
        self.attributes["activation"] = "Linear"
        self.attributes["kernel_dims"] = kernel_dims
        self.attributes["stride_dims"] = _pair(op.attr.stride_dims)
        self.attributes["dilation_dims"] = _pair(op.attr.dilation_dims)
        self.attributes["weights_size"] = (
            in_channels
            * kernel_dims[0]
            * (kernel_dims[1] if len(kernel_dims) > 1 else 1)
            * out_channels
        )
        self.attributes["biases_size"] = out_channels

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "convolution_transpose_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "convolution_transpose_forward.jinja"
        )
        self.include_list = [
            "utils/c/typedefs.h",
            "kernels/c/convolution_transpose.h",
        ]
        self.kernels_to_copy = []


@ExportLibC.register("ConvTranspose2D", _CONV_TRANSPOSE_NHWC_SPEC)
class ConvTranspose2DC(_ConvTransposeBase):
    pass


@ExportLibC.register_metaop(
    "ConvTranspose2DAct",
    "ConvTranspose2D->ReLU",
    _CONV_TRANSPOSE_NHWC_SPEC,
)
class ConvTranspose2DActC(_ConvTransposeBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop(
    ["PadConvTranspose2D", "PaddedConvTranspose2D"],
    "ConvTranspose2D->Pad",
    _CONV_TRANSPOSE_NHWC_SPEC,
)
class PadConvTranspose2DC(_ConvTransposeBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pad_nodes = get_node_from_metaop(node, "Pad")
        if pad_nodes:
            self.attributes["padding"] = list(pad_nodes[0].get_operator().attr.pads)


@ExportLibC.register_metaop(
    "PadConvTranspose2DAct",
    "ConvTranspose2D->Pad->ReLU",
    _CONV_TRANSPOSE_NHWC_SPEC,
)
class PadConvTranspose2DActC(PadConvTranspose2DC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"
