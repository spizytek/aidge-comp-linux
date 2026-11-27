"""2D max/average pooling C export nodes."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


_POOL_NHWC_SPEC = aidge_core.ImplSpec(
    aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.nhwc)
)


def _pair(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class _PoolBase(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        output_tensor = self.operator.get_output(0)
        if output_tensor.dtype not in (
            aidge_core.dtype.float32,
            aidge_core.dtype.int8,
        ):
            raise NotImplementedError(
                "aidge_export_c supports only float32 or int8 pooling layers."
            )
        self.attributes["is_int8"] = output_tensor.dtype == aidge_core.dtype.int8

        self.attributes["stride_dims"] = [1, 1]
        self.attributes["padding"] = [0, 0, 0, 0]
        self.attributes["pool_type"] = "Max"
        self.attributes["activation"] = "Linear"

        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "pooling_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "pooling_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/pooling.h"]
        self.kernels_to_copy = []


@ExportLibC.register("MaxPooling2D", _POOL_NHWC_SPEC)
class MaxPool2DC(_PoolBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pool_nodes = get_node_from_metaop(node, "MaxPooling2D")
        pool_operator = pool_nodes[0].get_operator() if pool_nodes else self.operator
        self.attributes["pool_type"] = "Max"
        self.attributes["kernel_dims"] = _pair(pool_operator.attr.kernel_dims)
        self.attributes["stride_dims"] = _pair(pool_operator.attr.stride_dims)


@ExportLibC.register("AvgPooling2D", _POOL_NHWC_SPEC)
class AvgPool2DC(_PoolBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pool_nodes = get_node_from_metaop(node, "AvgPooling2D")
        pool_operator = pool_nodes[0].get_operator() if pool_nodes else self.operator
        self.attributes["pool_type"] = "Average"
        self.attributes["kernel_dims"] = _pair(pool_operator.attr.kernel_dims)
        self.attributes["stride_dims"] = _pair(pool_operator.attr.stride_dims)


@ExportLibC.register_metaop(
    "MaxPoolAct",
    "MaxPooling2D->ReLU",
    _POOL_NHWC_SPEC,
)
class MaxPoolActC(MaxPool2DC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop(
    "AvgPoolAct",
    "AvgPooling2D->ReLU",
    _POOL_NHWC_SPEC,
)
class AvgPoolActC(AvgPool2DC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop(
    ["PadMaxPool", "PaddedMaxPooling2D"],
    "Pad->MaxPooling2D",
    _POOL_NHWC_SPEC,
)
class PadMaxPoolC(MaxPool2DC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pad_nodes = get_node_from_metaop(node, "Pad")
        if pad_nodes:
            self.attributes["padding"] = list(pad_nodes[0].get_operator().attr.pads)


@ExportLibC.register_metaop(
    "PadMaxPoolAct",
    "Pad->MaxPooling2D->ReLU",
    _POOL_NHWC_SPEC,
)
class PadMaxPoolActC(PadMaxPoolC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop(
    ["PadAvgPool", "PaddedAvgPooling2D"],
    "Pad->AvgPooling2D",
    _POOL_NHWC_SPEC,
)
class PadAvgPoolC(AvgPool2DC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        pad_nodes = get_node_from_metaop(node, "Pad")
        if pad_nodes:
            self.attributes["padding"] = list(pad_nodes[0].get_operator().attr.pads)
