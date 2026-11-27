"""Elementwise C export nodes (Add/Sub/Mul/Div and fused activations)."""

import aidge_core
from aidge_core.export_utils import get_node_from_metaop

from .. import C_ROOT
from ..export_registry import ExportLibC
from ..node_export import ExportNodeC


_ELEMWISE_SPEC = aidge_core.ImplSpec(
    aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any_1)
)


class _ElemWiseBase(ExportNodeC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)

        input0 = self.operator.get_input(0)
        input1 = self.operator.get_input(1)
        output = self.operator.get_output(0)
        is_int8 = (
            input0.dtype == aidge_core.dtype.int8
            and input1.dtype == aidge_core.dtype.int8
            and output.dtype == aidge_core.dtype.int8
        )
        is_float = (
            input0.dtype == aidge_core.dtype.float32
            and input1.dtype == aidge_core.dtype.float32
            and output.dtype == aidge_core.dtype.float32
        )
        if not (is_int8 or is_float):
            raise NotImplementedError(
                "aidge_export_c supports only float32 or int8 elementwise ops."
            )
        self.attributes["is_int8"] = is_int8
        self.attributes["shift_value"] = 0
        self.attributes["scaling_coef"] = 1
        if is_int8:
            if self.get_quantizers():
                self.set_scaling_attributes()
                self.attributes["scaling_coef"] = self.attributes.get("coef_value", 1)
        elif get_node_from_metaop(node, "Quantizer"):
            raise NotImplementedError(
                "Quantized elementwise ops are not supported by aidge_export_c yet."
            )

        self.attributes["activation"] = "Linear"

        nbdims_out = len(self.attributes["out_dims"][0])
        dims_a = self.attributes["in_dims"][0]
        dims_b = self.attributes["in_dims"][1]
        ndim_a = [1] * nbdims_out
        ndim_b = [1] * nbdims_out

        idx_a = nbdims_out - len(dims_a)
        for i in range(nbdims_out):
            ndim_a[i] = 1 if i < idx_a else dims_a[i - idx_a]
        idx_b = nbdims_out - len(dims_b)
        for i in range(nbdims_out):
            ndim_b[i] = 1 if i < idx_b else dims_b[i - idx_b]

        contiguous_idx = nbdims_out - 1
        for i in range(nbdims_out - 1, -1, -1):
            if ndim_a[i] != ndim_b[i]:
                break
            contiguous_idx = i

        input0_contiguous_size = 1
        input1_contiguous_size = 1
        output_contiguous_size = 1
        for i in range(contiguous_idx, nbdims_out):
            input0_contiguous_size *= ndim_a[i]
            input1_contiguous_size *= ndim_b[i]
            output_contiguous_size *= self.attributes["out_dims"][0][i]

        self.attributes["input1_cont_size"] = input0_contiguous_size
        self.attributes["input2_cont_size"] = input1_contiguous_size
        self.attributes["output_cont_size"] = output_contiguous_size

        stride_post0 = [0] * contiguous_idx
        stride_post1 = [0] * contiguous_idx
        stride_step0 = [0] * contiguous_idx
        stride_step1 = [0] * contiguous_idx

        if contiguous_idx > 0:
            stride_post0[contiguous_idx - 1] = 1
            stride_post1[contiguous_idx - 1] = 1
            for i in range(contiguous_idx - 2, -1, -1):
                stride_post0[i] = stride_post0[i + 1] * ndim_a[i + 1]
                stride_post1[i] = stride_post1[i + 1] * ndim_b[i + 1]
            for i in range(contiguous_idx):
                stride_step0[i] = 1 - stride_post0[i] if ndim_a[i] == 1 else 1
                stride_step1[i] = 1 - stride_post1[i] if ndim_b[i] == 1 else 1

        offset_in0 = 0
        offset_in1 = 0
        nb_matrices = 1
        for i in range(contiguous_idx):
            nb_matrices *= self.attributes["out_dims"][0][i]

        self.attributes["offset_in1"] = [0]
        self.attributes["offset_in2"] = [0]
        for stack in range(1, nb_matrices):
            dim = contiguous_idx - 1
            tmp_stack = stack
            while tmp_stack % self.attributes["out_dims"][0][dim] == 0:
                tmp_stack //= self.attributes["out_dims"][0][dim]
                dim -= 1
            offset_in0 += stride_step0[dim]
            offset_in1 += stride_step1[dim]
            self.attributes["offset_in1"].append(offset_in0)
            self.attributes["offset_in2"].append(offset_in1)

        self.attributes["elemwise_op"] = "Add"
        self.config_template = str(
            C_ROOT / "templates" / "configuration" / "elemwise_config.jinja"
        )
        self.forward_template = str(
            C_ROOT / "templates" / "kernel_forward" / "elemwise_forward.jinja"
        )
        self.include_list = ["utils/c/typedefs.h", "kernels/c/elemwise.h"]
        self.kernels_to_copy = []


@ExportLibC.register("Add", _ELEMWISE_SPEC, aidge_core.ProdConso.in_place_model)
class AddC(_ElemWiseBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.attributes["elemwise_op"] = "Add"


@ExportLibC.register("Sub", _ELEMWISE_SPEC, aidge_core.ProdConso.in_place_model)
class SubC(_ElemWiseBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.attributes["elemwise_op"] = "Sub"


@ExportLibC.register("Mul", _ELEMWISE_SPEC, aidge_core.ProdConso.in_place_model)
class MulC(_ElemWiseBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.attributes["elemwise_op"] = "Mul"


@ExportLibC.register("Div", _ELEMWISE_SPEC, aidge_core.ProdConso.in_place_model)
class DivC(_ElemWiseBase):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        self.attributes["elemwise_op"] = "Div"


@ExportLibC.register_metaop("AddAct", "Add->Quantizer?->ReLU", _ELEMWISE_SPEC)
class AddActC(AddC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("SubAct", "Sub->Quantizer?->ReLU", _ELEMWISE_SPEC)
class SubActC(SubC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("MulAct", "Mul->Quantizer?->ReLU", _ELEMWISE_SPEC)
class MulActC(MulC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("DivAct", "Div->Quantizer?->ReLU", _ELEMWISE_SPEC)
class DivActC(DivC):
    def __init__(self, node, mem_info, instance_count):
        super().__init__(node, mem_info, instance_count)
        if get_node_from_metaop(node, "ReLU"):
            self.attributes["activation"] = "Rectifier"


@ExportLibC.register_metaop("QAdd", "Add->Quantizer", _ELEMWISE_SPEC)
class QAddC(AddC):
    pass


@ExportLibC.register_metaop("QSub", "Sub->Quantizer", _ELEMWISE_SPEC)
class QSubC(SubC):
    pass


@ExportLibC.register_metaop("QMul", "Mul->Quantizer", _ELEMWISE_SPEC)
class QMulC(MulC):
    pass


@ExportLibC.register_metaop("QDiv", "Div->Quantizer", _ELEMWISE_SPEC)
class QDivC(DivC):
    pass
