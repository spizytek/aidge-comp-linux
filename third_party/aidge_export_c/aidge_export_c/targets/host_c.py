"""Host C target and C project generation strategy."""

import shutil
from pathlib import Path

import aidge_core
from aidge_core.export_utils import (
    ExportLib,
    generate_file,
    register_target_export_strategy,
)
from aidge_core.hw_model.components import Memory
from aidge_core.hw_model.device import Device
from aidge_core.hw_model.execution import (
    BuildSystem,
    ExecutionDomain,
    ExecutionEnvironment,
)
from aidge_core.hw_model.processing_elements import CPU
from aidge_core.hw_model.registry import register_target
from aidge_core.mem_info import generate_optimized_memory_info

from .. import C_ROOT
from ..export_registry import ExportLibC


class CCPU(CPU):
    """CPU whose Aidge export backend is the C export library."""

    def __init__(self):
        super().__init__()
        self._lib = ["export_c"]


@register_target("host_c")
class HostC(Device):
    """Current host machine, exporting a standalone C project."""

    def __init__(self):
        super().__init__(name="host_c")
        self.cpu = CCPU()
        self.mem = Memory(capacity=1_000_000_000, priority=0)
        self.linux_domain = ExecutionDomain(
            name="linux",
            environment=ExecutionEnvironment.LINUX,
            processing_elements=[self.cpu],
            memories=[self.mem],
            build_system=BuildSystem.CMAKE,
            xnn_configurable=False,
            cmsis_configurable=False,
        )
        self.default_execution_domain = self.linux_domain


def _copy_static_files(export_folder: Path):
    for source, destination in ExportLibC.static_files.items():
        source_path = Path(source)
        destination_path = export_folder / destination
        destination_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path / source_path.name)


def _generate_data_file(
    export_folder: Path,
    array_name: str,
    tensor,
    header_dir: str,
    source_dir: str,
):
    header_rel = Path(header_dir) / f"{array_name}.h"
    source_rel = Path(source_dir) / f"{array_name}.c"
    (export_folder / header_rel).parent.mkdir(parents=True, exist_ok=True)
    (export_folder / source_rel).parent.mkdir(parents=True, exist_ok=True)

    import aidge_core

    tensor_host = aidge_core.Tensor()
    tensor_host.to_dtype(tensor.dtype)
    tensor_host.to_backend("cpu")
    tensor_host.copy_from(tensor)
    tensor_host.resize([tensor.size])

    generate_file(
        export_folder / header_rel,
        C_ROOT / "templates" / "data" / "producer_header.h.jinja",
        name=array_name,
        dims=tensor.dims,
        dtype=aidge_core.export_utils.aidge2c(tensor.dtype),
        dtype_name=tensor.dtype,
    )
    generate_file(
        export_folder / source_rel,
        C_ROOT / "templates" / "data" / "producer_src.c.jinja",
        name=array_name,
        declaration_path=f"{header_dir}/{array_name}.h",
        dims=tensor.dims,
        dtype=aidge_core.export_utils.aidge2c(tensor.dtype),
        dtype_name=tensor.dtype,
        literals=tensor_host.to_cstring(),
    )
    return header_rel.as_posix()


@register_target_export_strategy("host_c")
def compose_c_project(ctx, application):
    """Generate a standalone C project for the prepared export context."""

    if application.kind != "standalone":
        raise ValueError(
            "aidge_export_c currently supports only standalone applications "
            f"(requested: {application.kind!r})."
        )

    export_folder = Path(ctx.export_folder)
    if export_folder.is_dir():
        shutil.rmtree(export_folder)
    export_folder.mkdir(parents=True, exist_ok=True)

    dnn_folder = export_folder / "dnn"
    (dnn_folder / "src").mkdir(parents=True, exist_ok=True)
    (dnn_folder / "include").mkdir(parents=True, exist_ok=True)

    memory_alignment = 16 if ctx.dtype == "int8" else 1
    mem_peaks, mem_info = generate_optimized_memory_info(
        ctx.scheduler,
        stats_folder=str(export_folder / "stats"),
        multiple_mems=ctx.multiple_mems,
        memory_components=ctx.device.memories,
        memory_alignment=memory_alignment,
        auto_concat=ctx.auto_concat,
    )
    if isinstance(mem_peaks, int):
        mem_peaks = [mem_peaks]

    _copy_static_files(export_folder)

    list_forward_nodes = ctx.scheduler.get_sequential_static_scheduling()
    node_names = [node.name() for node in list_forward_nodes]
    graph_is_cyclic = len(node_names) != len(set(node_names))
    instance_count = dict.fromkeys(list_forward_nodes, 0)

    list_configs = []
    list_actions = []
    export_nodes = {}

    for node in list_forward_nodes:
        op_impl = node.get_operator().get_impl()
        if not isinstance(op_impl, ExportLib):
            raise RuntimeError(
                f"Operator {node.name()}[{node.type()}] has no C export backend "
                f"(got {op_impl!r})."
            )

        required_specs = op_impl.get_required_spec()
        specs = op_impl.get_best_match(required_specs)
        export_node_cls = op_impl.get_export_node(specs)
        if export_node_cls is None:
            raise RuntimeError(
                f"No C export node for {node.name()}[{node.type()}] with specs "
                f"{required_specs}. Available: {op_impl.get_available_impl_specs()}"
            )

        op = export_node_cls(
            node,
            mem_info[node],
            instance_count[node] if graph_is_cyclic else None,
        )
        export_nodes[node] = op
        list_configs += op.export(dnn_folder)
        list_actions += op.forward()

        if graph_is_cyclic:
            instance_count[node] += 1

    graph_view = ctx.scheduler.graph_view()

    inputs_name = []
    inputs_dtype = []
    for in_node, in_idx in graph_view.get_ordered_inputs():
        if in_node.get_operator().get_input(in_idx) is not None:
            op = export_nodes[in_node]
            inputs_name.append(op.attributes["in_name"][in_idx])
            inputs_dtype.append(op.attributes["in_cdtype"][in_idx])

    outputs_name = []
    outputs_dtype = []
    output_sizes = []
    for out_node, out_idx in graph_view.get_ordered_outputs():
        op = export_nodes[out_node]
        outputs_name.append(op.attributes["out_name"][out_idx])
        outputs_dtype.append(op.attributes["out_cdtype"][out_idx])
        output_sizes.append(int(out_node.get_operator().get_output(out_idx).size))

    input_headers = []
    for index, input_tensor in enumerate(ctx.input_tensors):
        if index >= len(inputs_name):
            break
        input_headers.append(
            _generate_data_file(
                export_folder,
                inputs_name[index],
                input_tensor,
                "data",
                "data",
            )
        )

    generate_file(
        dnn_folder / "include" / "forward.h",
        C_ROOT / "templates" / "forward.h.jinja",
        include_guard_name="DNN_H",
        func_name="model_forward",
        inputs_name=inputs_name,
        inputs_dtype=inputs_dtype,
        outputs_name=outputs_name,
        outputs_dtype=outputs_dtype,
    )

    generate_file(
        dnn_folder / "src" / "forward.c",
        C_ROOT / "templates" / "forward.c.jinja",
        include_forward_header="forward.h",
        func_name="model_forward",
        headers=set(str(header) for header in list_configs),
        actions=list_actions,
        mem_peaks=mem_peaks,
        inputs_name=inputs_name,
        inputs_dtype=inputs_dtype,
        outputs_name=outputs_name,
        outputs_dtype=outputs_dtype,
    )

    generate_file(
        export_folder / "main.c",
        C_ROOT / "templates" / "main.c.jinja",
        include_forward_header="forward.h",
        func_name="model_forward",
        inputs_name=inputs_name,
        inputs_dtype=inputs_dtype,
        outputs_name=outputs_name,
        outputs_dtype=outputs_dtype,
        output_sizes=output_sizes,
        input_headers=input_headers,
    )

    shutil.copy2(C_ROOT / "static" / "CMakeLists.txt", export_folder / "CMakeLists.txt")

    return {
        "memory_peak_bytes": mem_peaks,
        "expected_artifact": "build/run_export",
        "protocol": "aidge-stdout/v1",
    }
