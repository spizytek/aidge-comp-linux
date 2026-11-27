"""C export library registration."""

import aidge_core
from aidge_core.export_utils import ExportLib

from . import C_ROOT


class ExportLibC(ExportLib):
    """Aidge export implementation backed by C code generation."""

    _name = "export_c"
    _default_dformat = aidge_core.dformat.nhwc

    # Static files copied into the generated project.  Destinations are
    # relative to the export root.
    static_files = {
        str(C_ROOT / "static" / "kernel" / "typedefs.h"): "dnn/include/utils/c",
        str(C_ROOT / "static" / "kernel" / "activation.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "fullyconnected.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "convolution.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "convolution_depthwise.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "pooling.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "elemwise.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "transpose.h"): "dnn/include/kernels/c",
        str(C_ROOT / "static" / "kernel" / "convolution_transpose.h"): "dnn/include/kernels/c",
    }

    # Only used by the legacy adaptation path.  The non-legacy adapt_graph
    # uses the per-class meta-operator registry populated by register_metaop.
    _fusion_list = [
        "FCAct",
    ]
