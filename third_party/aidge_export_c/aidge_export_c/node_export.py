"""C export node base class."""

from aidge_core.export_utils import ExportNodeCpp


class ExportNodeC(ExportNodeCpp):
    """C-flavoured export node.

    The code-generation mechanics are shared with ``ExportNodeCpp`` (copy the
    kernels, render the configuration template, render the forward call); only
    the output paths and extensions differ.
    """

    def __init__(self, aidge_node, mem_info=None, instance_count=None):
        super().__init__(aidge_node, mem_info, instance_count)
        self.kernels_path = "include/kernels/c"
        self.config_path = "include/layers"
        self.config_extension = "h"
