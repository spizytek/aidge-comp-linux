"""AIDGE GDN/IGDN unit operator.

Runtime (no aidge_export_c needed):

    from aidge_ext_c import register_gdn_cpu, make_gdn_node, install_gdn_cpu

C export:

    from aidge_ext_c import register_gdn_export
    register_gdn_export()

Everything at once:

    from aidge_ext_c import register_gdn
    register_gdn(export_c=True)
"""

from .gdn import (
    GDNImpl,
    install_gdn_cpu,
    make_gdn_node,
    register_gdn_cpu,
)


def register_gdn(export_c: bool = False) -> None:
    """Register the CPU implementation, and optionally the C exporter."""
    register_gdn_cpu()
    if export_c:
        from .gdn_export_c import register_gdn_export
        register_gdn_export()


__all__ = [
    "GDNImpl",
    "install_gdn_cpu",
    "make_gdn_node",
    "register_gdn",
    "register_gdn_cpu",
]
