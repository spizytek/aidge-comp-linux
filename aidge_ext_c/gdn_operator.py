"""Backward-compatible import for the older scripts.

Prefer ``from aidge_ext_c import ...`` in new code.  This module keeps the
``import gdn_operator`` pattern working (used by the early prototype scripts)
and auto-registers both the CPU implementation and, if available, the C
exporter.
"""

from __future__ import annotations

try:  # package import
    from .gdn import GDNImpl, install_gdn_cpu, make_gdn_node, register_gdn_cpu
    from .gdn_export_c import GdnC, register_gdn_export
except ImportError:  # flat sys.path import (sys.path.insert(0, "aidge_ext_c"))
    from gdn import GDNImpl, install_gdn_cpu, make_gdn_node, register_gdn_cpu
    from gdn_export_c import GdnC, register_gdn_export

register_gdn_cpu()
register_gdn_export()

__all__ = [
    "GDNImpl",
    "GdnC",
    "install_gdn_cpu",
    "make_gdn_node",
    "register_gdn_cpu",
    "register_gdn_export",
]
