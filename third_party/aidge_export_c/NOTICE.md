# Vendored `aidge_export_c`

This directory contains the AIDGE standalone **C export backend**
(`aidge_export_c` 0.6.0) used by the GDN C exporter and the LoCod flow.

It depends on Eclipse AIDGE (`aidge_core`, `aidge_export_cpp` ecosystem,
EPL-2.0). Install it in editable mode from this directory:

```bash
pip install -e third_party/aidge_export_c
```

The package expects `aidge_core` to expose
`dtype.any_or_packed_last` / `dtype.any_or_packed_full`.
`aidge-core` **0.11.0** has them; the 0.10.1 PyPI wheel does not. Use 0.11.0
or build AIDGE `core` from source if you need 0.10 compatibility.
