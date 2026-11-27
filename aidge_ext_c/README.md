# AIDGE GDN / IGDN unit operator

A single-node **GDN** operator for Eclipse AIDGE, with:

- graph type `GDN` (not a Sigmoid/HardSigmoid stand-in),
- attributes `channels`, `inverse`, `gamma[C,C]`, `beta[C]`,
- a NumPy CPU implementation (inference / PTQ calibration),
- an `aidge_export_c` backend that emits a plain C kernel.

AIDGE's generic ONNX importer already turns an unknown `op_type="GDN"` node
into a `GenericOperator("GDN")` and copies the attributes, so no custom ONNX
converter is needed.

---

## 1. Quick start — import an ONNX containing GDN

```python
import aidge_core
import aidge_backend_cpu
import aidge_onnx

from aidge_ext_c import install_gdn_cpu

graph = aidge_onnx.load_onnx("encoder_gdn_unit_64.onnx")  # contains GDN nodes
install_gdn_cpu(graph)          # attaches forward_dims + CPU impl to every GDN

graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, 3, 64, 64]])
scheduler = aidge_core.SequentialScheduler(graph)
...
```

Run the end-to-end smoke test:

```bash
python examples/use_gdn.py
# output (1, 8, 4, 4) ...
# max |aidge - numpy| = 0.0
# GDN OK
```

`examples/use_gdn.py` builds a tiny ONNX with a `GDN` node, imports it through
AIDGE, runs inference and compares with NumPy.  It needs only
`aidge-core`, `aidge-backend-cpu`, `aidge-onnx`, `onnx`.

---

## 2. Quick start — build the node programmatically

```python
from aidge_ext_c import make_gdn_node, install_gdn_cpu

gdn = make_gdn_node(gamma, beta, inverse=False, name="gdn")  # gamma [C,C], beta [C]
inp.add_child(gdn, 0, 0)
graph = aidge_core.get_connected_graph_view(gdn)
install_gdn_cpu(graph)       # important before compile()
graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, C, H, W]])
```

`gamma` and `beta` must be the **effective, non-negative** parameters
(`gamma_reparam`/`beta_reparam` from CompressAI), not the raw reparameterised
tensors.

---

## 3. Getting a GDN ONNX from the CompressAI encoder

CompressAI decomposes GDN into `Pow → Reshape → Conv1x1 → Sub → Sqrt → Div →
Mul`.  AIDGE cannot turn that into a single GDN node automatically.  Use:

```bash
# requires torch + compressai (Linux or the Windows .venv-ml)
python tools/build_gdn_unit_onnx.py
# -> analysis/encoder_gdn_unit_64.onnx  (4 convs + 3 GDN nodes)
```

The script loads `bmshj2018_factorized(quality=8)`, extracts the effective
gamma/beta, and writes `input → conv0 → GDN → conv2 → GDN → conv4 → GDN →
conv6 → x_out`.

For a different model, either adapt that script or provide a graph rewrite
that replaces each decomposed GDN subgraph with `make_gdn_node(gamma, beta)`.

---

## 4. C export

Install `aidge_export_c` first (the C backend used by the LoCod flow), then:

```python
from aidge_ext_c import register_gdn_export

register_gdn_export()

from aidge_core.export_utils import export
from aidge_core.export_utils.config import ExportConfig

export(
    graph,
    ExportConfig(target="host_c", export_folder="out_c",
                 dtype="float32", dformat="nchw", compile=False),
    input_tensors=calibration_tensors,
)
```

The exporter emits `kernels/c/gdn.h` and `gdn_{config,forward}.jinja` output.
It uses the **in-place** `ProdConso` model; the kernel computes all channel
norms before writing so in-place is safe.

### int8

AIDGE's PTQ does not know `GDN` yet.  `GDN` must be added to
`aidge_quantization`'s hardcoded `notQuantizedNodeTypes` set, then AIDGE must
be rebuilt from source.  See:

```text
aidge_ext_c/patches/0001-aidge-quantization-gdn.md
```

The patch is one line:

```diff
-                                                          "HardSigmoid"});
+                                                          "HardSigmoid",
+                                                          "GDN"});
```

Until that rebuild, `quantize_network` stops with:

```text
makeStrategy(): no strategy registered for node type 'gdn_1[GDN]'
```

`not_quantize_node(gdn)` is not sufficient: `makeStrategy()` runs before the
node policy is consulted.

---

## 5. Requirements

Runtime only:

```text
aidge-core
aidge-backend-cpu
aidge-onnx
onnx
```

Optional:

```text
aidge_export_c        # C export
aidge-quantization    # int8, after the GDN patch
torch, compressai     # only for tools/build_gdn_unit_onnx.py
```

`requirements-gdn.txt` lists these.

---

## 6. Known AIDGE issues encountered

1. **PTQ strategy table is hardcoded** (above).  Proper fix = one-line source
   patch + rebuild, or upstream the `GDN` registration.
2. **NHWC conv config bug in `aidge_export_c`**: for convs after a custom op,
   `IN_HEIGHT`/`OUT_HEIGHT` can receive the channel count (e.g. `192` instead
   of `32`).  Use NCHW when possible, or correct the generated
   `layers/_*_PadConv_*.h` H/W defines before compiling.
3. **int8 graph boundaries**: after PTQ, a float GDN between int8 convs
   produces `Quantizer`/`Dequantizer` meta-ops; `aidge_export_c` 0.6.0 expands
   them into `Cast`/`Round`/`Select` nodes and can fail.  Until that is fixed
   upstream, either:
   - keep the whole model float32, or
   - quantize only the convs and stitch the float GDN kernel by hand
     (the GDN C kernel is already self-contained).
4. **Layout**: the CPU implementation assumes NCHW.  The C kernel in `gdn.h`
   is NCHW.  If you export NHWC, set the GDN output format explicitly and
   use the NHWC kernel variant (the proto in this repo originally used NHWC;
   keep one convention across the graph).

---

## 7. Files

| Path | Purpose |
|---|---|
| `aidge_ext_c/gdn.py` | runtime: `make_gdn_node`, `register_gdn_cpu`, `install_gdn_cpu`, CPU impl |
| `aidge_ext_c/gdn_export_c.py` | optional `aidge_export_c` backend, `register_gdn_export()` |
| `aidge_ext_c/gdn_operator.py` | backward-compatible import (auto-registers CPU + export) |
| `aidge_ext_c/gdn.h` | fast-rsqrt GDN C kernel (NCHW, in-place safe) |
| `aidge_ext_c/templates/gdn_{config,forward}.jinja` | C layer/config templates |
| `aidge_ext_c/patches/0001-aidge-quantization-gdn.md` | PTQ patch + rebuild notes |
| `examples/use_gdn.py` | import-and-run smoke test |
| `tools/build_gdn_unit_onnx.py` | CompressAI encoder → GDN-unit ONNX |

---

## 8. Verification

```bash
python examples/use_gdn.py
```

Expected:

```text
output (1, 8, 4, 4) range ...
max |aidge - numpy| = 0.0
GDN OK
```

For the full encoder, the AIDGE float check against PyTorch is ~8e-5 max
difference (the remaining error is float32 accumulation order).
