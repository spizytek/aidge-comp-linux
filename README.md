# AIDGE in Space — Linux / PYNQ side

This repository is the Linux and PYNQ-Z2 working copy of the **AIDGE in Space**
project: a neural image compressor based on the **J. Ballé** model, built with
**Eclipse AIDGE** (unit operators) and **LoCod** (HLS), targeting a
space-qualifiable **Zynq-7020 / PYNQ-Z2** (PS + PL).

The immediate goal is the **encoder `g_a`** on the rover side: input image →
latent → entropy-coded bitstream. The decoder `g_s` runs on the platform side
and is developed in a second step.

---

## Status

| Area | State |
|---|---|
| Encoder ONNX (`Model/bmshj2018_factorized_encoder.onnx`) | q8, 64×64 tiles, encoder-only |
| Dataset (`DataSet/`) | 9,632 train / 1,248 test, 64×64 RGB space imagery |
| AIDGE **GDN unit operator** | ✅ import/use works (`aidge_ext_c`) |
| GDN CPU implementation | ✅ matches NumPy exactly; full encoder matches PyTorch to ~8e-5 |
| Model profiling / int8 QDQ / RD tools | ✅ see `docs/model-optimization.md` |
| int8 QDQ budget | ✅ −0.474 dB / +0.084 bpp on the full test set |
| AIDGE **PTQ + int8 C export** | ⚠️ needs the one-line `GDN` quantization patch + AIDGE rebuild |
| AIDGE **float C export** | ⚠️ generates/compiles; numerical C run still under validation (upstream NHWC conv-config bug) |
| LoCod / PYNQ-Z2 flow | ✅ friend's `locod.sh -t pynqz2` flow works for demo and small kernels |
| Full Balle encoder on PL | ⏳ not yet — see `docs/pynq-deployment.md` |

---

## Quick start — use GDN in AIDGE

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gdn.txt

# optional, for C export
pip install -e third_party/aidge_export_c

python examples/use_gdn.py
```

Expected output:

```text
output (1, 8, 4, 4) range ...
max |aidge - numpy| = 0.0
GDN OK
```

`examples/use_gdn.py` builds a tiny ONNX with a real `GDN` node, imports it
through AIDGE, installs the CPU implementation, runs inference and compares
with NumPy.

For the full 4-conv + 3-GDN encoder:

```bash
# needs torch + compressai; use requirements-opt.txt
python tools/build_gdn_unit_onnx.py
# -> analysis/encoder_gdn_unit_64.onnx
```

Then:

```python
import aidge_core, aidge_backend_cpu, aidge_onnx
from aidge_ext_c import install_gdn_cpu

graph = aidge_onnx.load_onnx("analysis/encoder_gdn_unit_64.onnx")
install_gdn_cpu(graph)
graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, 3, 64, 64]])
```

See **`aidge_ext_c/README.md`** for the API, C export and the PTQ patch.

---

## Model optimization

The full log, measurements and failed experiments are in:

**`docs/model-optimization.md`**

Key tools:

| Script | Purpose |
|---|---|
| `tools/profile_encoder.py` | params / MACs / peak workspace of an ONNX encoder |
| `tools/eval_rd_64.py` | bpp / PSNR / SSIM / speed on `DataSet/` |
| `tools/simulate_int8.py` | int8 QDQ simulation (weights + activations) |
| `tools/eval_approx_gdn.py` | hardware-friendly GDN vs exact |
| `tools/build_gdn_unit_onnx.py` | CompressAI encoder → 4-conv + 3-GDN ONNX |
| `training/` | model surgery + Modal T4 fine-tuning harness |

Baseline on the full 1,248-tile test set:

```text
float q8 : 2.1046 bpp · 37.813 dB · 0.9481 SSIM
int8     : 2.1882 bpp · 37.339 dB · 0.9436 SSIM   (−0.474 dB)
```

Install the optimization dependencies with `requirements-opt.txt` (torch,
compressai, onnxruntime, pytorch-msssim, optional Modal).

---

## PYNQ-Z2 deployment

The board plan — PS/PL partition, DDR weight streaming, tiling, int8, GDN on
PL, LoCod/DMA details, phases and metrics — is in:

**`docs/pynq-deployment.md`**

Short version:

```text
PS: image load, int8 quant, ANS/rANS entropy, control
PL: conv0 → GDN1 → conv2 → GDN3 → conv4 → GDN5 → conv6
AXI crossing: ~12 KB image in + ~5 KB latent out
DDR: 3.39 MB int8 weights, streamed
BRAM: 196 KB peak activation tile, double-buffered
```

---

## Original LoCod/PYNQ material

- `AidgeSetuponLInux.txt` — full setup notes (AIDGE, Vivado, LoCod, PYNQ).
- `FIX_REPORT_pynqz2.txt` — Pynq-Z2 target reconstruction and fixes.
- `locod_test_encoder_model/` — the monolithic encoder C (golden reference;
  **`params.h` is deliberately git-ignored**, ~14 MB).
- `test_locod.c` — LoCod Pascal-matrix bring-up test.

The full `locod.sh -t pynqz2` flow is documented in `FIX_REPORT_pynqz2.txt`;
the generated bitstreams and `locod/` itself are git-ignored.

---

## Repository map

```text
DataSet/                 64×64 space imagery (train/test)
Model/                   q8 encoder ONNX
locod_test_encoder_model/ monolithic encoder C (reference; params.h ignored)
aidge_ext_c/             proper AIDGE GDN/IGDN unit operator + C kernel
third_party/aidge_export_c/  vendored AIDGE standalone C export backend
examples/                GDN import-and-run example
tools/                   profiling, RD, int8 simulation, GDN ONNX builder
training/                model surgery + Modal fine-tuning
scripts/                 Linux setup/check scripts
docs/                    model optimization log + PYNQ deployment plan
```

## Known issues

1. **AIDGE PTQ and `GDN`** — the quantizer strategy table is hardcoded. Apply
   `aidge_ext_c/patches/0001-aidge-quantization-gdn.md` and rebuild, or wait
   for the upstream `GDN` registration. Until then PTQ stops with
   `makeStrategy(): no strategy registered for node type '... [GDN]'`.
2. **`aidge_export_c` NHWC conv config** — convs after a custom op can get
   `IN_HEIGHT`/`OUT_HEIGHT` set to the channel count. Use NCHW or correct the
   generated `layers/_*_PadConv_*.h` H/W defines.
3. **int8 graph boundaries** — `Quantizer`/`Dequantizer` meta-ops around a
   float GDN are not fully stable in `aidge_export_c` 0.6.0.
4. **`params.h`** — not in git; regenerate from `test_acetone.py` or the
   release archive.
5. **`locod/`** — not in git; clone `viveris/LoCod` and check out the pinned
   `locod-fpga` commit per `FIX_REPORT_pynqz2.txt`.

---

## Notes for Windows ↔ Linux

The `analysis/` directory contains generated/large files; only small JSON
reports are tracked. Model/optimization paths in `docs/model-optimization.md`
were recorded on Windows; use repo-relative paths on Linux.

The AIDGE GDN operator itself is pure Python + plain C and has no OS-specific
paths.
