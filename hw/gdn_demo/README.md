# GDN on PYNQ-Z2 — bring-up demo + setup guide

This is a **small, self-contained GDN accelerator** for the PYNQ-Z2 / LoCod
flow. It validates the **GDN unit operator** on the PL before the full
4-conv + 3-GDN encoder.

Shape: **C=16, H=W=8, NCHW, batch=1**. `1/sqrt` is a 5-step Newton iteration
(only add/mul/sub + comparisons; no `sqrtf`, no division).

> This README lives on the `gdn-demo` branch / PR #1:
> `https://github.com/spizytek/aidge-comp-linux/pull/1`
>
> Clone the branch if the PR is not merged yet:
> `git clone https://github.com/herambpatilofficial/aidge-comp-linux.git`
> then `git switch gdn-demo`.

---

## 0. End-to-end checklist

```text
[ ] Python venv + AIDGE runtime installed
[ ] aidge_export_c installed (vendored)
[ ] examples/use_gdn.py prints GDN OK
[ ] (optional) encoder_gdn_unit_64.onnx generated
[ ] (optional) AIDGE GDN quantization patch applied + AIDGE rebuilt
[ ] (optional) PTQ / QDQ check runs
[ ] make + ./bin/run_gdn prints RESULT: PASS
[ ] LoCod software build succeeds
[ ] full locod.sh -t pynqz2 produces fpga.bit/fpga.bin/locod-cpu
[ ] board run prints RESULT: PASS
```

---

## 1. What this demo tests

```text
input x[16,8,8]
  → for each output channel o and pixel p:
       norm = beta[o] + sum_j gamma[o,j] * x[j,p]^2
       y[o,p] = x[o,p] / sqrt(norm)       # Newton rsqrt
  → output y[16,8,8]
```

- `gamma[16,16]` and `beta[16]` are compile-time constants in `params.h`.
- CPU reference and FPGA call use the **same** `gdn.c`, so the comparison
  isolates the HLS/FPGA implementation.
- This is a **correctness/resource bring-up test**, not the full encoder.
  The full encoder needs int8, tiling and DDR weight streaming.

---

## 2. Files

| File | Purpose |
|---|---|
| `main.c` | LoCod CPU driver, test vector, CPU reference, PASS/FAIL |
| `gdn.c` | accelerator (`void gdn(struct param_gdn*, struct result_gdn*)`) |
| `io.h` | fixed-size structs + GDN dimensions |
| `params.h` | `GDN_GAMMA[16][16]`, `GDN_BETA[16]` |
| `locod_host.h` | desktop shim for `FPGA`/`wait`/timer |
| `Makefile` | desktop build |

---

## 3. Get the code and create the environment

```bash
git clone https://github.com/herambpatilofficial/aidge-comp-linux.git
cd aidge-comp-linux
git switch gdn-demo

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gdn.txt
```

`requirements-gdn.txt` installs `aidge-core`, `aidge-backend-cpu`,
`aidge-onnx` and `onnx`.

Shortcut (does the same plus the import smoke test):

```bash
bash scripts/setup_gdn.sh
```

**Version note:** use **AIDGE 0.11.0** (or newer) if you want the C export
backend. `aidge_export_c` needs `dtype.any_or_packed_last/full`, which the
0.10.1 PyPI wheel does not expose.

---

## 4. Set up the AIDGE GDN operator

### 4.1 Runtime (inference / graph import)

```bash
python examples/use_gdn.py
```

Expected:

```text
output (1, 8, 4, 4) range ...
max |aidge - numpy| = 0.0
GDN OK
```

What the example does:

```python
import aidge_core, aidge_backend_cpu, aidge_onnx
from aidge_ext_c import install_gdn_cpu

graph = aidge_onnx.load_onnx("gdn_demo.onnx")   # op_type="GDN"
install_gdn_cpu(graph)                          # attach dims + CPU impl
graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, C, H, W]])
```

AIDGE’s generic ONNX importer turns an unknown `op_type="GDN"` into a
`GenericOperator("GDN")` and copies `channels`, `inverse`, `gamma`, `beta`.

Programmatic use:

```python
from aidge_ext_c import make_gdn_node, install_gdn_cpu

gdn = make_gdn_node(gamma, beta)   # gamma [C,C], beta [C]
inp.add_child(gdn, 0, 0)
graph = aidge_core.get_connected_graph_view(gdn)
install_gdn_cpu(graph)
```

### 4.2 C export backend

The backend is vendored so the friend does not need a separate package:

```bash
pip install -e third_party/aidge_export_c
```

Then:

```python
from aidge_ext_c import register_gdn_export
register_gdn_export()
```

The exporter emits `kernels/c/gdn.h` and the GDN config/forward templates.
The kernel uses the in-place `ProdConso` model; it computes all channel norms
before writing.

### 4.3 Full encoder graph (optional)

To get the 4-conv + 3-GDN encoder ONNX from the CompressAI q8 model:

```bash
pip install -r requirements-opt.txt     # torch, compressai, onnxruntime, ...
python tools/build_gdn_unit_onnx.py
# -> analysis/encoder_gdn_unit_64.onnx
```

Then load it as in 4.1.

---

## 5. Set up quantization (AIDGE PTQ)

### 5.1 Why a patch is needed

AIDGE’s PTQ does not know `GDN`. The strategy table is hardcoded, so PTQ
stops with:

```text
makeStrategy(): no strategy registered for node type 'gdn_1[GDN]'
```

`not_quantize_node()` is not enough: `makeStrategy()` runs first. Add `GDN`
to the `notQuantizedNodeTypes` set (the same set as `Sigmoid`, `Tanh`,
`Softmax`, …), then rebuild AIDGE.

The policy is correct: **GDN stays float; the surrounding convs are int8.**

### 5.2 Apply the patch and rebuild (Ubuntu)

```bash
git clone --recursive https://gitlab.eclipse.org/eclipse/aidge/aidge.git
cd aidge

# edit:
# aidge_quantization/include/aidge/quantization/PTQ/PTQTypes.hpp
#
# static const std::set<std::string> notQuantizedNodeTypes({...,
#     "HardSigmoid",
#     "GDN"});

./setup.sh -m core -m backend_cpu -m onnx -m quantization --release --python
```

On Windows (VS 2022) the same change is built with:

```powershell
.\setup.ps1 -Modules core,backend_cpu,onnx,quantization -BuildType Release
```

Full patch notes: `aidge_ext_c/patches/0001-aidge-quantization-gdn.md`.

Verify:

```bash
python -c "import aidge_core, aidge_backend_cpu, aidge_quantization; print('ok')"
```

### 5.3 Quantize the encoder graph (AIDGE 0.11 API)

Build the calibration set from the PNG tiles (the `.npz` is ~128 MB and is
**not** committed):

```bash
python tools/pack_dataset.py
# -> analysis/dataset_64_uint8.npz  train (9632,3,64,64) / test (1248,3,64,64)
```

Then run PTQ (API from the official AIDGE docs):

```python
import numpy as np
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization
from aidge_ext_c import install_gdn_cpu

graph = aidge_onnx.load_onnx("analysis/encoder_gdn_unit_64.onnx")
install_gdn_cpu(graph)
graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, 3, 64, 64]])

raw = np.load("analysis/dataset_64_uint8.npz")["train"][:256]
calib = [
    aidge_core.Tensor(raw[i : i + 1].astype(np.float32) / 255.0)
    for i in range(raw.shape[0])
]

ptq = aidge_quantization.PTQ(
    graph,
    activation_precision=aidge_core.dtype.int8,
    accumulation_precision=aidge_core.dtype.int32,
    inference_precision=aidge_core.dtype.float32,
)
ptq.use_zero_point = True        # asymmetric activations
ptq.channel_wise = False         # per-channel crashes on this graph today
ptq.single_shift_approximation = False
qgraph = ptq.process(calib)
```

Per-node error report (optional, useful to see where the 0.5 dB budget goes):

```python
diag = ptq.get_diagnostic()
report = diag.compute_quantization_errors(
    ["SNR"], isolate_error=False, producer_error=False)
```

If a layout error appears while calibrating, set the tensor format explicitly
before appending it:

```python
t = aidge_core.Tensor(raw[i : i + 1].astype(np.float32) / 255.0)
t.set_data_format(aidge_core.dformat.nchw)
calib.append(t)
```

On AIDGE 0.10.x the legacy one-liner is:

```python
aidge_quantization.quantize_network(graph, 8, calib)   # legacy API
```

### 5.4 Check the int8 quality

The fastest check does not require AIDGE PTQ; it is the QDQ simulation:

```bash
python tools/simulate_int8.py --mode both --clipping max --calib 256
```

Expected (full 1,248-tile test set):

```text
float baseline : bpp=2.1046 psnr=37.813 ssim=0.9481
int8 both      : bpp=2.1882 psnr=37.339 ssim=0.9436
  dPSNR=-0.474 dB  dbpp=+0.0836
```

After AIDGE PTQ, compare the float and quantized latents on the same tiles
and decode both with the pretrained `g_s`; the target is the same ≈0.5 dB
budget. The quantization is “convs int8, GDN float”.

### 5.5 int8 C export (experimental)

After PTQ, you can try exporting the pre-quantized graph:

```python
from aidge_core.export_utils import export
from aidge_core.export_utils.config import ExportConfig

export(
    qgraph,
    ExportConfig(
        target="host_c",
        export_folder="out_int8_c",
        dtype="int8",
        dformat="nchw",
        model_prequantized=True,
        compile=False,
    ),
    input_tensors=calib,     # required by export()
)
```

Known AIDGE issues on this path (see `aidge_ext_c/README.md`):

- `aidge_export_c` can mis-set `IN_HEIGHT`/`OUT_HEIGHT` after a custom op
  (NHWC case); prefer NCHW, or correct the generated layer headers.
- A float GDN between int8 convs produces `Quantizer`/`Dequantizer` meta-ops
  that `aidge_export_c` 0.6.0 may expand into `Cast`/`Round`/`Select` and
  fail. Keep the whole model float, or stitch int8 convs + the float GDN
  kernel by hand until this is fixed upstream.

---

## 6. Build and test on the desktop

```bash
cd hw/gdn_demo
make
./bin/run_gdn
```

Expected:

```text
FPGA time = 0 ns
max |FPGA - CPU| = 0
out[0] FPGA=... CPU=...
RESULT: PASS
```

On the desktop the FPGA path is the shim, so the diff is exactly zero; this
checks the math and build.

---

## 7. Build with LoCod — software only

From the **LoCod root**:

```bash
cd /path/to/LoCod

bash ./locod.sh -t pynqz2 --no-hard \
  -f /path/to/aidge-comp-linux/hw/gdn_demo/main.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/gdn.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/io.h \
     /path/to/aidge-comp-linux/hw/gdn_demo/params.h
```

This produces the CPU executable without requiring Vivado/SDK.

---

## 8. Build with LoCod — full bitstream

From the **LoCod root** (Vivado + SDK installed):

```bash
bash ./locod.sh -t pynqz2 \
  -f /path/to/aidge-comp-linux/hw/gdn_demo/main.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/gdn.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/io.h \
     /path/to/aidge-comp-linux/hw/gdn_demo/params.h
```

Outputs in `locod-output/`:

```text
fpga.bit
fpga.bin
locod-cpu
```

### Important LoCod rules

- `main.c` **must be the first** `-f` argument.
- The startup call and the accelerator call must each stay on **one line**
  with literal arguments. Do not write their syntax in comments: `locod.sh`
  greps the raw text of `main.c`.
- Do not move the accelerator call into another file.
- Pass `params.h` in the `-f` list so Bambu can find it.

---

## 9. Run on the PYNQ-Z2

Copy the outputs:

```bash
scp locod-output/fpga.bit locod-output/fpga.bin locod-output/locod-cpu \
    xilinx@<board-ip>:~/mytest/
```

On the board:

```bash
ssh xilinx@<board-ip>
cd ~/mytest
chmod +x locod-cpu

# If fpgautil exists:
sudo fpgautil -b fpga.bin

# If fpgautil is absent (stock PYNQ image): use fpga_manager.
# The mainline driver wants a byte-swapped .bin; see FIX_REPORT_pynqz2.txt.
sudo ./locod-cpu
```

Expected:

```text
FPGA time = <N> ns
max |FPGA - CPU| = <small>
...
RESULT: PASS
```

`FPGA time` is the hardware cycle-counter value from the accelerator wrapper.
`max |FPGA - CPU|` should be below `1e-3`; float scheduling can reorder
additions, so it is not always bit-exact.

If the board rejects the bitstream with “could not find a sync word”, apply
the byte-swap in `FIX_REPORT_pynqz2.txt` and load `fpga_swapped.bin` through
`/sys/class/fpga_manager/fpga0/firmware`.

---

## 10. Quantizing the demo kernel (next step)

The current `gdn.c` is **float**, deliberately: it validates the GDN datapath.
The full model uses **int8 convs + float GDN** (or a fixed-point GDN later).

To take the demo to int8:

1. Calibrate the input range on the same synthetic vector / real layer input.
2. Quantize `gamma` per output channel and input per tensor; store int8
   weights/scales.
3. Compute `norm` in int32, dequantize, apply `rsqrt` in float/fixed-point,
   requantize the output to int8.
4. Keep the same `main.c` comparison; compare against the float/int8
   reference.
5. For the full model, use AIDGE PTQ (5.3) and `tools/simulate_int8.py` (5.4)
   as the numerical authority.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no accelerator generated` | `main.c` not first, call split across lines, or `FPGA(`/`init_locod(` text in comments |
| Bambu cannot find `params.h` | include it in the `-f` list |
| Bambu fails on libc headers | `gdn.c` must not include libc; it includes only `io.h`/`params.h` |
| `sqrt`/`div` functional-unit error | this demo uses Newton rsqrt only; do not add `sqrtf` |
| `max |FPGA−CPU|` large | verify `params.h`/NCHW layout; check the wrapper binds `gdn` |
| `fpgautil` not found | use `fpga_manager` + byte swap; see `FIX_REPORT_pynqz2.txt` |
| `makeStrategy(): no strategy ... GDN` | apply the quantization patch (5.2) + rebuild |
| `ImportError: DLL load failed while importing aidge_quantization` (Windows) | the 0.11 wheel needs its MSVC/CUDA runtime deps; use a Linux env (recommended) or build AIDGE from source |
| `FileNotFoundError: analysis/dataset_64_uint8.npz` | run `python tools/pack_dataset.py` (5.3) |
| `aidge_export_c` import: `dtype.any_or_packed_*` missing | use AIDGE 0.11.0 or build core from source |
| `aidge_export_c` import: CUDA DLL error | AIDGE 0.11 imports `aidge_backend_cuda`; install its CUDA deps or use 0.10.1/source |
| PTQ per-channel crash | use `channel_wise=False` on this graph |

---

## 12. Why this shape and parameters

- **Small C/H/W** keeps the first PL build quick and the resource report
  readable; it exercises the same nested-loop/GDN datapath as the full layer.
- **Beta ~0.5–0.7, gamma ~0.005–0.035, x ∈ [-0.5, 0.5]** keeps
  `norm` in ≈[0.5, 1.0], where the Newton seed `0.5*(1+z)` converges in 5
  steps. For the full model use the range-robust piecewise-linear +
  step-clamped Newton form (`docs/model-optimization.md`).
- **No `sqrtf`/division** avoids Bambu’s missing sqrt functional unit.

---

## 13. Next after this demo

1. Replace the demo parameters with the real `g_a.1` gamma/beta (C=192) and
   measure the larger GDN block.
2. Add a `conv + GDN` layer (start with `g_a.2` or a small conv).
3. Move to the tiled full encoder with int8 convs, DDR weight streaming and
   double buffering; see `docs/pynq-deployment.md`.
4. Apply the AIDGE quantization patch and validate the full int8 path.
