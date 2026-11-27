# Model Optimization Log — Ballé factorized encoder (q8, 64×64)

> **Repo:** `C:\Users\HP\Desktop\AIDGE\aidge-comp-linux` (friend's repo, branch `main`)
> **Platform:** Windows, 2026-11-27
> **Model:** `Model/bmshj2018_factorized_encoder.onnx` — CompressAI `bmshj2018_factorized`, quality 8, N=192, M=320
> **Dataset:** `DataSet/` — 9,632 train / 1,248 test, 64×64 RGB space images
> **Python envs (outside this repo):** `E:\spaceviv\.venv-ml` (torch/compressai), `E:\spaceviv\.venv-acetone` (onnxruntime), `E:\spaceviv\.venv` (AIDGE 0.10.1)
> **Reference test set:** all **1,248** test tiles, per-image averages (the first 32 tiles are much easier and are **not** used for decisions)

---

## 0. Status snapshot

| Area | Status |
|---|---|
| Baseline profile (params / MACs / workspace) | ✅ measured |
| Baseline RD on the full 64×64 test set | ✅ 2.1046 bpp · 37.813 dB · 0.9481 SSIM |
| Repo ONNX == CompressAI q8 | ✅ verified (1.8e-6 relative) |
| int8 QDQ simulation (weights + activations) | ✅ measured: −0.474 dB, +0.084 bpp |
| int8 weight-only | ✅ measured: −0.074 dB, +0.012 bpp |
| Hardware-friendly GDN for the encoder | ✅ piecewise-linear + step-clamped Newton, **−0.002 dB** |
| AIDGE PTQ on the raw model | ❌ blocked: `Sqrt` / `Pow` have no PTQ strategy |
| PReLU / scaled-PReLU replacement | ❌ rejected (collapses to 8.5 / 12.2 dB before training) |
| Decoder (IGDN) approximation | ❌ deferred (14.9 dB with the same scheme; decoder kept exact) |
| Modal GPU training harness | ✅ smoke-tested (T4, torch 2.15, compressai 1.2.8); full retrain not started |
| 5×5 → two 3×3 / channel reduction | ⏳ planned (training harness supports it) |

---

## 1. Baseline

### 1.1 Parameters, MACs, workspace at 64×64

From `tools/profile_encoder.py` (freezes the data-dependent GDN reshapes first):

| Layer | Op | Output | MACs | Params |
|---|---|---|---:|---:|
| `g_a.0` | Conv 3→192, 5×5, s2 | 1×192×32×32 | 14.75 M | 14,592 |
| `g_a.1` | GDN (1×1 conv 192→192) | 1×192×32×32 | 37.75 M | 36,864 + 192 |
| `g_a.2` | Conv 192→192, 5×5, s2 | 1×192×16×16 | **235.93 M** | 921,792 |
| `g_a.3` | GDN (1×1 conv 192→192) | 1×192×16×16 | 9.44 M | 36,864 + 192 |
| `g_a.4` | Conv 192→192, 5×5, s2 | 1×192×8×8 | 58.98 M | 921,792 |
| `g_a.5` | GDN (1×1 conv 192→192) | 1×192×8×8 | 2.36 M | 36,864 + 192 |
| `g_a.6` | Conv 192→320, 5×5, s2 | 1×320×4×4 | 24.58 M | 1,536,320 |
| **Total** | | | **383.78 M** | **3,505,679** |

- Weights: **14.02 MB fp32 → 3.39 MB int8**.
- The friend's `locod_test_encoder_model/accel.c` uses a **6,213,632-byte** workspace (no buffer reuse).
- With liveness-based buffer reuse: **2.51 MB fp32** peak, **~627 KB int8**. The 627 KB is just under the 630 KB BRAM of the Zynq-7020; GDN's temporaries are what push it that high.
- `g_a.2` alone is **61.5% of all MACs**.

### 1.2 RD baseline on the full test set

`tools/eval_rd_64.py` (or `tools/eval_approx_gdn.py` for the exact side):

| Metric | Value |
|---|---:|
| bpp | **2.1046** |
| PSNR | **37.8128 dB** |
| SSIM | **0.94812** |
| Images | 1,248 |

> **Important:** the first 32 test tiles give 1.385 bpp / 39.598 dB. Those tiles are much easier. All optimization decisions below use the full 1,248-tile set.

The repo ONNX was compared against `compressai.zoo.bmshj2018_factorized(quality=8)` on a real tile: latent shape `(1,320,4,4)`, max relative difference **1.8e-6**. So the full pretrained decoder / entropy model is available for RD measurement and fine-tuning.

---

## 2. Tools created

All commands are run from the repo root. `$PY` = `E:\spaceviv\.venv-ml\Scripts\python.exe`, `$PYA` = `E:\spaceviv\.venv-acetone\Scripts\python.exe`.

### `tools/profile_encoder.py`
Profiles an ONNX encoder at a fixed input size. It first freezes the dynamic GDN `Reshape` shape inputs with ONNX Runtime (same trick as `test_acetone.py`), then reports params, per-op/per-layer MACs, and peak activation workspace under a liveness allocator.

```powershell
& $PYA tools\profile_encoder.py --input 1,3,64,64 --top 12 --json analysis\encoder_profile_q8_64.json
```

### `tools/eval_rd_64.py`
Rate/speed/loss evaluation on `DataSet/<split>` for `bmshj2018_factorized` / `bmshj2018_hyperprior`. Computes bpp from likelihoods, PSNR, single-scale SSIM, and encode/decode time. MS-SSIM is skipped below 160 px (our tiles are 64×64).

```powershell
& $PY tools\eval_rd_64.py --split test --limit 1248 --quality 8 --out analysis\rd_factorized_q8_64_full.json
```

### `tools/prepare_encoder_onnx.py`
Extracts the `x_out` encoder subgraph, freezes the dynamic Reshape targets, prunes the entropy helpers, and verifies the result is bit-identical to the original ONNX.

```powershell
& $PYA tools\prepare_encoder_onnx.py --size 64
# -> analysis/encoder_static_64.onnx (13.4 MB), max |static - original| = 0
```

### `tools/simulate_int8.py`
QDQ simulation of int8 PTQ:
- per-output-channel symmetric int8 weights
- per-tensor symmetric int8 activations, calibrated on `DataSet/train`
- bias and GDN `sqrt/div` stay float32
- decoder + entropy model stay float32

Modes: `weight`, `act`, `both`; clipping: `max`, `mse`.

```powershell
& $PY tools\simulate_int8.py --mode both --clipping max --calib 256 --test-limit 0
```

### `tools/eval_approx_gdn.py`
Evaluates the hardware-friendly encoder GDN against the exact model on the full test set and writes a JSON with the deltas.

```powershell
& $PY tools\eval_approx_gdn.py
```

### `training/approx_gdn.py`
`ApproxGDN`: drop-in GDN/IGDN using only `Conv/Mul/Add/ReLU` (no `Sqrt`, `Pow`, or `Div`). Details in §4.

### `training/model_surgery.py`
Builds a `FactorizedPrior` with optional encoder surgery:
- `act = keep` (leave GDN), `approx` (ApproxGDN), `prelu`, `leaky`, `relu`, `scaled_prelu`
- `factor_layers = (2,4,6)` replaces selected 5×5 stride-2 convs with two 3×3 convs
- `mids = {2:192, 4:192, 6:256}` controls the intermediate width

### `training/train_core.py`
Training/evaluation harness. Fine-tunes with CompressAI's `RateDistortionLoss(lmbda=0.18)` (q8), Adam + cosine, random flips, per-image PSNR/SSIM/bpp. Saves `last.pt`, `best.pt`, `history.json`, `result.json`.

### `training/modal_train.py`
Modal app (T4) with the dataset volume `/data` and checkpoints under `/data/runs/<name>`. Smoke test and full run entrypoints.

### `tools/modal_check.py`
One-off Modal GPU/torch/compressai check.

---

## 3. Experiment 1 — int8 PTQ simulation

### 3.1 Why simulation

AIDGE's PTQ (`aidge_quantization.quantize_network`) refuses the CompressAI GDN decomposition:

```
[ERROR] - GraphView can't be quantized : node type Sqrt is not supported !
```

In `aidge_quantization` `PTQTypes.hpp`, the strategy sets cover `Div`, `Shape`, `Gather`, etc., but **not `Sqrt` or `Pow`**, so `makeStrategy()` aborts before quantization. The raw encoder therefore cannot be PTQ'd in AIDGE as-is.

The QDQ simulation measures the same numerical effect while keeping `sqrt/div` float (which is what the friend's `accel.c` already does).

### 3.2 Results (full 1,248 test tiles, calib = 256 train tiles)

| Variant | bpp | PSNR (dB) | SSIM | ΔPSNR | Δbpp | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 2.1046 | 37.813 | 0.94812 | — | — | — |
| int8 weights | 2.1167 | 37.739 | 0.9476 | **−0.074** | +0.0120 | −0.0006 |
| int8 weights + activations (`max`) | 2.1882 | 37.339 | 0.9436 | **−0.474** | +0.0836 | −0.0045 |

- Weights: **3.39 MB int8** (from 13.57 MB fp32 of conv/gamma weights).
- Activation clipping with `MSE` was only tested on 32 tiles and was worse for PSNR/SSIM than `max`; keep `max`.
- **Conclusion:** int8 is safe for this model. The quality budget is ≈ **0.5 dB** for full int8, before any retraining.

### 3.3 Caveat

This is a **QDQ simulation**, not a real int8 kernel. It tells us the numerical budget and the memory/throughput direction, not the final hardware number.

---

## 4. Experiment 2 — hardware-friendly GDN (encoder)

### 4.1 What we tried and why it failed

| Attempt | Initial full-model result | Verdict |
|---|---|---|
| GDN → `PReLU(init=0.1)` | 8.5 dB (encoder-only, 32 tiles) | ❌ too destructive |
| GDN → `scaled_prelu` (per-channel gain from `1/sqrt(beta)`) | 12.2 dB | ❌ still not viable without a long/full retrain |
| GDN → piecewise-linear `rsqrt` + Newton | **37.811 dB vs 37.813 exact** | ✅ |

GDN is **channel-mixing divisive normalization** (`gamma` is a full C×C 1×1 conv), not an elementwise activation; an elementwise replacement changes the latent distribution too much.

### 4.2 The working approximation

For each GDN:

```
z = beta + conv1x1(x*x, gamma)                  # same as GDN
z = clamp(z, z0, zK)                            # via ReLU only
r = y0 + m0*(z-z0)
  + sum_i (m_i - m_{i-1}) * ReLU(z - z_i)       # chord seed, K segments
r = ReLU(r) + 1e-12
repeat 2x:                                      # step-clamped Newton
    t = 3 - z*r*r
    t = ReLU(t)                                 # lower clamp 0
    t = 3 - ReLU(3 - t)                         # upper clamp 3
    r = 0.5*r*t
y = x * r                                       # GDN
# IGDN would use y = x * z * r
```

**Configuration used:** `K=16`, `z0=0.01`, `zK=10000`, 2 Newton iterations.
The step clamp is essential: without it, large-norm outliers (observed `norm` up to **8,177** on the full test set) make Newton diverge/NaN.

### 4.3 Result (encoder only, full 1,248 test tiles)

| Model | bpp | PSNR (dB) | SSIM | ΔPSNR | Δbpp | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| exact GDN | 2.10465 | 37.81285 | 0.948123 | — | — | — |
| approx GDN (`K=16`, 2 iters) | 2.10491 | 37.81101 | 0.948064 | **−0.0018** | +0.00027 | −0.00006 |

This is effectively lossless and uses **no `Sqrt`, `Pow`, or `Div`** — only `Conv2D`, `Mul`, `Add`, `ReLU`. That is exactly the operator set needed for AIDGE PTQ and `aidge_export_c` (which supports Conv2D, Add, Sub, Mul, Div, ReLU; no Max/Min/Clip, and the approximation avoids them).

### 4.4 Decoder (IGDN) — deferred

The same scheme on `g_s` produces 14.9 dB on the full test set. The decoder IGDN norm range is tiny (`1e-6 … 1.5`), and float32 cancellation in the chord seed / clamp makes the approximation unreliable there. For now:
- **Encoder GDN → approximate** (deployment target / rover side).
- **Decoder IGDN → exact** (`sqrt`) on the platform side.

A future option is a custom fast-rsqrt kernel (bit-hack + 2 Newton steps) for both GDN and IGDN, which avoids float32 chord cancellation entirely.

### 4.5 Failed intermediate variants (kept for the record)

1. **`min` of chord lines** — extrapolation of other segments produced negative values; fixed by the hinge/ReLU formulation.
2. **Direct chord + Newton over `[0.01, 3000]`** — 32-tile result looked fine (38.545 dB), but the full 128-tile set exposed outliers up to 8,177; some images collapsed to 8–12 dB.
3. **Wide `z0=1e-6`** — float32 cancellation produced NaNs; the encoder's actual norm floor is ~0.013, so `z0=0.01` is correct.
4. **Unclamped Newton** — diverged for large `z`; the step clamp `t ← clamp(3 − z r², 0, 3)` fixed it.
5. **`zK=100000`** — still stable with the clamp, but `zK=10000` covers the observed encoder range with better accuracy.

---

## 5. Modal training harness

- Modal CLI **1.6.0**, profile `eurosat`, GPU **Tesla T4**.
- Image: Debian slim + `torch 2.15.0+cu132`, `compressai 1.2.8`, `numpy 1.26.4`, `pytorch-msssim`.
- Dataset volume: `aidge-data`, file `/dataset_64_uint8.npz` (9,632 train + 1,248 test, uint8, 127.5 MB).
- Smoke test: 1 epoch / 128 images, **4.2 s**, loss high as expected for a random-init surgery.

```powershell
modal volume create aidge-data
modal volume put aidge-data analysis\dataset_64_uint8.npz /dataset_64_uint8.npz
modal run training\modal_train.py::smoke
# full run example:
modal run training\modal_train.py --name approx_3x3 --epochs 30 --act approx --factor 2,4,6 --mids 2:192,4:192,6:256
```

> The full retrain has **not** been run yet — only the smoke test. `model_surgery.py` supports `act=approx` so the next run keeps GDN's channel mixing while using the hardware-friendly rsqrt.

### 5.1 Retraining plan (next)

1. **Stage A:** `act=approx`, no 5×5 factoring — verify fine-tuning stability and reproduce the baseline (should stay ~37.8 dB).
2. **Stage B:** `factor=2,4,6` (5×5 → 3×3) with a **warm start**, not random init. Expected:
   - MACs 383.8 M → ~290 M (**−24%**)
   - params 3.51 M → ~2.63 M (**−25%**)
3. **Stage C:** channel reduction `N=192→128`, `M=320→256` (needs a separate training run; expected ~−50% MACs/params).
4. Evaluate every variant on the full 1,248-tile test set and repeat the int8 QDQ simulation.

> Random-initialized 5×5 → 3×3 replacement collapsed the model in the smoke test (12.8 dB after one epoch on 128 images). Use a warm start / least-squares factorization or factor after Stage A.

---

## 6. Decisions

1. **Reference = full 1,248 test tiles, per-image metrics.** The 32-tile numbers are deprecated for decisions.
2. **int8 is accepted**: full int8 (weights + activations, `max` clipping) costs ≈0.47 dB and 0.084 bpp; weight-only costs ≈0.07 dB. Weights drop 14.02 MB → 3.39 MB; workspace 2.51 MB fp32 → ~627 KB int8 under perfect reuse.
3. **Keep GDN, replace its transcendentals**: encoder `rsqrt` via `K=16` chord + 2 step-clamped Newton iterations (Conv/Mul/Add/ReLU only). Quality is unchanged (−0.002 dB). This unblocks AIDGE PTQ in principle, because `Sqrt`/`Pow` are gone.
4. **Decoder stays exact** for now; IGDN approximation is deferred.
5. **Retraining uses Modal T4**; the harness is ready, the full run is the next action.
6. **5×5 → two 3×3** only after a warm start; `g_a.2` is the dominant layer (61.5% MACs), so this is the highest-value structural change.
7. **Do not use PReLU/ReLU replacement** for GDN unless doing a full from-scratch retrain — it destroys the pretrained latent.

---

## 7. Reproduction commands

```powershell
cd C:\Users\HP\Desktop\AIDGE\aidge-comp-linux
$PY  = "E:\spaceviv\.venv-ml\Scripts\python.exe"
$PYA = "E:\spaceviv\.venv-acetone\Scripts\python.exe"
$PYAIDGE = "E:\spaceviv\.venv\Scripts\python.exe"

# 0) baseline profile (params/MACs/workspace)
& $PYA tools\profile_encoder.py --input 1,3,64,64 --top 12 --json analysis\encoder_profile_q8_64.json

# 1) baseline RD on the full test set
& $PY tools\eval_rd_64.py --split test --limit 1248 --quality 8 --out analysis\rd_factorized_q8_64_full.json

# 2) static encoder-only ONNX (needed for PTQ/profiling)
& $PYA tools\prepare_encoder_onnx.py --size 64

# 3) int8 QDQ simulation (full test)
& $PY tools\simulate_int8.py --mode weight --calib 256 --out analysis\int8_weight_full.json
& $PY tools\simulate_int8.py --mode both  --clipping max --calib 256 --out analysis\int8_both_max_full.json

# 4) hardware-friendly encoder GDN (full test)
& $PY tools\eval_approx_gdn.py --out analysis\approx_gdn_encoder_full.json

# 5) Modal
modal volume put aidge-data analysis\dataset_64_uint8.npz /dataset_64_uint8.npz
modal run training\modal_train.py::smoke
modal run training\modal_train.py --name approx_3x3 --epochs 30 --act approx --factor 2,4,6 --mids 2:192,4:192,6:256
```

> `tools/simulate_int8.py` accepts `--data`, `--calib`, `--test-limit`, `--out`; `--test-limit 0` means all 1,248 test tiles.

---

## 8. Artifacts

| Path | Size | Purpose |
|---|---:|---|
| `analysis/encoder_profile_q8_64.json` | 16 KB | per-node params/MACs/shapes |
| `analysis/encoder_static_64.onnx` | 13.4 MB | static encoder-only graph |
| `analysis/rd_factorized_q8_64_test32.json` | 9 KB | **legacy** 32-tile RD (deprecated) |
| `analysis/int8_weight_full.json` | small | int8 weight-only, full test |
| `analysis/int8_both_max_full.json` | small | int8 weights+acts, full test |
| `analysis/int8_weight.json`, `int8_both_max.json`, `int8_both_mse.json` | small | 32-tile int8 experiments |
| `analysis/approx_gdn_encoder_full.json` | small | exact vs approx GDN, full test |
| `analysis/dataset_64_uint8.npz` | 127.5 MB | packed train/test tiles for Modal |
| `analysis/calib_64x64_128.npy` | 6 MB | 128 calibration tiles |
| `analysis/test_64x64_32.npy` | 1.5 MB | 32 test tiles (legacy) |
| `analysis/smoke/` | small | local training smoke checkpoints |
| Modal volume `aidge-data:/dataset_64_uint8.npz` | 127.5 MB | GPU training data |

---

## 9. Open issues / next steps

1. **AIDGE PTQ on the approximated encoder** — export `g_a` with `ApproxGDN` expanded to `Conv/Mul/Add/ReLU`, import into AIDGE, constant-fold, run `auto_assign_node_precision` + `quantize_network`, verify float vs int8 latents.
2. **int8 C export** — use `aidge_export_c` (`dtype="int8"`, target `host_c`) on the approximated graph; verify against PyTorch/AIDGE CPU. If GDN is kept as a custom op, register a C kernel with the fast rsqrt.
3. **5×5 → 2×3×3 retrain** — Stage B on Modal with a warm start; measure the RD drop and the MACs/params reduction.
4. **Channel reduction** — Stage C (`N=128`, `M=256`); train from scratch or prune+fine-tune; target the 7020 BRAM/DSP budget.
5. **Decoder** — exact IGDN for now; later a custom fast-rsqrt kernel or a decoder retrain.
6. **Real hardware numbers** — replace QDQ with actual int8 kernels, then regenerate `locod_test_encoder_model/accel.c`, run Bambu HLS, Vivado, and the PYNQ-Z2 flow; measure latency/resources.
7. **Documentation for the friend/repo** — `locod_test_encoder_model/params.h` (~14 MB) is still missing from the repo; the encoder C cannot be rebuilt from the clone without it. Keep the generated weights/checkpoints in the Modal volume or a release.

---

## 10. Caveats

- All quantized numbers are **QDQ simulations**, not measured int8 kernels.
- Metrics are on **64×64** tiles. At 256×256, MACs and workspace scale ×16.
- PSNR/SSIM are per-image averages (not pooled per batch). Pooled MSE gives a lower PSNR (~36.4 dB) because it weights high-error images more; don't mix the two.
- Calibration used 256 train tiles; rare high-norm tiles can exceed `zK=10000` (observed max 8,177 on test). Extend the range or use a custom kernel if the full training set shows larger outliers.
- The encoder GDN approximation is validated on the **q8** model; re-validate after any retraining.
- The repo ONNX is the **encoder only**; the decoder/entropy model used for RD comes from `compressai.zoo.bmshj2018_factorized(quality=8)`.
