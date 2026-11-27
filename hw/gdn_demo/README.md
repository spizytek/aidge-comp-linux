# GDN on PYNQ-Z2 — bring-up demo

This is a **small, self-contained GDN accelerator** for the PYNQ-Z2 / LoCod
flow. It validates the new GDN unit operator on the PL before we attempt the
full 4-conv + 3-GDN encoder.

Shape: **C=16, H=W=8, NCHW, batch=1**, `1/sqrt` via a 5-step Newton iteration
(only add/mul/sub + comparisons; no `sqrtf`, no division).

---

## 1. What it tests

```
input x[16,8,8]
  → for each output channel o and pixel p:
       norm = beta[o] + sum_j gamma[o,j] * x[j,p]^2
       y[o,p] = x[o,p] / sqrt(norm)       # Newton rsqrt
  → output y[16,8,8]
```

- `gamma[16,16]` and `beta[16]` are compile-time constants in `params.h`.
- CPU reference and FPGA call use the **same** `gdn.c`, so the numerical
  comparison isolates the HLS/FPGA implementation.
- This is a **correctness / resource bring-up test**, not the full encoder. The full
  encoder needs int8, tiling and DDR weight streaming.

---

## 2. Files

| File | Purpose |
|---|---|
| `main.c` | LoCod CPU driver, test vector, CPU reference, PASS/FAIL |
| `gdn.c` | the accelerator (`void gdn(struct param_gdn*, struct result_gdn*)`) |
| `io.h` | fixed-size structs + GDN dimensions |
| `params.h` | `GDN_GAMMA[16][16]`, `GDN_BETA[16]` |
| `locod_host.h` | desktop shim for `FPGA`/`wait`/timer |
| `Makefile` | desktop build |

---

## 3. Desktop check (no board)

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
only checks the math and build.

---

## 4. LoCod software-only build

From the **LoCod root**:

```bash
cd /path/to/LoCod

bash ./locod.sh -t pynqz2 --no-hard \
  -f /path/to/aidge-comp-linux/hw/gdn_demo/main.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/gdn.c \
     /path/to/aidge-comp-linux/hw/gdn_demo/io.h \
     /path/to/aidge-comp-linux/hw/gdn_demo/params.h
```

This produces the CPU executable without requiring Vivado/SDK, useful to
validate the LoCod parsing and the CPU reference.

---

## 5. Full hardware build (bitstream)

From the **LoCod root** (Vivado + SDK installed, as on the friend’s machine):

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
  with literal arguments.  Do not write their syntax in comments: `locod.sh`
  greps the raw text of `main.c`.
- Do not move the accelerator call into another file.

---

## 6. Run on the PYNQ-Z2

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

The `FPGA time` is the hardware cycle-counter value from the accelerator
wrapper; `max |FPGA - CPU|` should be well below `1e-3` (float scheduling can
reorder additions, so it is not always bit-exact).

If the board rejects the bitstream with “could not find a sync word”, apply
the byte-swap described in `FIX_REPORT_pynqz2.txt` and load
`fpga_swapped.bin` through `/sys/class/fpga_manager/fpga0/firmware`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no accelerator generated` | `main.c` not first, or call split across lines, or `FPGA(`/`init_locod(` text in comments |
| Bambu cannot find `params.h` | include it in the `-f` list (as above) |
| Bambu fails on libc headers | `gdn.c` must not include libc; it currently includes only `io.h`/`params.h` |
| `sqrt`/`div` functional-unit error | this demo uses Newton rsqrt only; do not add `sqrtf` |
| `max |FPGA−CPU|` large | verify `params.h`/NCHW layout; check the generated wrapper binds `gdn` |
| `fpgautil` not found | use the `fpga_manager` path + byte swap; see `FIX_REPORT_pynqz2.txt` |

---

## 8. Why this shape and parameters

- **Small C/H/W** keeps the first PL build quick and the resource report
  readable; it exercises the same nested-loop/GDN datapath as the full layer.
- **Beta ~0.5–0.7, gamma ~0.005–0.035, x ∈ [-0.5, 0.5]** keeps
  `norm` in ≈[0.5, 1.0], where the Newton seed `0.5*(1+z)` converges in 5
  steps. For a general GDN, use the range-robust piecewise-linear +
  step-clamped Newton form (see `docs/model-optimization.md`).
- **No `sqrtf`/division** avoids Bambu’s missing sqrt functional unit.

---

## 9. Next after this demo

1. Replace the demo parameters with the real `g_a.1` gamma/beta (C=192) and
   measure a larger GDN block.
2. Add a `conv + GDN` layer (start with `g_a.2` or a small conv).
3. Move to the tiled full encoder with int8 convs, DDR weight streaming and
   double buffering; see `docs/pynq-deployment.md`.
