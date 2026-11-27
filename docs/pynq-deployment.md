# PYNQ-Z2 deployment — PS/PL acceleration plan

Target: **Xilinx Zynq-7020 (XC7Z020) on PYNQ-Z2**, the development proxy for a
space-qualifiable SoC. The encoder `g_a` runs on the rover; the decoder `g_s`
runs on the platform side.

---

## 1. Board and model budget

| Block | Resource |
|---|---|
| PS | 2× Cortex-A9 (~667 MHz), NEON/FPU, DDR3 |
| PL | ~85K logic cells, 53K LUT, 106K FF |
| DSP | **220 × DSP48E1** |
| BRAM | **4.9 Mb ≈ 630 KB** |
| PL clock | 100–200 MHz (current target: 100 MHz) |

Model at 64×64 (q8, encoder only):

| Metric | Value |
|---|---:|
| MACs | **383.8 M** |
| Params | 3.51 M |
| int8 weights | **3.39 MB** |
| Peak activation | 192×32×32 int8 = **196 KB** |
| Double-buffered activation | ~392 KB |
| Latent output | 320×4×4 = **5 KB** |

Dominant layers: `g_a.2` **61.5%** of MACs, `g_a.4` 15.4%, GDN 12.9%,
`g_a.6` 6.4%, `g_a.0` 3.8%.

---

## 2. Recommended partition: cut at the latent

```text
PS (Cortex-A9)                              PL (Artix-7)
────────────────────────                   ──────────────────────────────
image load / int8 quantize  ──AXI HP0──►   conv0 → GDN1 → conv2 → GDN3
                                           → conv4 → GDN5 → conv6
ANS/rANS, bitstream         ◄──AXI HP0──   latent (5 KB)
control / link                              weights streamed from DDR
```

- PS↔PL traffic per image: **~17 KB** (12 KB image + 5 KB latent).
- Avoid per-layer PS↔PL cuts: a single 192×32×32 feature map is ~196 KB int8
  and crossing several per image costs more than the compute.
- Keep intermediates inside PL/DDR; only the latent crosses to PS.

---

## 3. Memory strategy

- **Weights cannot live in BRAM**: 3.39 MB int8 > 630 KB.
  Stream them from DDR through the AXI HP port, from a weight buffer prepared
  by PS. Reuse the same weights across images/batches.
- **Activation tiles in BRAM**, double-buffered:
  - 192×32×32 int8 = 196 KB; two buffers ≈ 392 KB.
  - With 128 channels this drops to 131 KB per buffer — one reason channel
    reduction is attractive.
- **Weight tile per output-channel group** for the hot layer:
  - 16 output channels × 192 input × 5×5 int8 ≈ 77 KB; fits BRAM.
- Line buffers for 5×5 stride-2 convs; use streaming/tiling rather than
  materializing full layers.

---

## 4. Compute strategy

- **int8 convs**: DSP48E1 packs 2 int8 MACs. 220 DSPs → ~440 MAC/cycle
  theoretical; realistic 64–128 MAC/cycle after routing/line-buffer limits.
  - 64 MAC/cycle @100 MHz → ~60 ms
  - 128 MAC/cycle @100 MHz → ~30 ms
  - 128 MAC/cycle @150 MHz → ~20 ms
- **5×5 → two 3×3** (retrain) reduces MACs ~24% and maps better to DSPs.
- **192 → 128 channels / 320 → 256** (retrain) roughly halves MACs and params;
  target ~130 M MACs and ~1.8 MB weights → ~10–20 ms envelope.
- **GDN**:
  - Exact float `sqrt` is HLS-hostile; use the piecewise-linear + step-clamped
    Newton approximation validated at −0.002 dB (Conv/Mul/Add/ReLU only), or a
    fixed-point rsqrt LUT/Newton.
  - GDN gamma is a 192×192 1×1 conv (36 KB int8/block); ~47 M MACs total.
- **DDR weight bandwidth**: 3.39 MB/image at ~1 GB/s ≈ 3–4 ms; amortize by
  processing multiple images per weight load.

---

## 5. Implementation phases

| Phase | Work | Deliverable |
|---|---|---|
| **A** | PS-only optimized C baseline | latency / CPU load / PSNR reference |
| **B** | int8 `g_a.2` accelerator on PL (LoCod) | biggest layer offloaded; DSP/BRAM/AXI numbers |
| **C** | add GDN + `g_a.4`, then full `g_a` chain | full encoder on PL, latent to PS |
| **D** | PS ANS/rANS entropy + bitstream | end-to-end image latency/bpp |
| **E** | retrain 5×5→3×3, channel reduction | re-synthesize final architecture |

Start with `g_a.2`: if its PL speedup and AXI cost are acceptable, chain the
rest; if not, the bottleneck is memory/DMA and the design must change before
investing in the full encoder.

---

## 6. LoCod / DMA notes

- Keep the accelerator signature `void f(struct param*, struct result*)`.
- Do **not** put the 14 MB of weights in `param` (LoCod `memcpy`s the struct)
  or as `static const` in the bitstream. Use a DDR weight buffer read through
  the AXI master (`M_AXI` → S_AXI_HP0), as in LoCod's `bram_to_ddr` demo.
- Use large aligned DMA bursts and double buffering; flush/invalidate caches
  around `/dev/mem` buffers.
- The authored pynqz2 top already maps control at `0x43C00000` (GP0) and
  DDR/DMA through HP0; base address matches `locod-core/src/pynqz2_defs.h`.
- Build flow: `locod.sh -t pynqz2` (see `FIX_REPORT_pynqz2.txt`), load
  `fpga.bin` via `fpga_manager`, run `locod-cpu`.

---

## 7. Metrics to report

- Per-image latency (hardware timer + PS wall clock), throughput.
- Vivado: LUT / FF / DSP / BRAM utilization, WNS/WHS.
- PS CPU load, power.
- bpp / PSNR / MS-SSIM.
- Compare: PS-only float · PS-only int8 · PL hot layer · PL full encoder ·
  retrained model.

---

## 8. Current blockers

1. **AIDGE PTQ does not know `GDN`** — apply the one-line patch in
   `aidge_ext_c/patches/0001-aidge-quantization-gdn.md` and rebuild AIDGE.
2. **`aidge_export_c` NHWC conv config bug** — H/W can receive the channel
   count after a custom op; use NCHW or correct the generated headers.
3. **int8 Q/DQ boundaries** — a float GDN between int8 convs produces
   `Quantizer`/`Dequantizer` meta-ops that `aidge_export_c` 0.6.0 expands into
   `Cast`/`Round`/`Select`; keep the model float or stitch the int8 convs and
   float GDN kernel by hand until this is fixed.
4. **Monolithic encoder C** (`locod_test_encoder_model/accel.c`) has 6.2 MB
   workspace + 14 MB static weights; it is a golden reference, not
   synthesizable. Replace with the tiled/streaming per-layer design.
