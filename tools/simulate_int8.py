"""Simulate int8 PTQ on the Ballé factorized encoder (g_a).

AIDGE's PTQ refuses the CompressAI GDN decomposition because `Sqrt` has no
quantization strategy.  This script measures the same numerical effect that a
hardware int8 encoder would see:

  * per-output-channel symmetric int8 weights
  * per-tensor symmetric int8 activations (calibrated on DataSet/train)
  * bias and the GDN sqrt/div stay float32
  * decoder + entropy model stay float32 (they run on the platform / PS)

Reports end-to-end bpp / PSNR / SSIM vs the float baseline, plus int8 weight
size.  This is the "before retraining" number.

Usage:
    python tools/simulate_int8.py --mode weight
    python tools/simulate_int8.py --mode both --clipping max
    python tools/simulate_int8.py --mode both --clipping mse --calib 128
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from compressai.zoo import bmshj2018_factorized

ROOT = Path(__file__).resolve().parent.parent


def load_npy(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).float()


def quant_weight(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 (round-to-nearest)."""
    c = w.shape[0]
    scale = w.reshape(c, -1).abs().amax(dim=1).clamp_min(1e-8) / 127.0
    scale = scale.reshape(c, *([1] * (w.dim() - 1)))
    return torch.clamp(torch.round(w / scale), -127, 127) * scale


def mse_scale(x: torch.Tensor, candidates: int = 32) -> float:
    """Pick a clipping threshold in [0.90, 1.00]*max|x| minimising MSE."""
    amax = float(x.abs().max())
    if amax == 0.0:
        return 1e-8
    best_s, best_mse = amax / 127.0, float("inf")
    for frac in np.linspace(0.90, 1.0, candidates):
        clip = amax * float(frac)
        s = clip / 127.0
        xq = torch.clamp(torch.round(x / s), -127, 127) * s
        mse = float(((x - xq) ** 2).mean())
        if mse < best_mse:
            best_mse, best_s = mse, s
    return best_s


def quant_act(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


class QConv(torch.nn.Module):
    """QDQ wrapper around nn.Conv2d."""

    def __init__(self, conv: torch.nn.Conv2d, act_scale: float | None):
        super().__init__()
        self.conv = conv
        self.act_scale = act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = quant_weight(self.conv.weight)
        if self.act_scale is not None and self.act_scale > 0:
            x = quant_act(x, self.act_scale)
        return F.conv2d(x, w, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


def g_a_convs(model) -> list[tuple[str, torch.nn.Conv2d]]:
    return [(n, m) for n, m in model.g_a.named_modules()
            if isinstance(m, torch.nn.Conv2d)]


@torch.no_grad()
def collect_ranges(model, calib: torch.Tensor) -> dict[str, float]:
    """Per-conv input abs-max over the calibration set."""
    convs = dict(g_a_convs(model))
    amax = {name: 0.0 for name in convs}
    hooks = []
    for name, conv in convs.items():
        def hook(mod, inp, out, _name=name):
            v = float(inp[0].abs().max())
            amax[_name] = max(amax[_name], v)
        hooks.append(conv.register_forward_hook(hook))
    for i in range(calib.shape[0]):
        model.g_a(calib[i:i + 1])
    for h in hooks:
        h.remove()
    return amax


@torch.no_grad()
def evaluate(model, test: torch.Tensor) -> dict:
    rows = []
    for i in range(test.shape[0]):
        x = test[i:i + 1]
        out = model(x)  # forward uses g_a (possibly QDQ) + entropy + g_s
        x_hat = out["x_hat"].clamp(0, 1)
        n, _, h, w = x.shape
        bits = sum((-torch.log2(lk)).sum()
                   for lk in out["likelihoods"].values())
        bpp = float(bits / (n * h * w))
        mse = F.mse_loss(x, x_hat).item()
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        rows.append({"bpp": bpp, "psnr": psnr,
                     "ssim": float(ssim(x, x_hat, data_range=1.0))})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("weight", "act", "both"), default="both")
    ap.add_argument("--clipping", choices=("max", "mse"), default="max")
    ap.add_argument("--data", type=Path,
                    default=ROOT / "analysis" / "dataset_64_uint8.npz")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--test-limit", type=int, default=0,
                    help="0 = all test tiles")
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = np.load(args.data)
    train = torch.from_numpy(data["train"]).float().div_(255.0)
    test = torch.from_numpy(data["test"]).float().div_(255.0)
    if args.test_limit:
        test = test[:args.test_limit]
    calib = train[:args.calib]

    model = bmshj2018_factorized(quality=args.quality,
                                 pretrained=True).eval()
    base = evaluate(model, test)
    print(f"float baseline : bpp={base['bpp']:.4f} psnr={base['psnr']:.3f} "
          f"ssim={base['ssim']:.4f}")

    ranges = collect_ranges(model, calib)
    if args.mode in ("both", "act") and args.clipping == "mse":
        # recompute a clipping-optimal scale per conv on the calibration set
        convs = dict(g_a_convs(model))
        inputs: dict[str, list[torch.Tensor]] = {n: [] for n in convs}
        hooks = []
        for name, conv in convs.items():
            def hook(mod, inp, out, _name=name):
                inputs[_name].append(inp[0].detach())
            hooks.append(conv.register_forward_hook(hook))
        for i in range(calib.shape[0]):
            model.g_a(calib[i:i + 1])
        for h in hooks:
            h.remove()
        active = {n: mse_scale(torch.cat(v, 0)) for n, v in inputs.items()}
    else:
        active = {n: v / 127.0 for n, v in ranges.items()}

    qmodel = copy.deepcopy(model)
    total_w_bytes = 0
    for name, conv in list(g_a_convs(qmodel)):
        parent = qmodel.g_a
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1],
                QConv(conv, active[name] if args.mode in ("both", "act") else None))
    for _, conv in g_a_convs(model):
        total_w_bytes += int(conv.weight.numel())  # int8 bytes

    res = evaluate(qmodel, test)
    print(f"int8 {args.mode:6s} ({args.clipping}) : "
          f"bpp={res['bpp']:.4f} psnr={res['psnr']:.3f} "
          f"ssim={res['ssim']:.4f}")
    print(f"  dPSNR={res['psnr']-base['psnr']:+.3f} dB  "
          f"dbpp={res['bpp']-base['bpp']:+.4f}  "
          f"dSSIM={res['ssim']-base['ssim']:+.4f}")
    print(f"  int8 weights: {total_w_bytes/1e6:.2f} MB "
          f"(fp32 {total_w_bytes*4/1e6:.2f} MB)")

    out = args.out or (ROOT / "analysis" /
                       f"int8_{args.mode}_{args.clipping}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "clipping": args.clipping,
                               "quality": args.quality, "calib": args.calib,
                               "float": base, "int8": res,
                               "int8_weight_bytes": total_w_bytes},
                              indent=2))
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
