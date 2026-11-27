"""Rate-distortion + speed evaluation for the 64x64 space dataset.

Mirrors E:\\spaceviv\\tools\\eval_rd.py, but:
  * defaults to this repo's DataSet/
  * computes single-scale SSIM as well, because MS-SSIM needs >=160 px
  * reports enc/dec time per image and a JSON summary

Usage:
    python tools/eval_rd_64.py --split test --limit 32 --quality 8
    python tools/eval_rd_64.py --split train --limit 200 --quality 3 5 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
from torchvision.transforms.functional import to_tensor

from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior

ROOT = Path(__file__).resolve().parent.parent
ZOO = {"factorized": bmshj2018_factorized, "hyperprior": bmshj2018_hyperprior}


def load_images(split: str, limit: int | None):
    files = sorted((ROOT / "DataSet" / split).glob("*.png"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no PNGs in {ROOT / 'DataSet' / split}")
    out = []
    for path in files:
        img = Image.open(path).convert("RGB")
        out.append((path.name, to_tensor(img).unsqueeze(0)))
    return out


def metrics(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[float, float, float]:
    n, _, h, w = x.shape
    x_hat = x_hat.clamp(0, 1)
    mse = F.mse_loss(x, x_hat).item()
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim_v = float(ssim(x, x_hat, data_range=1.0))
    if min(h, w) >= 160:
        msssim_v = float(ms_ssim(x, x_hat, data_range=1.0))
    else:
        msssim_v = float("nan")
    return psnr, ssim_v, msssim_v


def evaluate(model, imgs):
    rows = []
    for name, x in imgs:
        captured = {}
        handle = model.g_s.register_forward_pre_hook(
            lambda mod, inp: captured.__setitem__("y_hat", inp[0].detach()))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        t_full = time.perf_counter() - t0
        handle.remove()

        with torch.no_grad():
            t1 = time.perf_counter()
            x_hat = model.g_s(captured["y_hat"])
            t_dec = time.perf_counter() - t1

        n, _, h, w = x.shape
        bits = sum((-torch.log2(lk)).sum() for lk in out["likelihoods"].values())
        bpp = float(bits / (n * h * w))
        psnr, ssim_v, msssim_v = metrics(x, x_hat)
        rows.append({"image": name, "bpp": bpp, "psnr": psnr,
                     "ssim": ssim_v, "msssim": msssim_v,
                     "enc_s": max(t_full - t_dec, 0.0), "dec_s": t_dec})
        print(f"  {name:28s} bpp={bpp:7.4f}  psnr={psnr:7.3f}  "
              f"ssim={ssim_v:.4f}  enc={rows[-1]['enc_s']*1e3:6.2f}ms "
              f"dec={t_dec*1e3:6.2f}ms")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--variant", nargs="+", default=["factorized"],
                    choices=tuple(ZOO))
    ap.add_argument("--quality", nargs="+", type=int, default=[8])
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    imgs = load_images(args.split, args.limit)
    print(f"evaluating {len(imgs)} images from DataSet/{args.split} "
          f"({imgs[0][1].shape[-1]}x{imgs[0][1].shape[-2]})")

    results = {}
    for variant in args.variant:
        for q in args.quality:
            tag = f"{variant}_q{q}"
            model = ZOO[variant](quality=q, pretrained=True).eval()
            print(f"\n[{tag}]")
            rows = evaluate(model, imgs)
            keys = ("bpp", "psnr", "ssim", "msssim", "enc_s", "dec_s")
            mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
            results[tag] = {"per_image": rows, "mean": mean}
            print(f"  MEAN bpp={mean['bpp']:.4f} psnr={mean['psnr']:.3f} "
                  f"ssim={mean['ssim']:.4f} "
                  f"enc={mean['enc_s']*1e3:.1f}ms dec={mean['dec_s']*1e3:.1f}ms")

    out = args.out or (ROOT / "analysis" / "rd_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
