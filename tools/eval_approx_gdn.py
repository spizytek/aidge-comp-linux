"""Evaluate the hardware-friendly encoder GDN approximation on the dataset.

Usage:
    python tools/eval_approx_gdn.py --data analysis/dataset_64_uint8.npz
    python tools/eval_approx_gdn.py --limit 128 --out analysis/approx.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from compressai.zoo import bmshj2018_factorized  # noqa: E402
from approx_gdn import replace_gdn  # noqa: E402
from train_core import evaluate, load_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=ROOT / "analysis" / "dataset_64_uint8.npz")
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--z0", type=float, default=0.01)
    ap.add_argument("--zk", type=float, default=10000.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis" / "approx_gdn_encoder_full.json")
    args = ap.parse_args()

    _, test = load_data(str(args.data), args.limit or None)
    model = bmshj2018_factorized(quality=args.quality, pretrained=True).eval()
    exact = evaluate(model, test, torch.device("cpu"))
    n = replace_gdn(model.g_a, k=args.k, iters=args.iters,
                    z0=args.z0, zk=args.zk)
    approx = evaluate(model, test, torch.device("cpu"))

    result = {
        "exact": exact,
        "approx_encoder_gdn": approx,
        "config": {"K": args.k, "iters": args.iters, "z0": args.z0,
                   "zk": args.zk, "step_clamped_newton": True,
                   "replaced": n},
        "delta": {k: approx[k] - exact[k] for k in ("bpp", "psnr", "ssim")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
