"""Pack DataSet PNG tiles into the .npz used by the optimization/PTQ tools.

Usage:
    python tools/pack_dataset.py
    python tools/pack_dataset.py --data DataSet --out analysis/dataset_64_uint8.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_split(split_dir: Path, size: int | None) -> np.ndarray:
    files = sorted(split_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNGs in {split_dir}")
    out = np.empty((len(files), 3, size or 0, size or 0), dtype=np.uint8)
    for i, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        if size:
            img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        if i == 0:
            out = np.empty((len(files), *arr.shape), dtype=np.uint8)
        out[i] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "DataSet")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "analysis" / "dataset_64_uint8.npz")
    ap.add_argument("--size", type=int, default=0,
                    help="resize to size x size (0 = keep source size)")
    args = ap.parse_args()

    train = load_split(args.data / "train", args.size or None)
    test = load_split(args.data / "test", args.size or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, train=train, test=test)
    print(f"wrote {args.out}")
    print(f"  train {train.shape} {train.dtype}")
    print(f"  test  {test.shape} {test.dtype}")


if __name__ == "__main__":
    main()
