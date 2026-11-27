"""Training / evaluation core for the optimized Ballé factorized encoder.

Runs locally (CPU/GPU) and on Modal.  Data is a uint8 .npz with `train` and
`test` arrays shaped (N, 3, 64, 64).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from compressai.losses import RateDistortionLoss

from model_surgery import build_model, describe


def load_data(path: str, limit: int | None = None):
    data = np.load(path)
    train = torch.from_numpy(data["train"]).float().div_(255.0)
    test = torch.from_numpy(data["test"]).float().div_(255.0)
    if limit:
        train = train[:limit]
        test = test[:min(limit, test.shape[0])]
    return train, test


@torch.no_grad()
def evaluate(model, test: torch.Tensor, device, batch: int = 64) -> dict:
    model.eval()
    bpps, psnrs, ssims = [], [], []
    for i in range(0, test.shape[0], batch):
        x = test[i:i + batch].to(device)
        out = model(x)
        n, _, h, w = x.shape
        bits = sum((-torch.log2(lk)).sum(dim=(1, 2, 3))
                   for lk in out["likelihoods"].values())
        bpps.extend((bits / (h * w)).tolist())
        x_hat = out["x_hat"].clamp(0, 1)
        mse = F.mse_loss(x, x_hat, reduction="none").mean(dim=(1, 2, 3))
        psnrs.extend((10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).tolist())
        ssims.extend(ssim(x, x_hat, data_range=1.0,
                          size_average=False).tolist())
    return {"bpp": float(np.mean(bpps)), "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims)), "n": int(test.shape[0])}


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = load_data(cfg["data_path"], cfg.get("limit"))
    model = build_model(
        quality=int(cfg.get("quality", 8)),
        act=str(cfg.get("act", "prelu")),
        factor_layers=tuple(cfg.get("factor_layers", (2, 4, 6))),
        mids={int(k): int(v) for k, v in cfg.get("mids", {}).items()},
        widths=tuple(cfg.get("widths", (192, 320))),
        pretrained=bool(cfg.get("pretrained", True)),
    ).to(device)

    criterion = RateDistortionLoss(lmbda=float(cfg.get("lmbda", 0.18)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    epochs = int(cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(cfg.get("lr", 1e-4)) * 0.05)

    out_dir = Path(cfg.get("out_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(cfg.get("batch", 32))
    steps_per_epoch = (train_set.shape[0] + batch - 1) // batch

    print(f"device={device} train={tuple(train_set.shape)} "
          f"test={tuple(test_set.shape)} steps/epoch={steps_per_epoch}")
    print("model:", describe(model))

    history = []
    best = {"psnr": -1e9}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_set.shape[0])
        running = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = order[step * batch:(step + 1) * batch]
            x = train_set[idx].to(device)
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-2])
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, x)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"  epoch {epoch:02d} step {step:4d}/{steps_per_epoch} "
                      f"loss={float(loss):.4f}", flush=True)
        scheduler.step()
        model.update(force=True)
        metrics = evaluate(model, test_set, device)
        metrics.update({"epoch": epoch, "train_loss": running / steps_per_epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch_s": round(time.time() - t0, 1)})
        history.append(metrics)
        print(f"epoch {epoch:02d} loss={metrics['train_loss']:.4f} "
              f"bpp={metrics['bpp']:.4f} psnr={metrics['psnr']:.3f} "
              f"ssim={metrics['ssim']:.4f} ({metrics['epoch_s']}s)", flush=True)

        ckpt = {"model": model.state_dict(), "cfg": cfg, "metrics": metrics}
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["psnr"] > best["psnr"]:
            best = metrics
            torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    result = {"best": best, "last": history[-1], "history": history,
              "cfg": cfg, "model": describe(model)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
