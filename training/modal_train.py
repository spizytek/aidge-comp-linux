"""Fine-tune the hardware-friendly Ballé encoder on Modal (or locally).

Smoke test:
    modal run training/modal_train.py::smoke

Full run:
    modal run training/modal_train.py --name prelu_3x3 --epochs 30
"""

from __future__ import annotations

import json

import modal

app = modal.App("aidge-balle-opt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install(
        "torch",
        "numpy<2",
        "compressai==1.2.8",
        "pytorch-msssim",
        "Pillow",
    )
    .add_local_file("training/model_surgery.py", "/root/model_surgery.py")
    .add_local_file("training/train_core.py", "/root/train_core.py")
)

volume = modal.Volume.from_name("aidge-data", create_if_missing=True)


def _cfg(**overrides) -> dict:
    cfg = {
        "name": "prelu_3x3",
        "quality": 8,
        "act": "prelu",
        "factor_layers": [2, 4, 6],
        "mids": {"2": 192, "4": 192, "6": 256},
        "widths": [192, 320],
        "pretrained": True,
        "lmbda": 0.18,
        "lr": 1e-4,
        "batch": 32,
        "epochs": 30,
        "limit": 0,
    }
    cfg.update(overrides)
    return cfg


@app.function(image=image, gpu="T4", timeout=7200,
              volumes={"/data": volume})
def train_remote(cfg: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from train_core import train

    cfg = dict(cfg)
    cfg["data_path"] = "/data/dataset_64_uint8.npz"
    cfg["out_dir"] = f"/data/runs/{cfg['name']}"
    result = train(cfg)
    volume.commit()
    return result


@app.local_entrypoint()
def main(name: str = "prelu_3x3", epochs: int = 30, batch: int = 32,
         lr: float = 1e-4, lmbda: float = 0.18, act: str = "prelu",
         limit: int = 0, quality: int = 8, pretrained: bool = True,
         widths: str = "192,320", mids: str = "2:192,4:192,6:256",
         factor: str = "2,4,6") -> None:
    cfg = _cfg(
        name=name, epochs=epochs, batch=batch, lr=lr, lmbda=lmbda, act=act,
        limit=limit or None, quality=quality, pretrained=pretrained,
        widths=[int(v) for v in widths.split(",")],
        mids={k: int(v) for k, v in
              (kv.split(":") for kv in mids.split(",") if kv)},
        factor_layers=[int(v) for v in factor.split(",") if v],
    )
    result = train_remote.remote(cfg)
    print("BEST:", json.dumps(result["best"], indent=2))
    print("LAST:", json.dumps(result["last"], indent=2))


@app.local_entrypoint()
def smoke() -> None:
    cfg = _cfg(name="smoke", epochs=1, batch=8, limit=128)
    result = train_remote.remote(cfg)
    print("SMOKE:", json.dumps(result["last"], indent=2))
