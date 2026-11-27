"""Modal smoke test: GPU + torch + compressai availability."""
import modal

app = modal.App("aidge-gpu-check")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install("torch", "numpy<2", "compressai==1.2.8")
)


@app.function(image=image, gpu="T4", timeout=900)
def check() -> str:
    import sys
    import torch
    import compressai
    import numpy as np

    return (
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} "
        f"compressai={compressai.__version__} numpy={np.__version__}"
    )


@app.local_entrypoint()
def main() -> None:
    print(check.remote())
