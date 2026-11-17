"""Export the Aidge MNIST demo to C with the ACETONE backend."""

from pathlib import Path
import os
import subprocess

import numpy as np

import aidge_backend_cpu  # noqa: F401  (registers the CPU backend)
import aidge_core
import aidge_export_acetone
import aidge_onnx

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output_acetone_mnist"
ASSET_URL = (
    "https://gitlab.eclipse.org/eclipse/aidge/aidge/-/raw/main/"
    "examples/tutorials/101_first_step/"
)
INPUT_SHAPE = [1, 1, 28, 28]


def asset(name: str) -> Path:
    """Return a demo asset, downloading it into the output folder if missing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    aidge_core.utils.download_file(file_path=str(path), file_url=ASSET_URL + name)
    return path


def main() -> None:
    model = aidge_onnx.load_onnx(str(asset("MLP_MNIST.onnx")))

    digit = np.load(asset("input_digit.npy")).astype(np.float32)
    tensor = aidge_core.Tensor(digit)
    tensor.to_backend("cpu")
    tensor.set_data_format(aidge_core.dformat.nchw)
    tensor.to_dtype(aidge_core.dtype.float32)

    model.compile(
        backend=aidge_export_acetone.ExportLibAcetone._name,
        datatype=aidge_core.dtype.float32,
        dims=[INPUT_SHAPE],
    )

    scheduler = aidge_core.SequentialScheduler(model)
    scheduler.generate_scheduling()

    export_dir = OUTPUT_DIR / "export"
    aidge_export_acetone.export.scheduler_export(
        scheduler,
        str(export_dir),
        aidge_export_acetone.ExportLibAcetone,
        memory_manager=aidge_core.mem_info.compute_default_mem_info,
    )
    aidge_export_acetone.export.generate_main_display_output(
        str(export_dir), model, [tensor]
    )

    subprocess.run(
        ["make", f"-j{os.cpu_count() or 1}"], cwd=export_dir, check=True
    )
    prediction = subprocess.run(
        [str(export_dir / "bin" / "run_export")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    print(f"C code: {export_dir}")
    print(f"Model output: {prediction.strip()}")


if __name__ == "__main__":
    main()
