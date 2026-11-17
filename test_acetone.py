"""Export the bmshj2018 factorized encoder with the ACETONE backend.

The script:
  1. Prepares an ACETONE-friendly, static-shape copy of the ONNX model.
  2. Loads one image from DataSet/ as the example input.
  3. Runs the Aidge -> ACETONE export to generate C source code.
  4. Builds and runs the generated C code, and compares it with ONNX Runtime.

Everything the script produces is written under ``output_acetone/`` (inside this
workspace) so the source tree stays clean.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnx.utils import extract_model
from PIL import Image

import aidge_backend_cpu  # noqa: F401  (registers the CPU backend)
import aidge_core
import aidge_export_acetone
import aidge_onnx

# --------------------------------------------------------------------------- #
# Paths and model configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Model"
DATA_DIR = ROOT / "DataSet"
DEFAULT_OUTPUT_DIR = ROOT / "output_acetone"

MODEL_PATH = MODEL_DIR / "bmshj2018_factorized_encoder.onnx"

# The ONNX model exposes three outputs: ``x_out`` (the encoder latent) plus
# ``indexes``/``medians`` (entropy-coding helpers built from ops ACETONE does not
# support). Only the encoder itself is exported here.
ENCODER_OUTPUT = "x_out"

# Images in DataSet are 64x64. The encoder is fully convolutional, but a fixed
# size is required to resolve its data-dependent Reshape shapes.
INPUT_SHAPE = (1, 3, 64, 64)


# --------------------------------------------------------------------------- #
# Model preparation
# --------------------------------------------------------------------------- #


def _prune_unused_nodes(graph: onnx.GraphProto) -> None:
    """Drop nodes and initializers that no longer contribute to graph outputs."""
    nodes = list(graph.node)
    producers = {
        output: index for index, node in enumerate(nodes) for output in node.output
    }
    needed: set[int] = set()
    stack = [output.name for output in graph.output]

    while stack:
        name = stack.pop()
        index = producers.get(name)
        if index is None or index in needed:
            continue
        needed.add(index)
        stack.extend(nodes[index].input)

    kept_nodes = [node for index, node in enumerate(nodes) if index in needed]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    used = {name for node in graph.node for name in node.input}
    kept_initializers = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)


def _freeze_reshape_shapes(model: onnx.ModelProto, input_shape: tuple[int, ...]) -> None:
    """Replace data-dependent Reshape shape inputs with constants.

    The encoder builds its Reshape target shapes at runtime from ``Shape``/
    ``Gather``/``Concat``. For a fixed input size those values are constant, so
    they can be evaluated once with ONNX Runtime and stored as initializers.
    Aidge can then propagate dimensions through the Reshape.
    """
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    reshape_nodes = [
        node
        for node in graph.node
        if node.op_type == "Reshape" and node.input[1] not in initializer_names
    ]
    if not reshape_nodes:
        return

    shape_inputs = [node.input[1] for node in reshape_nodes]

    # Add the shape tensors as temporary outputs so ONNX Runtime returns them.
    augmented = onnx.ModelProto()
    augmented.CopyFrom(model)
    existing_outputs = {output.name for output in augmented.graph.output}
    for name in shape_inputs:
        if name not in existing_outputs:
            augmented.graph.output.append(helper.make_empty_tensor_value_info(name))

    session = ort.InferenceSession(
        augmented.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dummy = np.zeros(input_shape, dtype=np.float32)
    values = session.run(shape_inputs, {input_name: dummy})

    for index, (node, value) in enumerate(zip(reshape_nodes, values, strict=True)):
        constant_name = f"static_reshape_shape_{index}"
        node.input[1] = constant_name
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, dtype=np.int64), constant_name)
        )


def prepare_encoder_model(model_path: Path, input_shape: tuple[int, ...], work_dir: Path) -> Path:
    """Create a trimmed, static-shape ONNX copy of the encoder."""
    work_dir.mkdir(parents=True, exist_ok=True)

    source = onnx.load(str(model_path))
    output_names = [output.name for output in source.graph.output]
    encoder_output = ENCODER_OUTPUT if ENCODER_OUTPUT in output_names else output_names[0]

    trimmed_path = work_dir / "_encoder_trimmed.onnx"
    extract_model(
        str(model_path),
        str(trimmed_path),
        input_names=[source.graph.input[0].name],
        output_names=[encoder_output],
    )

    model = onnx.load(str(trimmed_path))
    _freeze_reshape_shapes(model, input_shape)
    _prune_unused_nodes(model.graph)
    onnx.checker.check_model(model)

    static_path = work_dir / "encoder_static.onnx"
    onnx.save(model, str(static_path))
    trimmed_path.unlink(missing_ok=True)
    return static_path


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def pick_image(split: str, image: Path | None) -> Path:
    """Return the requested image, or the first one from the chosen split."""
    if image is not None:
        path = image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    split_dir = DATA_DIR / split
    images = sorted(split_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found in {split_dir}")
    return images[0]


def load_image(path: Path, input_shape: tuple[int, ...]) -> tuple[np.ndarray, aidge_core.Tensor]:
    """Load an image as a float32 NCHW tensor in [0, 1]."""
    image = Image.open(path).convert("RGB")
    expected = (input_shape[3], input_shape[2])  # (width, height)
    if image.size != expected:
        image = image.resize(expected)

    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None]
    array = array / 255.0

    tensor = aidge_core.Tensor(array)
    tensor.to_backend("cpu")
    tensor.set_data_format(aidge_core.dformat.nchw)
    tensor.to_dtype(aidge_core.dtype.float32)
    return array, tensor


# --------------------------------------------------------------------------- #
# Aidge / ACETONE export
# --------------------------------------------------------------------------- #


def export_with_acetone(
    static_model_path: Path,
    input_tensor: aidge_core.Tensor,
    export_dir: Path,
    input_shape: tuple[int, ...],
) -> aidge_core.GraphView:
    """Run the Aidge graph through the ACETONE backend and generate C code."""
    model = aidge_onnx.load_onnx(str(static_model_path))

    # ACETONE has no Identity exporter, and the identity nodes are no-ops.
    aidge_core.remove_identity(model)

    # Prepare constant folding on the CPU backend. This evaluates the constant
    # gamma/beta re-parameterisation branches, including the Reshape nodes.
    model.set_backend("cpu")
    model.set_datatype(aidge_core.dtype.float32)
    model.forward_dims(dims=[list(input_shape)], allow_data_dependency=True)
    aidge_core.constant_folding(model, constant_shape=True)

    model.compile(
        backend=aidge_export_acetone.ExportLibAcetone._name,
        datatype=aidge_core.dtype.float32,
        dims=[list(input_shape)],
    )

    scheduler = aidge_core.SequentialScheduler(model)
    scheduler.generate_scheduling()

    if export_dir.is_dir():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    aidge_export_acetone.export.scheduler_export(
        scheduler,
        str(export_dir),
        aidge_export_acetone.ExportLibAcetone,
        memory_manager=aidge_core.mem_info.compute_default_mem_info,
    )
    aidge_export_acetone.export.generate_main_display_output(
        str(export_dir), model, [input_tensor]
    )

    return model


# --------------------------------------------------------------------------- #
# Build, run and verify
# --------------------------------------------------------------------------- #


def build_and_run(export_dir: Path, jobs: int) -> np.ndarray:
    """Compile the generated C project and return the program's outputs."""
    binary = export_dir / "bin" / "run_export"
    build = subprocess.run(
        ["make", f"-j{jobs}", "build"],
        cwd=export_dir,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stdout)
        sys.stderr.write(build.stderr)
        raise RuntimeError("Compilation of the generated C code failed")

    run = subprocess.run(
        [str(binary)],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    if not run.stdout.strip():
        raise RuntimeError("The generated program produced no output")

    return np.array(run.stdout.split(), dtype=np.float32)


def reference_output(static_model_path: Path, array: np.ndarray) -> np.ndarray:
    """Evaluate the static ONNX model with ONNX Runtime."""
    session = ort.InferenceSession(
        str(static_model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: array})[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the ACETONE export backend on the factorized encoder."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to pick an example image from (default: test).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Explicit image to use instead of the first one in the split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for every generated file (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel build jobs (default: all CPUs).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Only generate the C export; do not compile or run it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "model"
    export_dir = output_dir / "export"
    jobs = args.jobs or (os.cpu_count() or 1)

    print(f"Workspace : {ROOT}")
    print(f"Output    : {output_dir}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image_path = pick_image(args.split, args.image)
    print(f"Model     : {MODEL_PATH.name}")
    print(f"Image     : {image_path.relative_to(ROOT)}")

    print("\n[1/4] Preparing static-shape encoder model...")
    static_model = prepare_encoder_model(MODEL_PATH, INPUT_SHAPE, model_dir)
    print(f"      -> {static_model.relative_to(output_dir)}")

    print("[2/4] Loading image and exporting with ACETONE...")
    array, input_tensor = load_image(image_path, INPUT_SHAPE)
    export_with_acetone(static_model, input_tensor, export_dir, INPUT_SHAPE)
    print(f"      -> {export_dir.relative_to(output_dir)}")

    if args.no_build:
        print("\n[3/4] Skipped build (--no-build).")
        print("[4/4] Skipped verification.")
        return

    print("[3/4] Building generated C code...")
    c_output = build_and_run(export_dir, jobs)
    print(f"      -> {export_dir.relative_to(output_dir)}/bin/run_export")

    print("[4/4] Comparing against ONNX Runtime...")
    reference = reference_output(static_model, array).reshape(-1)
    if c_output.size != reference.size:
        raise RuntimeError(
            f"Output size mismatch: C={c_output.size}, ONNX Runtime={reference.size}"
        )

    np.save(output_dir / "c_output.npy", c_output)
    np.save(output_dir / "onnxruntime_output.npy", reference)

    absolute_error = np.abs(c_output - reference)
    max_abs = float(absolute_error.max())
    reference_scale = float(np.abs(reference).max()) or 1.0
    normalized_error = max_abs / reference_scale

    summary = (
        f"model: {MODEL_PATH}\n"
        f"image: {image_path}\n"
        f"input shape: {INPUT_SHAPE}\n"
        f"outputs: {c_output.size}\n"
        f"max absolute error: {max_abs:.3e}\n"
        f"max error / max |reference|: {normalized_error:.3e}\n"
    )
    (output_dir / "results.txt").write_text(summary)

    print(summary, end="")
    if not np.allclose(c_output, reference, rtol=1e-4, atol=1e-3):
        raise SystemExit("ACETONE output does not match ONNX Runtime")
    print("Success: ACETONE export matches the reference.")


if __name__ == "__main__":
    main()
