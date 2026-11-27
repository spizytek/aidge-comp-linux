# aidge_export_c

Standalone **C** export backend for Aidge. It is the C counterpart of
`aidge_export_cpp` and generates a buildable C11 project from an Aidge graph.

## Install

```bash
# from this folder
pip install -e .

# or from the Aidge repo root
pip install -e aidge/aidge_export_c
```

Requires `aidge_core`, `aidge_onnx`, `aidge_backend_cpu`, `numpy`, `jinja2`,
and `aidge_quantization` for int8.

## Usage

Importing the package registers the `export_c` backend and the `host_c`
target. The import is required before calling `export()`.

```python
import aidge_export_c
from aidge_core.export_utils import export
from aidge_core.export_utils.config import ExportConfig

artifact = export(
    graph=model,
    config=ExportConfig(
        target="host_c",
        export_folder="my_export_c",
        dtype="float32",   # or "int8"
        dformat="nhwc",
        compile=False,
    ),
    input_tensors=[input_tensor],
)
```

For int8, set `dtype="int8"` and pass calibration tensors. The export
quantizes the graph and inputs and emits int8 kernels and rescaling
parameters.

Build the generated project with CMake or with `compile_export`:

```python
from aidge_core.export_utils import compile_export
compile_export("my_export_c")
```

```bash
cd my_export_c
cmake -S . -B build
cmake --build build --config Debug
./build/run_export                 # Linux/macOS
.\build\Debug\run_export.exe       # Windows
```

## Generated API

```c
void model_forward(const float *input, float **output);
/* int8_t* dtype when exporting int8 */
```

The project contains `main.c`, `forward.c/h`, layer configs, parameter
arrays, C kernels, and `CMakeLists.txt`.

## Support

- `float32` and `int8`
- Target: `host_c`, standalone application
- Operators: `Producer`, `FC`, `FCAct`, `QFC`, `ReLU`, `Quantizer`,
  `Conv2D`, `ConvAct`, `QConv`, `PadConv`, `PadConvAct`, `ConvDepthWise2D`,
  `ConvDwAct`, `QConvDw`, `PadConvDw`, `PadConvDwAct`, `ConvTranspose2D`
  and act/pad variants, `MaxPooling2D`, `AvgPooling2D` and pad/act variants,
  `Add`, `Sub`, `Mul`, `Div` and act/quantized variants, `Reshape`,
  `Transpose`
- Validated: MLP-MNIST, LeNet-MNIST, DS-CNN, ResNet8 (float32 and int8);
  `ConvTranspose2D` on a synthetic float32 graph
- Not yet supported: int8/1D/3D transposed convolution, benchmark/validation
  applications, non-host targets

Unsupported operators raise an explicit error rather than generating wrong
code.

## Examples

From the Aidge repo root:

```bash
python aidge/aidge_export_c/examples/mlp_mnist.py
python aidge/aidge_export_c/examples/mlp_mnist_int8.py
python aidge/aidge_export_c/examples/lenet.py
python aidge/aidge_export_c/examples/ds_cnn.py
python aidge/aidge_export_c/examples/resnet8.py
python aidge/aidge_export_c/examples/convtranspose2d.py
```

## Notes

- Aidge version compatibility matters: the package uses Aidge export internals
  (`ExportLib`, `ExportNodeCpp`, target strategies, memory planning).
- The library is additive; it does not modify `aidge_core` or
  `aidge_export_cpp`.
- Kernels are generic C loops; no vectorization or specialized HLS/ISA kernels.
