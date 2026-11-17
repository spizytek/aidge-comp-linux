import onnx
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "output_acetone"/"model"/"encoder_static.onnx"


m = onnx.load(DEFAULT_OUTPUT_DIR)
print("Opset:", m.opset_import[0].version)
print("Operators used:", sorted({n.op_type for n in m.graph.node}))