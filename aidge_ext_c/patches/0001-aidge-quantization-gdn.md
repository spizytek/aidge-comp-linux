Proper AIDGE GDN operator — quantization patch
================================================

The `aidge_quantization` PTQ strategy table is a hardcoded `std::set`.  A
`GDN` node currently fails with:

    makeStrategy(): no strategy registered for node type '<name>[GDN]'

`not_quantize_node(gdn)` is not enough: `PTQ::preProcessGraphView` calls
`makeStrategy()` for every non-producer *before* it looks at the node policy.

Patch (AIDGE source, `aidge_quantization/include/aidge/quantization/PTQ/PTQTypes.hpp`):

```diff
 static const std::set<std::string> notQuantizedNodeTypes({"Sigmoid",
                                                           "Tanh",
                                                           "Div",
                                                           "Softmax",
                                                           "Shape",
                                                           "Cast",
                                                           "Gather",
                                                           "HardSwish",
                                                           "Unsqueeze",
                                                           "Split",
-                                                          "HardSigmoid"});
+                                                          "HardSigmoid",
+                                                          "GDN"});
```

Why this is the right place: GDN is not quantization-friendly as a whole
(rsqrt/div), but the *convs around it are*.  The quantizer should insert
Q/DQ boundaries around GDN and leave the operator float, exactly as it does
for Sigmoid/Tanh/Softmax.  This is the same policy needed for a future native
`GDN_Op`; the only additional upstream work for a fully native operator is a
C++ `GDNOp`/`GDNImpl` and ONNX converters instead of the `GenericOperator`.

Build (Windows, VS 2022 + CMake):

  # from a Developer PowerShell for VS 2022
  .\setup.ps1 -Modules core,backend_cpu,onnx,quantization -BuildType Release

Then run PTQ with the proper GDN graph:

  from gdn_operator import GDNImpl     # registers ["cpu","GDN"]
  # ... load analysis/encoder_gdn_unit_64.onnx ...
  ptq = aidge_quantization.PTQ(...)
  qgraph = ptq.process(calibration_tensors)

For AIDGE 0.10.1 the equivalent call is:

  aidge_quantization.auto_assign_node_precision(graph, int8, int8, int32)
  aidge_quantization.quantize_network(network=graph,
                                      calibration_set=calibration_tensors,
                                      equalize=False, ...)

The `GDN` CPU implementation used for calibration is in
`aidge_ext_c/gdn_operator.py`; the C export backend is registered there too.
