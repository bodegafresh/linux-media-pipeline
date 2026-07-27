# ONNX Model Compatibility

The segmentation model must be validated before using it for live streaming.

Current runtime checks:

- model file exists;
- model input and output names are readable;
- input rank and shape are readable or supplied by config;
- RGB channel dimension is present;
- spatial dimensions are sane;
- output tensor rank can be 2, 3, or 4;
- binary or multi-channel probability masks are accepted;
- CPU fallback remains available.

Recommended model config:

```yaml
model_path: assets/models/person-segmentation.onnx
input_shape: 1x3x256x256
output_shape: 1x1x256x256
provider: migraphx
allow_provider_fallback: true
```

The startup log prints `onnx_runtime_model=...` with the detected input/output
names, tensor ranks, dimensions, and inferred layout. If that line does not
match the model documentation, fix the YAML overrides before streaming.
