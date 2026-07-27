# ONNX Provider Selection

ONNX provider availability is determined only by the active ONNX Runtime build.
Package names are useful context, but they are not proof that the process can
select a provider.

Use:

```bash
./build/dev/lmp --list-onnx-providers
```

Policy:

- `provider: cpu` selects `CPUExecutionProvider`.
- `provider: migraphx` requests `MIGraphXExecutionProvider`.
- `provider: auto` requests MIGraphX only when ONNX Runtime enumerates it;
  otherwise it remains on CPU.
- `provider: rocm` requests `ROCMExecutionProvider` explicitly when Fedora's
  ROCm ONNX Runtime build exposes it. This is not the same as MIGraphX.
- `provider: openvino` remains explicit and must not be presented as AMD GPU
  inference.

The live diagnostics must be read independently:

```text
segmentation_inference_backend=...
segmentation_inference_device=...
background_processing_backend=opencl
background_processing_device=...
```

An active OpenCL backend proves only that background processing is using OpenCL.
It does not prove ONNX inference is using the AMD GPU.
