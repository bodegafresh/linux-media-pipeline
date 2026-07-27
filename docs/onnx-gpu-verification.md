# ONNX GPU Verification

The conservative verification command is:

```bash
./scripts/verify-onnx-gpu.sh migraphx config/presets/ai-background-blur-migraphx.yaml
```

It writes:

```text
artifacts/onnx-gpu-verification.json
```

Current verification performs:

- real model session creation;
- warm-up runs;
- measured inference runs;
- output mask sanity checks;
- provider/fallback reporting.
- ROCm activity snapshots from the wrapper scripts when `rocm-smi` is present.

`gpu_execution_verified=true` requires all of:

- `MIGraphXExecutionProvider` selected by ONNX Runtime;
- real inference success;
- no provider fallback.

A successful provider request is not enough by itself. If the active provider is
still `CPUExecutionProvider`, the verification must remain false even if the
OpenCL blur path is using the AMD GPU.
