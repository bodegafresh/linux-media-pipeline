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
- stage timing:
  `average_preprocess_ms`, `average_onnx_run_ms`, and
  `average_postprocess_ms`.

`gpu_execution_verified=true` requires all of:

- `MIGraphXExecutionProvider` selected by ONNX Runtime;
- or, explicitly, `ROCMExecutionProvider` selected by ONNX Runtime when testing
  Fedora's ROCm package;
- real inference success;
- no provider fallback.

A successful provider request is not enough by itself. If the active provider is
still `CPUExecutionProvider`, the verification must remain false even if the
OpenCL blur path is using the AMD GPU.

During live streaming, the startup diagnostics also include:

```text
onnx_preprocess_ms=...
onnx_inference_ms=...
onnx_postprocess_ms=...
segmentation_mask_coverage_raw=...
segmentation_mask_coverage_refined=...
```

These are the last completed ONNX segmentation timings. They are intentionally
separate from OpenCL background processing and global `runtime_stats`.
