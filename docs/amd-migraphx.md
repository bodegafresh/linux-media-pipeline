# AMD MIGraphX Inference

`linux-media-pipeline` treats ONNX inference and OpenCL video processing as
separate backends.

- ONNX inference provider: `MIGraphXExecutionProvider`, `OpenVINOExecutionProvider`,
  or `CPUExecutionProvider`.
- Background processing backend: OpenCL.
- OpenCL device: selected independently from ONNX Runtime.

For the Fedora AMD workstation, the desired ONNX GPU path is:

```text
ONNX Runtime -> MIGraphXExecutionProvider -> AMD Radeon RX 6750 XT / gfx1031
```

The application never marks ONNX inference as GPU accelerated unless ONNX
Runtime both enumerates and activates `MIGraphXExecutionProvider`.

## Safe Check Flow

Run these commands before changing packages:

```bash
./scripts/snapshot-workstation.sh
./scripts/check-amd-inference-stack.sh
./scripts/dnf-migraphx-dry-run.sh
```

Review:

```text
artifacts/workstation-baseline/
artifacts/dnf-migraphx-dry-run.txt
```

Stop if DNF proposes removing or replacing Mesa, Vulkan, OpenCL ICD Loader,
FFmpeg, OBS, GCC, Clang, or kernel packages.

## Runtime Verification

After installing a MIGraphX-capable ONNX Runtime build:

```bash
./scripts/check-onnx-providers.sh
./scripts/verify-onnx-gpu.sh migraphx config/presets/ai-background-blur-migraphx.yaml
./scripts/benchmark-onnx-providers.sh config/presets/ai-background-blur-migraphx.yaml
```

Expected successful provider activation:

```text
onnx_runtime_provider_requested=migraphx
onnx_runtime_provider_active=MIGraphXExecutionProvider
gpu_execution_verified=true
```

The verification scripts also write ROCm activity snapshots:

```text
artifacts/onnx-gpu-activity.txt
artifacts/onnx-migraphx-gpu-activity.txt
```

Treat `gpu_execution_verified=true` as valid only together with
`onnx_runtime_provider_active=MIGraphXExecutionProvider` and no provider
fallback.
