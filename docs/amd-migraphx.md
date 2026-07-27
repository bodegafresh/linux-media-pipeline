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

Fedora 44's `onnxruntime-rocm` package may instead enumerate
`ROCMExecutionProvider` and no `MIGraphXExecutionProvider`. That is a real AMD
GPU ONNX Runtime provider, but it is not the desired MIGraphX path. Use it
explicitly with `provider: rocm` when you want to validate the currently
available Fedora GPU path.

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

## Installed But Not Enumerated

If `migraphx` is installed but `--list-onnx-providers` still shows only:

```text
CPUExecutionProvider
```

then the process is still loading the CPU ONNX Runtime library. Rebuild from a
clean build directory so `scripts/build.sh` can pass the ROCm ONNX Runtime root
detected from the RPM database:

```bash
rm -rf build/dev
./scripts/build.sh
./scripts/check-runtime-linkage.sh
./build/dev/lmp --list-onnx-providers
```

The CMake configure log should print a non-empty `ONNXRUNTIME_ROOT`,
`ONNXRUNTIME_LIBRARY`, and ideally `ONNXRUNTIME_MIGRAPHX_PROVIDER_LIBRARY`.
If the provider still does not enumerate, inspect:

```text
artifacts/runtime-linkage/ldd.txt
artifacts/runtime-linkage/rpm-files.txt
artifacts/runtime-linkage/duplicate-runtimes.txt
```

If the provider list shows:

```text
ROCMExecutionProvider
CPUExecutionProvider
```

then use the Fedora ROCm preset:

```bash
./scripts/verify-onnx-gpu.sh rocm config/presets/ai-background-blur-rocm.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur-rocm.yaml --stats-every 5
```

For quality validation, do not rely on provider activation alone. First compare
candidate models on static frames:

```bash
./scripts/stream.sh install-model all
./build/dev/lmp --config config/presets/ai-background-blur-rocm.yaml \
  --segment-diagnostics artifacts/frame.ppm \
  --segment-output artifacts/segmentation-diagnostics/frame-001 \
  --segment-providers cpu,rocm \
  --segment-model assets/models/mediapipe-selfie.onnx,assets/models/pphumanseg.onnx
```

Use the generated `report.json` and `*_overlay.ppm` files to choose the model
where ROCm matches CPU closely enough for live video. A GPU provider that is
fast but produces saturated or empty masks is not considered production-ready.
