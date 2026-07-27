# Milestone 1 Report: Audit And Diagnostics

## Completed

- Added `docs/onnx-gpu-audit.md`.
- Added `linux-media-pipeline --list-onnx-providers`.
- Added CMake status flags for:
  - `LMP_HAS_ONNXRUNTIME_MIGRAPHX`
  - `LMP_HAS_MIGRAPHX`
  - `LMP_HAS_ROCM_RUNTIME`
- Added non-invasive scripts:
  - `scripts/snapshot-workstation.sh`
  - `scripts/check-amd-inference-stack.sh`
  - `scripts/check-runtime-linkage.sh`
  - `scripts/check-onnx-providers.sh`
  - `scripts/check-workstation-regression.sh`
- Added `artifacts/dependency-impact-report.md`.

## Not Performed

- No DNF install was run.
- No package removal, replacement, distro sync, swap, or `--allowerasing` was
  run.
- MIGraphX provider activation is not implemented yet.
- GPU execution verification is not implemented yet.

## Required Fedora Commands

```bash
./scripts/snapshot-workstation.sh
./scripts/check-onnx-providers.sh
./scripts/check-amd-inference-stack.sh
./scripts/check-runtime-linkage.sh
```

Before installing MIGraphX/ONNX Runtime ROCm packages, capture:

```bash
sudo dnf install onnxruntime-rocm onnxruntime-rocm-devel migraphx migraphx-devel --assumeno
```

Do not proceed if DNF proposes changes to protected graphics, compiler, FFmpeg,
OBS, OpenCL loader, Mesa, Vulkan, or kernel packages.
