# Dependency Impact Report

Status: inspection only. No package installation has been approved or performed
by this report.

## Input Evidence

The workstation already has ROCm/HIP components installed, including
`rocm-runtime`, `rocm-hip`, `hipblas`, `hipfft`, `rocminfo`, and `rocm-opencl`.
The base ONNX Runtime packages are installed:

- `onnxruntime-1.22.2-2.fc44.x86_64`
- `onnxruntime-devel-1.22.2-2.fc44.x86_64`

Fedora offers, but they are not installed according to the provided package
snapshot:

- `onnxruntime-rocm`
- `onnxruntime-rocm-devel`
- `migraphx`
- `migraphx-devel`

## Packages To Install

Candidate only, not approved:

- `onnxruntime-rocm`
- `onnxruntime-rocm-devel`
- `migraphx`
- `migraphx-devel`

## Packages To Remove

Unknown until a correct dry run is captured with:

```bash
sudo dnf install onnxruntime-rocm onnxruntime-rocm-devel migraphx migraphx-devel --assumeno
```

If DNF proposes removal of protected packages, stop.

## Packages To Replace

Unknown until the dry run above is captured. Replacements of Mesa, Vulkan,
OpenCL loader, FFmpeg, OBS, GCC, Clang, or kernel packages are not acceptable.

## Packages To Upgrade

Unknown until dry run.

## Packages To Downgrade

Unknown until dry run.

## Files That Overlap

Unknown until dry run and package file inspection. Areas of concern:

- `libonnxruntime.so*`
- MIGraphX shared libraries
- HIP/HSA runtime libraries
- OpenCL ICD files

## Potential Impact On Unreal

High if the transaction changes ROCm compiler/runtime components, kernel
interfaces, Vulkan loader, Mesa/RADV, or global compiler/toolchain behavior.
The implementation must not modify Unreal Engine or its bundled toolchain.

## Potential Impact On OpenCL

High if the transaction replaces `OpenCL-ICD-Loader` or changes ICD ordering.
The existing OpenCL video-processing path is already working and must be
preserved.

## Potential Impact On Vulkan/RADV

High if DNF proposes Mesa or Vulkan loader changes. Such changes are forbidden
without explicit user approval.

## Potential Impact On OBS

Medium. OBS itself should not be touched, but V4L2/OpenCL/Vulkan regressions can
affect the current streaming workflow.

## Potential Impact On FFmpeg

Medium. FFmpeg must not be replaced or removed by the inference dependency
transaction.

## Rollback Command

No rollback command is safe to publish until the exact DNF transaction is known.
After an approved dry run, record the exact package set and use a targeted DNF
history rollback only if the transaction is isolated.

## Stop Conditions

Stop if the dry run proposes removing, replacing, or downgrading any of:

- `mesa-*`
- `vulkan-loader*`
- `OpenCL-ICD-Loader*`
- `opencl-headers`
- `ffmpeg*`
- `obs-studio*`
- `gcc*`
- `clang*`
- `kernel*`

Never use `--allowerasing` or `--skip-broken` for this work.
