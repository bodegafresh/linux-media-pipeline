# Fedora AMD Inference Setup

This workstation protection policy applies before any MIGraphX/ONNX Runtime ROCm
package change.

Safe diagnostics:

```bash
./scripts/snapshot-workstation.sh
./scripts/check-amd-inference-stack.sh
./scripts/check-onnx-providers.sh
./scripts/check-runtime-linkage.sh
```

Dry run only:

```bash
./scripts/dnf-migraphx-dry-run.sh
```

Stop if DNF proposes removing, replacing, or downgrading Mesa, Vulkan loader,
OpenCL loader, FFmpeg, OBS, GCC, Clang, kernel packages, or anything required by
the current graphics stack.

Never use `--allowerasing` or `--skip-broken` for this work.

The initial valid outcome may be that Fedora packages are present but the active
ONNX Runtime build still enumerates only `CPUExecutionProvider`.

## Approved Candidate Transaction

The observed Fedora dry run proposed only installing:

- `onnxruntime-rocm`
- `onnxruntime-rocm-devel`
- `migraphx`
- `migraphx-devel`

It proposed no removals. Disk impact was about 3 GiB installed.

After reviewing `artifacts/dnf-migraphx-dry-run.txt`, the manual install command
is:

```bash
sudo dnf install onnxruntime-rocm onnxruntime-rocm-devel migraphx migraphx-devel
```

Do not add `--allowerasing` or `--skip-broken`.

Then rebuild and verify:

```bash
./scripts/build.sh
./build/dev/lmp --list-onnx-providers
./scripts/benchmark-onnx-providers.sh config/presets/ai-background-blur-migraphx.yaml
```

Install the segmentation candidates used for Fedora validation:

```bash
./scripts/stream.sh install-model all
```

Before enabling a GPU inference provider in live video, compare CPU and ROCm on
captured GoPro frames:

```bash
mkdir -p artifacts
ffmpeg -y -i udp://0.0.0.0:8554 -frames:v 1 -update 1 -pix_fmt rgb24 artifacts/frame.ppm

./build/dev/lmp --config config/presets/ai-background-blur-rocm.yaml \
  --segment-diagnostics artifacts/frame.ppm \
  --segment-output artifacts/segmentation-diagnostics/frame-001 \
  --segment-providers cpu,rocm \
  --segment-model assets/models/mediapipe-selfie.onnx,assets/models/pphumanseg.onnx
```

Inspect `artifacts/segmentation-diagnostics/frame-001/report.json` and the
`*_overlay.ppm` outputs. A usable ROCm path must produce a mask that resembles
the CPU mask for the same model. Treat `coverage` close to `0.0` or `1.0`, or a
large `cpu_mask_mae`, as a failed model/provider combination. If `model_loaded`
is `false`, the model did not run and must not be considered for live video.

The provider list must show `MIGraphXExecutionProvider` before the MIGraphX live
preset can use AMD GPU inference. If Fedora shows `ROCMExecutionProvider`
instead, use:

```bash
./scripts/verify-onnx-gpu.sh rocm config/presets/ai-background-blur-rocm.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur-rocm.yaml --stats-every 5
```
