# linux-media-pipeline

linux-media-pipeline is a Linux-first C++20 media pipeline for low-latency camera
ingest, frame processing, and virtual-camera output.

The primary path is:

```text
GoPro Hero11 UDP MPEG-TS -> FFmpeg decode -> lmp::Frame -> FilterPipeline -> V4L2
```

The V4L2 output is a normal Linux camera device, so it can be used by OBS, Google
Meet, Zoom, Chrome, or any other app that can read `/dev/video*`.

## Requirements

- Fedora 44
- GCC 16
- CMake 4.x
- Conan 2.x
- Ninja
- FFmpeg 8.x and development headers
- OpenCL loader, headers, and a working GPU ICD/runtime
- v4l2loopback
- OBS 31+ or another V4L2 consumer

macOS is supported as a development/validation environment for building, tests,
FFmpeg, OpenCL, and ONNX Runtime provider checks. V4L2 virtual-camera output is
Linux-only, so OBS/Meet end-to-end camera validation still runs on Fedora.

## Quick Start

Fedora:

```bash
./scripts/install-fedora-deps.sh
./scripts/stream.sh install-model all
./scripts/build.sh
./scripts/test.sh
./scripts/setup-loopback.sh 20 linux-media-pipeline
./scripts/stream.sh doctor /dev/video20 config/presets/ai-background-blur.yaml
```

macOS local validation:

```bash
./scripts/install-macos-deps.sh
./scripts/stream.sh install-model all
./scripts/build.sh
./scripts/test.sh
./build/dev/lmp --list-onnx-providers
```

The default `dev` preset is optimized with debug symbols because the normal
scripts are also used for live video. Use `cmake --preset debug` only when you
need an unoptimized debugger build.

To tune AI background blur locally on macOS, convert a captured frame to binary
PPM and run the same preset through the diagnostic path:

```bash
ffmpeg -y -i frame.png -frames:v 1 -pix_fmt rgb24 artifacts/frame.ppm
./build/dev/lmp --config config/presets/ai-background-blur-rocm.yaml \
  --diagnose-frame artifacts/frame.ppm \
  --diagnose-output artifacts/frame-diagnostics/rocm-preset
```

The diagnostic writes:

```text
artifacts/frame-diagnostics/rocm-preset/input.ppm
artifacts/frame-diagnostics/rocm-preset/output.ppm
artifacts/frame-diagnostics/rocm-preset/metadata.txt
```

Convert the output to PNG for quick inspection:

```bash
ffmpeg -y -i artifacts/frame-diagnostics/rocm-preset/output.ppm \
  artifacts/frame-diagnostics/rocm-preset/output.png
```

Compare ONNX segmentation providers on the same captured frame before using a
GPU provider live:

```bash
./build/dev/lmp --config config/presets/ai-background-blur-rocm.yaml \
  --segment-diagnostics artifacts/frame.ppm \
  --segment-output artifacts/segmentation-diagnostics/frame-001 \
  --segment-providers cpu,rocm \
  --segment-model assets/models/mediapipe-selfie.onnx,assets/models/pphumanseg.onnx
```

The diagnostic writes `report.json`, raw `*_mask.pgm` masks, and `*_overlay.ppm`
visual overlays. A usable GPU path should have a sane coverage value and a low
`cpu_mask_mae` compared with the CPU mask for the same model. If ROCm coverage
is near `0.0` or `1.0`, or `cpu_mask_mae` is large, keep that model/provider out
of the live preset. If `model_loaded` is `false`, fix that model first; fallback
masks are intentionally excluded from the comparison.

To capture one frame directly from the GoPro UDP feed on Fedora:

```bash
mkdir -p artifacts
ffmpeg -y -i udp://0.0.0.0:8554 -frames:v 1 -update 1 -pix_fmt rgb24 artifacts/frame.ppm
```

Validate the virtual camera with a generated pattern:

```bash
./scripts/stream.sh test-pattern
```

Select `/dev/video20` in OBS, Google Meet, Zoom, or Chrome.

## GoPro Hero11

Start the in-process FFmpeg pipeline:

```bash
./scripts/stream.sh gopro-udp
```

This reads `udp://0.0.0.0:8554` from `config/default.yaml`, decodes video with
FFmpeg, converts frames to `lmp::Frame`, applies the configured `FilterPipeline`,
and writes RGB frames to `/dev/video20`.

Run with a filter preset:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/clean.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/recording-background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-presenter.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur-performance.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-color.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-image.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-video.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/debug.yaml
```

For simultaneous horizontal and vertical OBS sources from one GoPro stream:

```bash
./scripts/stream.sh gopro-dual /dev/video20 /dev/video21
```

This decodes the camera once and writes two filtered outputs:
`config/presets/dual-horizontal.yaml` to `/dev/video20` and
`config/presets/dual-vertical.yaml` to `/dev/video21`.
The default vertical output reuses the horizontal AI/background result and then
applies a 9:16 smart crop with a safe-area dead zone and smooth motion, so the
expensive segmentation step runs once per input frame instead of once per
virtual camera.

The application prints active filters when streaming starts:

```text
filter_backend=requested:opencl filters=1 filters_active=[color_adjust]
filter_backend_active=opencl
```

If OpenCL is not available at runtime, `color_adjust` falls back to CPU and the
active backend line will say `filter_backend_active=cpu`.

Use `config/presets/ai-presenter.yaml` for tutorial/meeting framing: it crops
toward the detected person and applies light OpenCL color cleanup. Use
`config/presets/ai-background-blur.yaml` when you also want background blur; that
preset uses CPU ONNX person segmentation from `assets/models/pphumanseg.onnx`
and still uses the AMD GPU through OpenCL for blur, compositing, auto-framing,
and color cleanup. The ROCm preset remains available for diagnostics, but it is
not the recommended live path unless static-frame diagnostics show ROCm matches
CPU for the selected model.

For background replacement, put static images, GIFs, or videos under
`assets/backgrounds/` and set `background_path` in the matching preset. FFmpeg
does the media decode, so common formats such as PNG, JPG, GIF, MP4, and WebM
are accepted when your FFmpeg build supports them.

## USB Camera

USB bridging currently uses FFmpeg as an external adapter:

```bash
./scripts/stream.sh usb /dev/video0 /dev/video20 1280 720 30
```

Filters are enabled in YAML:

```yaml
filters:
  - type: color_adjust
    enabled: true
    backend: opencl
    brightness: 2
    contrast: 1.04
    saturation: 1.03
```

The project architecture keeps USB capture separate from output, so this can be
replaced by an in-process `UsbCameraSource` without changing the V4L2 consumer.

## Useful Commands

```bash
./build/dev/lmp --help
./build/dev/lmp --open-capture
./build/dev/lmp --check-output
./build/dev/lmp --stream-live
./build/dev/lmp --test-pattern
./scripts/stream.sh doctor /dev/video20 config/presets/ai-background-blur.yaml
```

## Configuration

Default runtime configuration lives in [config/default.yaml](config/default.yaml).
The important output settings are:

```yaml
output:
  type: v4l2
  device: /dev/video20
  pixel_format: RGB24
  width: 1280
  height: 720
  fps: 30
```

## Architecture

The code is organized around replaceable adapters:

- Capture: GoPro UDP now, USB/RTSP/SRT/WebRTC later.
- Decoder: FFmpeg in-process for the GoPro path.
- Frame: project-owned `lmp::frame::Frame`.
- Filters: independent `IVideoFilter` implementations loaded from YAML.
- GPU: OpenCL and Vulkan backend contracts.
- AI: optional ONNX Runtime segmentation with deterministic fallback.
- Output: V4L2 virtual camera.

## AI Segmentation

If ONNX Runtime headers and library are available at configure time, the build
enables ONNX Runtime automatically. ONNX Runtime is the inference engine; the
actual AI model is the `.onnx` file. Put a person-segmentation model at:

```text
assets/models/person-segmentation.onnx
```

The expected input is RGB float with one `3` channel axis and two spatial axes,
for example `HxWx3`, `3xHxW`, `1x3xHxW`, `1xHxWx3`, or the same layout with
leading singleton dimensions. The output can be a single mask or a common
two-channel segmentation tensor such as `HxWx1`, `1x2xHxW`, or `1xHxWx2`; when
two channels exist, the person channel is used. If ONNX Runtime or the model is
missing, the pipeline keeps running with the deterministic fallback.
If a model does not expose readable tensor shapes, set `input_shape` and
`output_shape` in the preset, for example `1x3x256x256` and `1x1x256x256`.
The AI blur preset also refines the mask before compositing: `mask_expand`
protects the person edge and `mask_feather` softens the transition into the
blurred background.

Run the real AI background blur preset with:

```bash
./scripts/stream.sh install-model
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

The terminal should show:

```text
LMP_HAS_ONNXRUNTIME=ON
background_blur_backend_active=opencl
background_blur_mask_active=onnx
onnx_runtime_available_providers=[CPUExecutionProvider]
onnx_runtime_provider_requested=cpu
onnx_runtime_provider_active=CPUExecutionProvider
onnx_runtime_provider_fallback=false
openvino_available_devices=[CPU]
openvino_device_requested=CPU
openvino_device_active=CPU
background_processing_backend=opencl
```

If it prints `background_blur_mask_active=onnx_unavailable_center`,
`background_blur_mask_active=onnx_unavailable_tracked_center`,
`background_blur_mask_active=onnx_init_error_center`,
`background_blur_mask_active=onnx_init_error_tracked_center`,
`background_blur_mask_active=onnx_error_center`, or
`background_blur_mask_active=onnx_error_tracked_center`, or
`background_blur_mask_active=center`, the runtime is not using a real model yet;
check that the model file exists, that Fedora installed `onnxruntime` and
`onnxruntime-devel`, and that the model input/output shape is compatible.
`./scripts/stream.sh install-model all` installs the preferred ONNX Runtime
models and any external `.data` weights file when required.

The `provider` value in the preset can be `auto`, `cpu`, `migraphx`, `rocm`, or
`openvino`. `auto` requests `MIGraphXExecutionProvider` only when the active
ONNX Runtime build enumerates it; otherwise it keeps inference on
`CPUExecutionProvider` while OpenCL handles the GPU blur. Fedora's ROCm package
may expose `ROCMExecutionProvider` instead of MIGraphX; use `provider: rocm`
explicitly only for diagnostics or after `--segment-diagnostics` proves that
ROCm produces a mask close to CPU for the same model. `openvino` is supported as
an explicit CPU inference path on this AMD workstation and must not be treated
as AMD GPU inference.

Inspect providers with:

```bash
./build/dev/lmp --list-onnx-providers
```

Before installing MIGraphX/ONNX Runtime ROCm packages, capture a workstation
snapshot and run the dry-run report:

```bash
./scripts/snapshot-workstation.sh
./scripts/dnf-migraphx-dry-run.sh
```

Do not continue if DNF proposes replacing Mesa, Vulkan, OpenCL ICD Loader,
FFmpeg, OBS, GCC, Clang, or kernel packages. The project reports what ONNX
Runtime providers are available and which one is active. If
`onnx_runtime_provider_fallback=true`, inference is not running on the requested
provider.

FFmpeg recoverable startup logs are hidden by default so the runtime status lines
stay readable.
To temporarily restore FFmpeg diagnostics:

```bash
LMP_FFMPEG_LOG_LEVEL=error ./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

For live tuning, enable periodic runtime stats:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
```

The stats line reports real pipeline FPS plus average and worst frame time. If
FPS drops, increase `inference_interval` in the preset so the ONNX model runs
less often. If the crop follows too slowly, lower `mask_smoothing`; if it jitters,
raise it slightly.
If ONNX is unavailable, `fallback_mask_mode: tracked_center` keeps a broader
center mask while still using lightweight tracking for the crop.

See [docs/architecture.md](docs/architecture.md), [docs/live-video.md](docs/live-video.md),
[docs/filters.md](docs/filters.md), [docs/obs.md](docs/obs.md),
[docs/amd-migraphx.md](docs/amd-migraphx.md), and
[docs/workstation-protection.md](docs/workstation-protection.md).
