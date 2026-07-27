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

## Quick Start

```bash
./scripts/install-fedora-deps.sh
./scripts/build.sh
./scripts/test.sh
./scripts/setup-loopback.sh 20 linux-media-pipeline
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
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-presenter.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/debug.yaml
```

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
preset uses an ONNX person-segmentation mask when
`assets/models/person-segmentation.onnx` exists, fuses blur, auto-framing, and
color cleanup in the OpenCL blur filter, and falls back to a centered mask if the
model is missing.

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

The expected input is RGB float in `1x3xHxW` format. The output can be a single
mask or a common two-channel segmentation tensor such as `1x2xHxW` or `1xHxWx2`;
when two channels exist, the person channel is used. If ONNX Runtime or the
model is missing, the pipeline keeps running with the deterministic fallback.

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
```

If it prints `background_blur_mask_active=onnx_unavailable_center`,
`background_blur_mask_active=onnx_error_center`, or
`background_blur_mask_active=center`, the runtime is not using a real model yet;
check that the model file exists, that Fedora installed `onnxruntime` and
`onnxruntime-devel`, and that the model input/output shape is compatible.

FFmpeg warnings are hidden by default so the runtime status lines stay readable.
To temporarily restore FFmpeg diagnostics:

```bash
LMP_FFMPEG_LOG_LEVEL=warning ./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

For live tuning, enable periodic runtime stats:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
```

The stats line reports real pipeline FPS plus average and worst frame time. If
FPS drops, increase `inference_interval` in the preset so the ONNX model runs
less often. If the crop follows too slowly, lower `mask_smoothing`; if it jitters,
raise it slightly.

See [docs/architecture.md](docs/architecture.md), [docs/live-video.md](docs/live-video.md),
[docs/filters.md](docs/filters.md), and [docs/obs.md](docs/obs.md).
