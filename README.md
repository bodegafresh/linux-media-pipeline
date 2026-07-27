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
preset fuses blur and color cleanup in the OpenCL blur filter and falls back to
CPU if no GPU runtime is available.

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
enables real ONNX segmentation automatically. Put a person-segmentation model at:

```text
assets/models/person-segmentation.onnx
```

The expected model shape is a common segmentation layout: RGB float input in
`1x3xHxW` format and a mask-like float output. If ONNX Runtime or the model is
missing, the pipeline keeps running with the deterministic fallback.

See [docs/architecture.md](docs/architecture.md), [docs/live-video.md](docs/live-video.md),
[docs/filters.md](docs/filters.md), and [docs/obs.md](docs/obs.md).
