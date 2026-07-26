# linux-media-pipeline

linux-media-pipeline is a Linux-first C++20 project for building very low latency
GPU video pipelines. The first production target is:

GoPro Hero11 -> UDP MPEG-TS -> FFmpeg -> GPU -> V4L2 virtual camera -> OBS Studio.

The architecture is intentionally decoupled so future adapters can support USB
cameras, RTSP, RTP, SRT, WebRTC, files, NDI, PipeWire, GStreamer, Unreal Engine,
Galactic Explorer, and AI workloads.

## Status

Phase 1 is the current baseline:

- Professional repository layout.
- CMake/Ninja build.
- Conan 2 project metadata.
- Minimal C++20 core library and CLI.
- CTest smoke tests.
- Fedora-oriented operational scripts.
- CI skeleton.
- YAML configuration loading.
- Project-owned frame model.
- Filter registry, factory, pipeline, and identity filter.
- Initial visual filters: grayscale, negative, and sepia.
- Spatial filters: box blur, gaussian blur, sharpen, and sobel.
- Color correction filters: gamma, exposure, contrast, brightness, saturation,
  white balance, temperature, and tint.
- Overlay and analysis filters: text overlay, FPS overlay, timestamp overlay, and
  luminance histogram.
- GPU memory foundation: backend interface, OpenCL adapter skeleton, zero-copy host
  imports, and reusable buffer pool.
- Vulkan backend skeleton sharing the GPU buffer and zero-copy contracts.
- AI foundation: ONNX Runtime adapter skeleton, segmentation masks, and background
  blur filter with deterministic fallback segmentation.
- GoPro UDP capture source with Linux socket bind support.
- V4L2 output sink with explicit OBS endpoint validation.

## Requirements

Target platform:

- Fedora 44
- GCC 16
- CMake 4.x
- Conan 2.x
- Ninja
- OpenCL 3.0
- Vulkan 1.3+
- FFmpeg 8.x
- OBS 31+
- Linux kernel 6.x

The Phase 1 scaffold also builds with older CMake versions that support presets
version 6, which keeps local validation practical while the Fedora 44 toolchain
catches up.

## Quick Start

```bash
./scripts/install-fedora-deps.sh
./scripts/build.sh
./scripts/test.sh
./scripts/run.sh
```

To verify that the configured GoPro UDP listener can bind:

```bash
./build/dev/lmp --open-capture
```

The current CLI opens the UDP capture socket only when requested. MPEG-TS demuxing,
FFmpeg decode, frame conversion, and V4L2 output are the remaining pieces before
OBS receives live video.

To verify the virtual camera endpoint for OBS:

```bash
./scripts/setup-loopback.sh 20 linux-media-pipeline
./build/dev/lmp --check-output
```

To make OBS show a live validation image, keep this process running:

```bash
./build/dev/lmp --test-pattern
```

See [OBS Validation](docs/obs.md).

## Development

The project advances in phases. Each phase must compile, pass tests, run locally,
and update documentation before the next phase starts.

See [Architecture](docs/architecture.md) and [Roadmap](docs/roadmap.md).
