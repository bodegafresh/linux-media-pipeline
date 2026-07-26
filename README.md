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

## Requirements

Target platform:

- Fedora 44
- GCC 16
- CMake 4.x
- Conan 2.x
- Ninja
- OpenCL 3.0
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

## Development

The project advances in phases. Each phase must compile, pass tests, run locally,
and update documentation before the next phase starts.

See [Architecture](docs/architecture.md) and [Roadmap](docs/roadmap.md).
