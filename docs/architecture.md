# Architecture

linux-media-pipeline follows a hexagonal architecture. Domain interfaces live under
`include/lmp`, concrete adapters live under `src`, and applications compose those
parts at the edge.

## Initial Module Boundaries

- Capture: media ingress such as GoPro UDP, USB camera, RTSP, files, and future NDI,
  PipeWire, SRT, RTP, and WebRTC adapters.
- Decoder: FFmpeg adapter with hardware acceleration extension points.
- Frame: project-owned frame representation. FFmpeg-specific types must not leak
  outside the decoder adapter. The current `Frame` stores format, dimensions,
  strides, timestamp, metadata, and owned bytes.
- GPU: backend abstraction for OpenCL first, Vulkan Compute later.
- Filters: independent video filters depending on frame and GPU interfaces.
  Filters are created by `FilterRegistry` and executed by `FilterPipeline`.
  Phase 3 includes `identity`, `grayscale`, `negative`, and `sepia`. The visual
  filters currently support packed 8-bit `RGBA`, `RGB`, and `BGR` frames.
  Phase 4 adds CPU baseline spatial filters: `box_blur`, `gaussian_blur`,
  `sharpen`, and `sobel`. The `blur` type is an alias for `box_blur`.
- Output: V4L2 virtual camera first, OBS SDK, PipeWire, files, and RTMP later.
- Config: YAML-backed runtime configuration. Phase 2 supports the project config
  shape in `config/default.yaml`; hot reload arrives in a later phase.
- Metrics: Prometheus exporter for FPS, latency, processing time, queues, and memory.

## Filter Extension Flow

1. Implement `lmp::filters::IVideoFilter`.
2. Register the factory in `create_default_registry`.
3. Add YAML parameters to `config/default.yaml`.
4. Cover behavior with tests before enabling it by default.

## Phase Policy

Each phase must compile, pass tests, run locally, and update documentation before the
next phase starts.
