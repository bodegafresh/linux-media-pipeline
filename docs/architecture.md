# Architecture

linux-media-pipeline follows a hexagonal architecture. Domain interfaces live under
`include/lmp`, concrete adapters live under `src`, and applications compose those
parts at the edge.

## Initial Module Boundaries

- Capture: media ingress such as GoPro UDP, USB camera, RTSP, files, and future NDI,
  PipeWire, SRT, RTP, and WebRTC adapters.
- Decoder: FFmpeg adapter with hardware acceleration extension points.
- Frame: project-owned frame representation. FFmpeg-specific types must not leak
  outside the decoder adapter.
- GPU: backend abstraction for OpenCL first, Vulkan Compute later.
- Filters: independent video filters depending on frame and GPU interfaces.
- Output: V4L2 virtual camera first, OBS SDK, PipeWire, files, and RTMP later.
- Config: YAML-backed runtime configuration with hot reload in a later phase.
- Metrics: Prometheus exporter for FPS, latency, processing time, queues, and memory.

## Phase Policy

Each phase must compile, pass tests, run locally, and update documentation before the
next phase starts.
