# Architecture

linux-media-pipeline follows a hexagonal architecture. Domain interfaces live under
`include/lmp`, concrete adapters live under `src`, and applications compose those
parts at the edge.

## Initial Module Boundaries

- Capture: media ingress such as GoPro UDP, USB camera, RTSP, files, and future NDI,
  PipeWire, SRT, RTP, and WebRTC adapters. `GoProUdpSource` parses
  `udp://host:port` and can bind a Linux UDP socket for the configured endpoint.
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
  Phase 5 adds CPU baseline color correction filters: `gamma`, `exposure`,
  `contrast`, `brightness`, `saturation`, `white_balance`, `temperature`, and
  `tint`.
  Phase 6 adds bitmap text overlays (`text_overlay`, `fps_overlay`,
  `timestamp_overlay`) and `histogram`, which stores 16-bin luminance data in
  frame metadata and can draw a compact overlay.
- GPU: Phase 7 introduces `IGpuBackend`, `OpenClBackend`, `GpuBuffer`, and
  `BufferPool`. Host-memory imports are represented as shared buffers so later
  OpenCL kernels can use the same zero-copy contract.
- GPU: Phase 8 adds `VulkanBackend` behind the same `IGpuBackend` contract. The
  adapter is runtime-light for now and shares the buffer/import semantics used by
  OpenCL.
- AI: Phase 9 adds `IInferenceEngine`, `OnnxRuntimeEngine`, `SegmentationMask`,
  and `background_blur`. The current ONNX adapter is runtime-light and provides a
  deterministic luminance-based segmentation fallback until the real ONNX Runtime
  dependency is wired in.
- Output: V4L2 virtual camera first, OBS SDK, PipeWire, files, and RTMP later.
  `V4l2Output` can open the configured virtual camera device and write packed
  frame bytes.
- Config: YAML-backed runtime configuration. Phase 2 supports the project config
  shape in `config/default.yaml`; hot reload arrives in a later phase.
- Metrics: Prometheus exporter for FPS, latency, processing time, queues, and memory.

## Current Runtime Gap

The application now instantiates the configured GoPro UDP source and can open the
socket with `--open-capture`, and it can validate the V4L2 endpoint with
`--check-output`. The remaining live-video path is FFmpeg MPEG-TS demux/decode,
frame conversion into `lmp::frame::Frame`, and continuous V4L2 frame writes.

## Filter Extension Flow

1. Implement `lmp::filters::IVideoFilter`.
2. Register the factory in `create_default_registry`.
3. Add YAML parameters to `config/default.yaml`.
4. Cover behavior with tests before enabling it by default.

## Phase Policy

Each phase must compile, pass tests, run locally, and update documentation before the
next phase starts.
