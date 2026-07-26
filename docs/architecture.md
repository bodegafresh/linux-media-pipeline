# Architecture

linux-media-pipeline follows a hexagonal architecture. Domain-facing interfaces
live under `include/lmp`; concrete adapters live under `src`; the CLI composes
capture, decode, filters, and output at the edge.

## Runtime Path

```text
Capture -> Decoder -> Frame -> FilterPipeline -> Output
```

The GoPro runtime path is:

```text
GoPro UDP MPEG-TS -> FfmpegDecoder -> lmp::frame::Frame -> FilterPipeline -> V4l2Output
```

## Modules

- Capture: `ICaptureSource` and `GoProUdpSource` parse and validate
  `udp://host:port` endpoints.
- Decoder: `FfmpegDecoder` uses FFmpeg libraries when available and emits RGB
  `lmp::frame::Frame` instances.
- Frame: `Frame` owns pixel bytes, format, dimensions, strides, timestamp, and
  metadata. FFmpeg types do not leak outside the decoder adapter.
- Filters: `IVideoFilter`, `FilterRegistry`, and `FilterPipeline` compose
  independent filters from YAML.
- Output: `IOutputSink` and `V4l2Output` write to Linux virtual camera devices.
- GPU: `IGpuBackend`, `OpenClBackend`, `VulkanBackend`, `GpuBuffer`, and
  `BufferPool` define the future acceleration boundary.
- AI: `IInferenceEngine`, `OnnxRuntimeEngine`, and `SegmentationMask` define the
  future segmentation boundary.
- Config: `config/default.yaml` controls capture, output, and filters.

## Filters

Implemented CPU filters:

- Identity
- Grayscale, negative, sepia
- Box blur, Gaussian blur, sharpen, Sobel
- Gamma, exposure, contrast, brightness, saturation
- White balance, temperature, tint
- Text overlay, FPS overlay, timestamp overlay
- Histogram
- Background blur with deterministic fallback segmentation

## V4L2 Consumers

The output is a normal Linux V4L2 camera. OBS is only the primary validation
target; Google Meet, Zoom, Chrome, and other camera consumers can select the same
`/dev/video20` device.

## Extension Points

- Add a capture source by implementing `ICaptureSource`.
- Add a decoder by emitting `lmp::frame::Frame`.
- Add a filter by implementing `IVideoFilter` and registering it in
  `create_default_registry`.
- Add an output by implementing `IOutputSink`.
- Add GPU acceleration behind `IGpuBackend` without changing filters.
