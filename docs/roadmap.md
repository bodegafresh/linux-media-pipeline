# Roadmap

## Phase 1

- Repository structure.
- CMake/Ninja build.
- Conan 2 project file.
- Smoke-testable library and CLI.
- Scripts and CI skeleton.

## Phase 2

- YAML configuration loader. Done.
- `Frame` model. Done.
- `FilterRegistry`, `FilterFactory`, and `FilterPipeline`. Done.
- `Identity` filter. Done.

Phase 2 intentionally keeps runtime dependencies small. The YAML loader is strict
and supports the project configuration shape currently used by `config/default.yaml`.

## Phase 3

- `Grayscale` filter. Done.
- `Negative` filter. Done.
- `Sepia` filter. Done.
- Configurable YAML entries. Done.
- Unit coverage for RGBA/BGR behavior. Done.

The filters are CPU baselines for packed 8-bit frames. GPU implementations and
YUV/NV12/P010-specific paths belong to later FFmpeg/GPU phases.

## Phase 4

- `BoxBlur` filter. Done.
- `GaussianBlur` filter. Done.
- `Sharpen` filter. Done.
- `Sobel` filter. Done.
- YAML parameters for `radius` and `amount`. Done.
- Unit coverage for deterministic RGB behavior. Done.

The filters remain CPU baselines and are disabled in `config/default.yaml` so the
default executable path stays lightweight until capture/decoder/output modules are
introduced.

## Phase 5

- `Gamma` filter. Done.
- `Exposure` filter. Done.
- `Contrast` filter. Done.
- `Brightness` filter. Done.
- `Saturation` filter. Done.
- `WhiteBalance` filter. Done.
- `Temperature` filter. Done.
- `Tint` filter. Done.
- YAML parameters for color correction. Done.
- Unit coverage for deterministic RGBA behavior. Done.

These filters are CPU baselines for packed 8-bit `RGB`, `BGR`, and `RGBA`
frames. Later GPU phases can replace the implementation behind the same filter
interfaces.

## Phase 6

- `TextOverlay` filter. Done.
- `FpsOverlay` filter. Done.
- `TimestampOverlay` filter. Done.
- `Histogram` filter. Done.
- YAML parameters for overlay placement and histogram rendering. Done.
- Unit coverage for deterministic overlay pixels and histogram metadata. Done.

The overlay implementation uses a small built-in bitmap font so the phase remains
portable and dependency-light. A future renderer can replace it behind the same
filter interfaces.

## Phase 7

- `IGpuBackend` interface. Done.
- `OpenClBackend` adapter skeleton. Done.
- `GpuBuffer` with owned and shared host-memory modes. Done.
- Zero-copy host buffer import contract. Done.
- `BufferPool` for reusable frame/GPU staging memory. Done.
- Unit coverage for ownership, zero-copy write-through, pool reuse, and OpenCL
  adapter behavior. Done.

This phase establishes the memory and backend contracts. Real OpenCL context
creation, command queues, kernels, and device profiling belong to the next GPU
implementation pass.

## Phase 8

- `VulkanBackend` adapter skeleton. Done.
- Shared `IGpuBackend` contract with OpenCL. Done.
- Zero-copy host import contract for Vulkan. Done.
- Fedora dependency script includes Vulkan headers/loader development packages.
  Done.
- Unit coverage for Vulkan adapter name, ownership, and import behavior. Done.

Real Vulkan instance/device selection, descriptor sets, command buffers, and
compute pipelines belong to the production Vulkan implementation pass.

## Phase 9

- `IInferenceEngine` interface. Done.
- `OnnxRuntimeEngine` adapter skeleton. Done.
- `SegmentationMask` model. Done.
- `BackgroundBlur` filter. Done.
- YAML configuration for ONNX model path and background blur. Done.
- Unit coverage for mask indexing, fallback segmentation, and foreground
  preservation. Done.

This phase keeps ONNX Runtime optional so Fedora build/test/run remains stable
without downloading model/runtime artifacts. The production pass should replace
the fallback segmentation with real ONNX Runtime session execution.

## GoPro Runtime Integration

- `ICaptureSource` interface. Done.
- `GoProUdpSource` endpoint parser. Done.
- Linux UDP socket bind/open/close lifecycle. Done.
- CLI `--open-capture` flag for explicit capture listener validation. Done.
- `IOutputSink` and `V4l2Output` lifecycle. Done.
- CLI `--check-output` flag for OBS virtual camera endpoint validation. Done.
- CLI `--test-pattern` RGB24 writer for OBS visual validation. Done.
- External FFmpeg GoPro UDP MPEG-TS to V4L2 bridge. Done.
- External FFmpeg USB camera to V4L2 bridge. Done.
- In-process FFmpeg MPEG-TS demux/decode. Pending.
- In-process FilterPipeline before V4L2 streaming. Pending.

## Phase 10

- OBS validation documentation. Done.
- Release checklist. Done.
- V4L2 output endpoint validation. Done.
- Benchmark target placeholder. Done.
- Final roadmap gap documentation. Done.
