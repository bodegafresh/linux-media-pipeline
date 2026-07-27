# Implementation Audit

This document records the current implementation before deeper AI segmentation,
tracking, scheduling, and GPU-provider work. It is intentionally factual: it
describes what the code does today, not the final target architecture.

## Current Architecture

`lmp` loads YAML configuration, creates capture/output adapters, builds a
`FilterPipeline`, and processes frames in process. The main live path is:

`FFmpegDecoder -> frame::Frame -> FilterPipeline -> V4l2Output`.

The filter registry constructs filters from flat per-filter parameters. The
background blur filter owns ONNX Runtime state lazily and owns OpenCL resources
through a process-wide OpenCL resource object.

## Current Data Flow

GoPro UDP input is decoded by FFmpeg into `frame::Frame` objects. Frames are
normalized to RGB24 for live output. Filters mutate the same frame object and
attach runtime metadata such as active backend, mask mode, and ONNX provider
state. The processed frame is written to the configured V4L2 loopback device.

## Current Thread Model

The live path is a single processing loop in `main.cpp`. Capture/decode,
filtering, output writing, and stats reporting happen synchronously on that
loop. `pipeline.threads` is parsed as configuration but there is no bounded
decode/inference/render worker graph yet.

## Current Copies Per Frame

The baseline live path copies decoded data into an `lmp::Frame`. The OpenCL
background blur path copies frame bytes from host memory to an OpenCL buffer,
runs the kernel, then copies the output buffer back to host memory before V4L2
write. ONNX segmentation builds a resized float input tensor from the host frame
and returns an 8-bit mask. There is no zero-copy GPU interop between FFmpeg,
ONNX Runtime, OpenCL, and V4L2 yet.

## Current GPU/CPU Boundaries

OpenCL accelerates background blur, color adjustment, and related pixel work
where supported. ONNX Runtime inference is executed through the default CPU
session unless explicit execution-provider activation is implemented. Provider
availability is now reported so the runtime no longer silently suggests GPU
inference when it is not active.

## Current ONNX Provider Behavior

The YAML parameter `provider` accepts `auto`, `cpu`, `rocm`, `migraphx`, and
`openvino`. The engine reports available providers, requested provider, active
provider, whether fallback occurred, and fallback reason. Today, `auto` and
`cpu` select `CPUExecutionProvider`. Other providers are detected but not yet
activated; if fallback is allowed, the engine reports the fallback and continues
on CPU. If fallback is disabled, provider mismatch makes the engine unavailable.

## Current Mask Representation

Segmentation masks are `ai::SegmentationMask`: an 8-bit single-channel mask with
frame-sized sampling via `at(x, y)`. ONNX output is resized conceptually by mask
sampling in the OpenCL kernel. Fallback masks include luminance, center, and
tracked-center approximations.

## Current Known Limitations

- ONNX Runtime GPU execution providers are reported but not activated.
- The OpenCL blur path still copies frame bytes to and from host memory.
- ONNX inference runs inside the frame loop and is throttled by
  `inference_interval`.
- Mask refinement is basic; there is no dedicated temporal confidence model.
- Auto framing depends on mask quality and currently runs inside the blur
  filter.
- Audio is outside this process; sync depends on keeping video frame cadence
  stable for the consumer application.

## Planned Modifications

- Activate supported ONNX Runtime providers explicitly and fail loudly when a
  requested provider cannot be used and fallback is disabled.
- Split segmentation, mask refinement, tracking, background effects, and output
  pacing into explicit components with measured boundaries.
- Add model metadata validation and clearer errors for model shape/layout
  mismatches.
- Introduce a stable async frame scheduler only after profiling confirms where
  latency is coming from.
- Add benchmark and visual validation commands without creating one-off scripts
  for every scenario.

## Migration Risks

- GPU-provider APIs differ across ONNX Runtime builds and Linux distributions.
- Provider activation may require runtime libraries that Fedora packages do not
  install by default.
- More aggressive mask refinement can improve edges but add latency.
- Async processing can improve throughput but may make latency and frame drops
  harder to reason about.
- Tight coupling between blur and auto framing should be reduced gradually to
  avoid regressing the currently working OBS/Google Meet V4L2 flow.
