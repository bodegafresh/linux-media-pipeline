# ONNX GPU Audit

This audit records the current ONNX path before any MIGraphX package changes or
provider behavior changes that would require workstation approval.

## Current ONNX Runtime Discovery Logic

CMake enables `LMP_HAS_ONNXRUNTIME` when `pkg-config libonnxruntime` is found,
or when the ONNX Runtime header and library are found manually. Runtime provider
availability is queried through `Ort::GetAvailableProviders()`. Package names are
not treated as proof that a provider is available.

## Current Session Creation

`OnnxRuntimeEngine` owns one `Ort::Env`, one `Ort::SessionOptions`, and one
`Ort::Session`. The session is created lazily by `BackgroundBlurFilter` when the
first ONNX mask is requested.

Graph optimization is enabled with `ORT_ENABLE_ALL`. Intra-op threads are chosen
conservatively from hardware concurrency.

## Current Provider Selection

Supported requested values are `auto`, `cpu`, `migraphx`, and `openvino`.

For the AMD path, `auto` prefers `MIGraphXExecutionProvider` only if the active
ONNX Runtime build enumerates it. Otherwise it stays on
`CPUExecutionProvider`. `openvino` remains an explicit non-AMD option and does
not imply AMD GPU inference.

MIGraphX activation is not yet implemented; requesting it before runtime support
exists falls back to CPU when fallback is allowed.

## Current Model Input/Output Names

The engine reads input and output names from ONNX Runtime after session creation
and reports them through `onnx_runtime_model`.

## Current Tensor Shapes

The engine reads input/output tensor shapes through the ONNX Runtime C API.
When a model does not expose readable static shapes, config overrides such as
`input_shape: 1x3x256x256` and `output_shape: 1x1x256x256` are used.

## Current Model Opset

The current implementation does not yet inspect model opset or operator list.
That remains for the model compatibility phase.

## Current Preprocessing

Frames are decoded to RGB/BGR-compatible frame memory. Preprocessing samples the
video frame into the ONNX input dimensions and writes normalized RGB float
values according to the detected channel axis.

## Current Postprocessing

The output tensor is interpreted as a probability/logit mask. Single-channel and
common two-channel segmentation tensors are supported. The mask is kept at model
resolution, refined there, and scaled in the OpenCL compositing kernel.

## Current Inference Frequency

Inference is throttled with `inference_interval`. After the first mask, inference
runs asynchronously; the live video path keeps using the latest valid mask while
the next mask is computed.

## Current Allocations Per Inference

Input tensors and masks are still allocated per inference. Tensor pooling and
buffer reuse remain future optimization work.

## Current Inference Thread

The first mask is computed synchronously. Later masks run in a bounded
single-pending `std::future`; the live frame loop never accumulates an unbounded
queue.

## Current Fallback Behavior

If ONNX Runtime, the model, or the requested provider is unavailable, the stream
continues with deterministic fallback masks when fallback is allowed. Fallback is
reported explicitly.

## Current Metrics

Startup diagnostics include provider requested/active/fallback, model summary,
OpenVINO device fields, OpenCL platform/device, and coarse runtime frame stats.
Fine-grained preprocess/inference/postprocess timing is not implemented yet.
