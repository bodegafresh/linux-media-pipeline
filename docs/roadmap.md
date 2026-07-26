# Roadmap

## Available Now

- Fedora/Ninja/CMake build scripts.
- Config loading from YAML.
- GoPro UDP capture endpoint validation.
- In-process FFmpeg decode for configured GoPro UDP MPEG-TS input.
- Project-owned `lmp::frame::Frame`.
- YAML-driven filter pipeline.
- V4L2 virtual camera output.
- OBS and browser validation through `/dev/video20`.
- Unified streaming entrypoint: `scripts/stream.sh`.
- External USB camera bridge through FFmpeg CLI.
- GPU backend contracts for OpenCL and Vulkan.
- AI backend contract for ONNX Runtime segmentation.

## Next Work

- Replace the external USB bridge with an in-process `UsbCameraSource`.
- Add RTSP/SRT/WebRTC capture adapters.
- Move CPU filters behind GPU kernels where useful.
- Implement real OpenCL and Vulkan command queues/kernels.
- Wire real ONNX Runtime session execution for segmentation.
- Add Prometheus metrics endpoint.
- Add production benchmarks for latency, FPS, CPU, GPU, and queue depth.
- Package a release once end-to-end GoPro and USB paths are validated on Fedora.
