# OBS Validation

## Prepare the Virtual Camera

```bash
./scripts/setup-loopback.sh 20 linux-media-pipeline
v4l2-ctl --list-devices
```

The configured device is `/dev/video20`.

## Validate Application Endpoints

```bash
./scripts/build.sh
./scripts/test.sh
./scripts/run.sh
./build/dev/lmp --open-capture
./build/dev/lmp --check-output
```

Expected endpoint output:

```text
capture=gopro_udp udp=0.0.0.0:8554 capture_open=true
output=v4l2 device=/dev/video20 output_open=true
```

## Show Video In OBS

Keep the test pattern writer running:

```bash
./scripts/stream.sh test-pattern
```

Then open OBS and select `/dev/video20`. Use:

- Video format: `RGB24` if OBS offers it.
- Resolution: `1280x720`.
- FPS: `30`.

## Show The GoPro Feed

Keep this bridge running:

```bash
./scripts/stream.sh gopro-udp
```

Then select `/dev/video20` in OBS.

To verify filters visually:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/clean.yaml
```

The terminal prints `filters_active=[...]` when filters are enabled.

For the real AI background blur and auto-framing path, use:

```bash
./scripts/stream.sh install-model
./scripts/stream.sh doctor /dev/video20 config/presets/ai-background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

Expected runtime lines when the model is active:

```text
background_blur_backend_active=opencl
background_blur_mask_active=onnx
onnx_runtime_provider_active=CPUExecutionProvider
onnx_runtime_provider_fallback=false
```

For a tuning pass, add stats:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
```

Use the reported FPS and frame times to tune `inference_interval` and
`mask_smoothing` in `config/presets/ai-background-blur.yaml`.

If the mask line says `onnx_unavailable_center`, OBS will still receive video,
but the app is using the centered fallback because
`assets/models/person-segmentation.onnx` is missing or incompatible.

Recoverable FFmpeg startup logs are suppressed by default to keep the important
runtime lines visible. For decoder troubleshooting, opt in per command:

```bash
LMP_FFMPEG_LOG_LEVEL=error ./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

## OBS Setup

1. Add a Video Capture Device source.
2. Select `linux-media-pipeline` or `/dev/video20`.
3. Match the same resolution and FPS as the pipeline output.

## Other Applications

The output is a normal V4L2 camera. Google Meet, Zoom, Chrome, and other Linux
apps should use the same `/dev/video20` source after the stream command is
running.
