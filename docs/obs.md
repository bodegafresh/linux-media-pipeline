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
background_processing_backend=opencl
```

For a tuning pass, add stats:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
```

Use the reported FPS, frame times, and `dropped_frames` count to tune latency.
If the stream drifts behind audio, use the lighter preset:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur-performance.yaml --stats-every 5
```

For background replacement in OBS, keep the same V4L2 source and switch only the
preset:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-color.yaml --stats-every 5
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-image.yaml --stats-every 5
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-video.yaml --stats-every 5
```

Place images, GIFs, or videos under `assets/backgrounds/` and update
`background_path` in the preset.

## Horizontal And Vertical Outputs

For a normal stream plus a reels/shorts layout, run one capture pipeline and two
V4L2 outputs:

```bash
./scripts/setup-loopback.sh 20 lmp-horizontal 21 lmp-vertical
./scripts/stream.sh gopro-dual /dev/video20 /dev/video21
```

This decodes the GoPro UDP stream once, then branches in-process:

```text
GoPro UDP -> FFmpeg decode -> horizontal preset -> /dev/video20
                         `-> vertical preset   -> /dev/video21
```

The default dual presets are:

- `config/presets/dual-horizontal.yaml`: `1280x720`, image background,
  `/dev/video20`.
- `config/presets/dual-vertical.yaml`: `720x1280`, smart 9:16 crop reusing
  the primary processed frame, `/dev/video21`.

Both outputs must use the same FPS to stay synchronized. The resolutions and
filters can differ. In OBS, add two independent Video Capture Device sources:
one for `/dev/video20` in a horizontal scene and one for `/dev/video21` in a
vertical scene.

The vertical branch reuses the primary AI/background result and then applies a
smart 9:16 crop. The crop reads `segmentation_mask_bounds`, keeps headroom, and
uses a dead zone plus motion inertia so it does not chase every small mask
change. This keeps the vertical camera steadier while still following larger
side-to-side movement.

If the mask line says `onnx_unavailable_center`, OBS will still receive video,
but the app is using the centered fallback because
`assets/models/pphumanseg.onnx` is missing or incompatible.

Recoverable FFmpeg startup logs are suppressed by default to keep the important
runtime lines visible. For decoder troubleshooting, opt in per command:

```bash
LMP_FFMPEG_LOG_LEVEL=error ./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
```

## OBS Setup

1. Add a Video Capture Device source.
2. Select `linux-media-pipeline` or `/dev/video20`.
3. Match the same resolution and FPS as the pipeline output.

## Restream And Live Platforms

If OBS keeps streaming but Facebook or YouTube end the live after a few seconds,
first treat it as an encoder or platform-ingest compatibility issue. The V4L2
camera is only an OBS source; Restream, Facebook, and YouTube validate the final
encoded RTMP stream that OBS sends.

Use this conservative baseline for the first stable multi-platform test:

- OBS Video: `1280x720`, `30 FPS`.
- OBS Output Mode: `Advanced`.
- Encoder: `x264` with `veryfast` for the first test, or AMD hardware H.264
  only after the x264 test is stable.
- Rate Control: `CBR`.
- Video Bitrate: `4000 Kbps` for 720p30.
- Keyframe Interval: `2` seconds.
- Profile: `Main`.
- Audio: `AAC`, stereo, `44.1 kHz` or `48 kHz`, `128-160 Kbps`.
- OBS Advanced color: `NV12`, color space `709`.

After a 10 minute stable test, 1080p30 is reasonable with `4500-6000 Kbps`.
Keep OBS FPS at `30` because the virtual camera currently outputs 30 FPS; using
60 FPS in OBS duplicates frames and adds encoder pressure without adding real
camera motion.

Validate in this order:

1. Stream from OBS directly to an unlisted YouTube test for 5-10 minutes.
2. Stream from OBS to Restream with only YouTube enabled for 5-10 minutes.
3. Enable Facebook alone through Restream for 5-10 minutes.
4. Enable YouTube and Facebook together only after both single-destination tests
   are stable.

During each test, watch:

- OBS `Stats`: dropped frames, skipped frames, encoder overload, reconnects.
- Restream stream health: bitrate, keyframe interval, FPS, and destination
  errors.
- Platform dashboards: YouTube Live Control Room and Facebook Live Producer
  warnings.

If YouTube and Facebook end but OBS still looks connected to Restream, the most
useful evidence is the OBS log plus Restream channel-health error. In OBS use
`Help > Log Files > Upload Current Log File`, or inspect
`~/.config/obs-studio/logs/` on Linux.

## Other Applications

The output is a normal V4L2 camera. Google Meet, Zoom, Chrome, and other Linux
apps should use the same `/dev/video20` source after the stream command is
running.
