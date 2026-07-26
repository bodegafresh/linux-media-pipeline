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
./scripts/run-obs-test-pattern.sh
```

Then open OBS and select `/dev/video20`. Use:

- Video format: `RGB24` if OBS offers it.
- Resolution: `1280x720`.
- FPS: `30`.

## Show The GoPro Feed

Keep this bridge running:

```bash
./scripts/run-gopro-to-v4l2.sh
```

Then select `/dev/video20` in OBS.

## OBS Setup

1. Add a Video Capture Device source.
2. Select `linux-media-pipeline` or `/dev/video20`.
3. Match the same resolution and FPS as the pipeline output.

## Remaining Live Video Work

The project can now validate the GoPro UDP listener, stream a V4L2 test pattern,
and bridge GoPro UDP MPEG-TS to `/dev/video20` with FFmpeg. The next internal
integration is replacing the external FFmpeg process with an in-process adapter so
decoded frames can pass through `FilterPipeline`.
