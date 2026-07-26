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

## OBS Setup

1. Add a Video Capture Device source.
2. Select `linux-media-pipeline` or `/dev/video20`.
3. Match the same resolution and FPS as the pipeline output.

## Remaining Live Video Work

The project can now validate the GoPro UDP listener and V4L2 device. The final
live-video bridge still needs FFmpeg MPEG-TS demux/decode and continuous frame
writes into the V4L2 device.
