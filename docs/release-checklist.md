# Release Checklist

- `./scripts/install-fedora-deps.sh`
- `./scripts/build.sh`
- `./scripts/test.sh`
- `./scripts/run.sh`
- `./scripts/benchmark.sh`
- `./scripts/setup-loopback.sh 20 linux-media-pipeline`
- `./build/dev/lmp --open-capture`
- `./build/dev/lmp --check-output`
- `./scripts/run-obs-test-pattern.sh`
- `./scripts/run-gopro-to-v4l2.sh`
- `./scripts/run-usb-to-v4l2.sh /dev/video0 /dev/video20`
- OBS sees `/dev/video20`.
- Google Meet or another browser app sees `/dev/video20`.
- In-process FFmpeg/V4L2 frame integration is complete before tagging `1.0.0`.
