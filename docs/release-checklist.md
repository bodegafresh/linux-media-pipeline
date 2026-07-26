# Release Checklist

- `./scripts/install-fedora-deps.sh`
- `./scripts/build.sh`
- `./scripts/test.sh`
- `./scripts/run.sh`
- `./scripts/benchmark.sh`
- `./scripts/setup-loopback.sh 20 linux-media-pipeline`
- `./build/dev/lmp --open-capture`
- `./build/dev/lmp --check-output`
- OBS sees `/dev/video20`.
- FFmpeg/V4L2 live-frame integration is complete before tagging `1.0.0`.
