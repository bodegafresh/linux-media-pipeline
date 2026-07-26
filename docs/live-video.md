# Live Video Bridges

The core library exposes capture, decode, filter, GPU, AI, and output contracts.
The GoPro path uses an in-process FFmpeg decoder. USB bridging currently uses
FFmpeg as an external adapter until `UsbCameraSource` is implemented.

## GoPro Hero11 To V4L2

```bash
./scripts/setup-loopback.sh 20 linux-media-pipeline
./scripts/stream.sh gopro-udp
```

Then select `/dev/video20` in OBS, Google Meet, Zoom, Chrome, or any other V4L2
consumer.

GoPro input and V4L2 output settings come from `config/default.yaml`.

## USB Camera To V4L2

```bash
./scripts/stream.sh usb /dev/video0 /dev/video20 1280 720 30
```

## Test Pattern To V4L2

```bash
./scripts/stream.sh test-pattern /dev/video20
```

## Application Compatibility

The output is a normal Linux V4L2 virtual camera. OBS is only the validation app.
Any application that can read a V4L2 camera can consume the same device, including
Google Meet in Chrome.

## Next Internal Integration

The remaining engineering step for USB is replacing the external FFmpeg process
with an in-process capture adapter that emits `lmp::frame::Frame`, applies
`FilterPipeline`, and writes through `V4l2Output`.
