# Filters

Filters are configured in YAML under `filters`. A filter is active only when
`enabled: true`.

Run with the default config:

```bash
./scripts/stream.sh gopro-udp
```

Run with a preset:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/clean.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-presenter.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml
./scripts/stream.sh gopro-udp /dev/video20 config/presets/background-blur.yaml
```

The CLI prints active filters on startup:

```text
filter_backend=requested:opencl filters=1 filters_active=[color_adjust]
filter_backend_active=opencl
```

## Useful Presets

- `config/presets/realtime.yaml`: lowest-latency path with only identity.
- `config/presets/clean.yaml`: low-latency `color_adjust`, using OpenCL when
  the binary and system runtime support it.
- `config/presets/ai-presenter.yaml`: auto-framing and OpenCL color cleanup for
  tutorial/meeting framing.
- `config/presets/ai-background-blur.yaml`: auto-framing plus fused OpenCL
  background blur and color cleanup, with CPU fallback.
- `config/presets/background-blur.yaml`: fused OpenCL background blur plus light
  contrast.
- `config/presets/debug.yaml`: FPS overlay and histogram metadata for validation.

For calls, prefer `ai-presenter.yaml` when framing matters and
`ai-background-blur.yaml` when your GPU can keep up. Filters such as `sharpen`,
`sobel`, `histogram` overlays, and text overlays are CPU-heavy and can add
latency at 1280x720.

The CLI reports both the requested backend and the backend used by the first
processed frame:

```text
filter_backend=requested:opencl filters=1 filters_active=[color_adjust]
filter_backend_active=opencl
```

If it prints `filter_backend_active=cpu`, `color_adjust` fell back because
OpenCL was not compiled in or no usable OpenCL device was available.
`background_blur` reports its own runtime backend:

```text
background_blur_backend_active=opencl
```

For the AI preset it also reports the mask source:

```text
background_blur_mask_active=onnx
onnx_runtime_provider_requested=auto
onnx_runtime_provider_active=CPUExecutionProvider
onnx_runtime_provider_fallback=false
```

That line means `background_blur` is using the real
`assets/models/person-segmentation.onnx` model through ONNX Runtime. If it says
`onnx_unavailable_center` or `center`, the stream is still usable, but it is
using the fixed fallback mask instead of true person segmentation.
The provider lines show what ONNX Runtime actually selected. A fallback of
`true` means the requested provider was not used.

The AI blur preset exposes the main performance knobs:

```yaml
inference_interval: 3
mask_smoothing: 0.70
fallback_mask_mode: tracked_center
provider: auto
allow_provider_fallback: true
radius: 8
```

`inference_interval` controls how often ONNX runs. Higher values reduce CPU/GPU
pressure but make tracking less reactive. `mask_smoothing` damps mask jitter;
lower values follow movement faster, higher values look steadier.
`fallback_mask_mode: tracked_center` keeps video usable when ONNX fails by
tracking the crop with a lightweight foreground heuristic. `radius` controls
background blur strength; higher values blur more but cost more GPU time.

For calibration, print runtime stats every five seconds:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
```

`auto_frame`, `sharpen`, overlays, and histogram still run on CPU.

`auto_frame` crops toward the detected foreground and scales back to the output
resolution. Useful knobs:

```yaml
- type: auto_frame
  enabled: true
  target_fill: 0.62
  max_zoom: 1.7
  smoothing: 0.84
  dead_zone: 0.05
  foreground_threshold: 128
```

## Notes

`background_blur` can use ONNX Runtime when the project is built with the
ONNX Runtime headers/library and the configured model exists at
`assets/models/person-segmentation.onnx`. ONNX Runtime is the engine; the `.onnx`
file is the model. The model should accept RGB float input in `1x3xHxW` layout
and return either a single mask or a common two-channel segmentation output such
as `1x2xHxW` or `1xHxWx2`. Without that, it keeps using the deterministic
fallback so streaming still works.
