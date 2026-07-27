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
- `config/presets/ai-background-blur.yaml`: experimental auto-framing plus CPU
  background blur with the smallest blur radius.
- `config/presets/background-blur.yaml`: background blur plus light contrast.
- `config/presets/debug.yaml`: FPS overlay and histogram metadata for validation.

For calls, prefer `clean.yaml` or no filters. Filters such as `sharpen`,
`background_blur`, `sobel`, `histogram` overlays, and text overlays are CPU-heavy
and can add latency at 1280x720.

The CLI reports both the requested backend and the backend used by the first
processed frame:

```text
filter_backend=requested:opencl filters=1 filters_active=[color_adjust]
filter_backend_active=opencl
```

If it prints `filter_backend_active=cpu`, the filter fell back because OpenCL was
not compiled in or no usable OpenCL device was available. `auto_frame`,
`background_blur`, `sharpen`, overlays, and histogram still run on CPU.

`auto_frame` crops toward the detected foreground and scales back to the output
resolution. Useful knobs:

```yaml
- type: auto_frame
  enabled: true
  target_fill: 0.62
  max_zoom: 1.7
  foreground_threshold: 128
```

## Notes

`background_blur` currently uses the project's deterministic segmentation fallback.
It is useful for validating the processing path, but production-quality person
segmentation still requires wiring a real ONNX Runtime model.
