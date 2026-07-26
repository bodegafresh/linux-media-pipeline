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
./scripts/stream.sh gopro-udp /dev/video20 config/presets/background-blur.yaml
```

The CLI prints active filters on startup:

```text
filters=4 filters_active=[brightness,contrast,saturation,sharpen]
```

## Useful Presets

- `config/presets/clean.yaml`: subtle brightness, contrast, saturation, sharpen,
  and FPS overlay.
- `config/presets/background-blur.yaml`: background blur plus light contrast.

## Notes

`background_blur` currently uses the project's deterministic segmentation fallback.
It is useful for validating the processing path, but production-quality person
segmentation still requires wiring a real ONNX Runtime model.
