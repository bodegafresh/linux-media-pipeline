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
onnx_runtime_model=input_name=image input_rank=4 input_dims=[1x3x256x256] ...
```

That line means `background_blur` is using the real
`assets/models/person-segmentation.onnx` model through ONNX Runtime. If it says
`onnx_unavailable_center` or `center`, the stream is still usable, but it is
using the fixed fallback mask instead of true person segmentation.
The provider lines show what ONNX Runtime actually selected. A fallback of
`true` means the requested provider was not used.
Fedora OpenVINO packages are installed by the dependency script, but ONNX Runtime
must still report `OpenVINOExecutionProvider` before the pipeline can use it.
OpenVINO inference device and OpenCL background processing device are reported
separately. On an AMD Radeon workstation, the safe OpenVINO device is `CPU`;
OpenVINO `GPU` must not be used unless OpenVINO itself enumerates a compatible
Intel GPU.

The AI blur preset exposes the main performance knobs:

```yaml
inference_interval: 4
mask_smoothing: 0.35
mask_expand: 1
mask_feather: 2
fallback_mask_mode: tracked_center
provider: auto
allow_provider_fallback: true
openvino_device: CPU
radius: 5
```

Provider values are `auto`, `cpu`, `migraphx`, `rocm`, and `openvino`.
`rocm` is explicit for Fedora ONNX Runtime builds that expose
`ROCMExecutionProvider` instead of `MIGraphXExecutionProvider`.

The AI blur startup diagnostics include:

```text
segmentation_mask_coverage_raw=...
segmentation_mask_coverage_refined=...
```

ONNX masks are kept as grayscale probability mattes before thresholding,
expansion, and feathering. This is important for presenter use: a hard binary
mask tends to fragment hands, hair, and shoulders, while the probability matte
lets the blur keep a softer body-shaped foreground.

Values near `1.0` mean the model is classifying almost the whole frame as
foreground, so little or no background blur will be visible. Values around
`0.20` to `0.55` are usually more plausible for a seated presenter.

If the model output uses the opposite polarity, set:

```yaml
invert_mask: true
```

The startup diagnostics will print `segmentation_mask_inverted=true`.
Most person-segmentation models mark the person as high/white, so the ROCm
preset keeps `invert_mask: false`. Use `invert_mask: true` only when the
background is being protected and the presenter is being blurred.

For noisy masks that create patchy blur, keep only the main connected person
region before feathering:

```yaml
keep_largest_component: true
```

The diagnostics then include `segmentation_mask_coverage_component`.

Coverage guardrails reject obviously bad masks:

```yaml
min_mask_coverage: 0.08
max_mask_coverage: 0.80
```

When rejected, the stream prints `segmentation_mask_rejected=...` and falls back
to the configured stable mask mode instead of using a noisy mask for blur or
auto-framing.

For the ROCm preset, the intended path is body segmentation first and the
fallback is intentionally non-elliptical so it does not visibly pulse a center
oval over the video. A healthy run should normally show:

```text
background_blur_mask_active=onnx
segmentation_inference_backend=ROCMExecutionProvider
background_processing_backend=opencl
```

If `background_blur_mask_active=onnx_rejected_luminance` appears repeatedly, the
model output is still being rejected by coverage checks and the segmentation
tuning needs another pass.

For live streaming, a rejected or failed ONNX frame briefly reuses the last
accepted person matte before falling back to luminance. This avoids a single
bad inference suddenly blurring the presenter, but the reuse is capped so the
matte does not visibly lag behind movement for seconds. The stream reports this
as:

```text
background_blur_mask_active=onnx_previous
segmentation_mask_reused_previous=true
segmentation_mask_previous_reuse_count=...
```

For recording or streaming today, prefer the stable preset:

```bash
./scripts/stream.sh gopro-udp /dev/video20 config/presets/recording-background-blur.yaml
```

It uses OpenCL blur with a wide protected center region and does not rely on a
noisy ONNX mask or automatic reframing.

`inference_interval` controls how often ONNX runs. Higher values reduce CPU/GPU
pressure but make tracking less reactive. `mask_smoothing` damps mask jitter;
lower values follow movement faster, higher values look steadier.
`mask_expand` protects the person edge before blur is applied, while
`mask_feather` softens the matte so the transition to the blurred background is
less harsh.
The ONNX mask is intentionally kept at model resolution and scaled in the GPU
kernel; refining it at full 1280x720 is too expensive for live video.
ONNX inference runs asynchronously after the first mask, so the output keeps
using the latest valid mask while the next one is calculated.
`fallback_mask_mode: luminance` keeps video usable when ONNX fails without
showing a center oval over the presenter. `radius` controls
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

When `auto_frame` is enabled inside `background_blur`, the OpenCL path applies a
small deadband plus center/zoom step limits so the crop follows the presenter
without breathing in and out on every small segmentation change. The applied crop is
reported as:

```text
background_blur_auto_frame_crop=x,y,width,height
```

## Notes

`background_blur` can use ONNX Runtime when the project is built with the
ONNX Runtime headers/library and the configured model exists at
`assets/models/person-segmentation.onnx`. ONNX Runtime is the engine; the `.onnx`
file is the model. The model should accept RGB float input in `1x3xHxW` layout
and return either a single mask or a common two-channel segmentation output such
as `1x2xHxW` or `1xHxWx2`. Without that, it keeps using the deterministic
fallback so streaming still works.
