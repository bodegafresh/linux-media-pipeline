# Roadmap

## Phase 1

- Repository structure.
- CMake/Ninja build.
- Conan 2 project file.
- Smoke-testable library and CLI.
- Scripts and CI skeleton.

## Phase 2

- YAML configuration loader. Done.
- `Frame` model. Done.
- `FilterRegistry`, `FilterFactory`, and `FilterPipeline`. Done.
- `Identity` filter. Done.

Phase 2 intentionally keeps runtime dependencies small. The YAML loader is strict
and supports the project configuration shape currently used by `config/default.yaml`.

## Phase 3

- `Grayscale` filter. Done.
- `Negative` filter. Done.
- `Sepia` filter. Done.
- Configurable YAML entries. Done.
- Unit coverage for RGBA/BGR behavior. Done.

The filters are CPU baselines for packed 8-bit frames. GPU implementations and
YUV/NV12/P010-specific paths belong to later FFmpeg/GPU phases.

## Phase 4

- `BoxBlur` filter. Done.
- `GaussianBlur` filter. Done.
- `Sharpen` filter. Done.
- `Sobel` filter. Done.
- YAML parameters for `radius` and `amount`. Done.
- Unit coverage for deterministic RGB behavior. Done.

The filters remain CPU baselines and are disabled in `config/default.yaml` so the
default executable path stays lightweight until capture/decoder/output modules are
introduced.

## Phase 5

- `Gamma` filter. Done.
- `Exposure` filter. Done.
- `Contrast` filter. Done.
- `Brightness` filter. Done.
- `Saturation` filter. Done.
- `WhiteBalance` filter. Done.
- `Temperature` filter. Done.
- `Tint` filter. Done.
- YAML parameters for color correction. Done.
- Unit coverage for deterministic RGBA behavior. Done.

These filters are CPU baselines for packed 8-bit `RGB`, `BGR`, and `RGBA`
frames. Later GPU phases can replace the implementation behind the same filter
interfaces.

## Phase 6

- `TextOverlay` filter. Done.
- `FpsOverlay` filter. Done.
- `TimestampOverlay` filter. Done.
- `Histogram` filter. Done.
- YAML parameters for overlay placement and histogram rendering. Done.
- Unit coverage for deterministic overlay pixels and histogram metadata. Done.

The overlay implementation uses a small built-in bitmap font so the phase remains
portable and dependency-light. A future renderer can replace it behind the same
filter interfaces.

Later phases add filters, GPU acceleration, virtual camera output, metrics,
benchmarks, and release documentation.
