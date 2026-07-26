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

Later phases add filters, GPU acceleration, virtual camera output, metrics,
benchmarks, and release documentation.
