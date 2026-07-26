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

Later phases add filters, GPU acceleration, virtual camera output, metrics,
benchmarks, and release documentation.
