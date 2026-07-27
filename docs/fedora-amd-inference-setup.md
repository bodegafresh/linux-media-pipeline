# Fedora AMD Inference Setup

This workstation protection policy applies before any MIGraphX/ONNX Runtime ROCm
package change.

Safe diagnostics:

```bash
./scripts/snapshot-workstation.sh
./scripts/check-amd-inference-stack.sh
./scripts/check-onnx-providers.sh
./scripts/check-runtime-linkage.sh
```

Dry run only:

```bash
./scripts/dnf-migraphx-dry-run.sh
```

Stop if DNF proposes removing, replacing, or downgrading Mesa, Vulkan loader,
OpenCL loader, FFmpeg, OBS, GCC, Clang, kernel packages, or anything required by
the current graphics stack.

Never use `--allowerasing` or `--skip-broken` for this work.

The initial valid outcome may be that Fedora packages are present but the active
ONNX Runtime build still enumerates only `CPUExecutionProvider`.
