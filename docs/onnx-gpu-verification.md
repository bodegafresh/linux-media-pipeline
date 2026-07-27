# ONNX GPU Verification

The conservative verification command is:

```bash
./scripts/verify-onnx-gpu.sh migraphx config/presets/ai-background-blur-migraphx.yaml
```

It writes:

```text
artifacts/onnx-gpu-verification.json
```

Current verification performs:

- real model session creation;
- warm-up runs;
- measured inference runs;
- output mask sanity checks;
- provider/fallback reporting.

It intentionally does not set `gpu_execution_verified=true` yet. That flag
requires all of:

- `MIGraphXExecutionProvider` selected by ONNX Runtime;
- real inference success;
- evidence of AMD GPU activity during a standalone inference benchmark;
- CPU versus MIGraphX output comparison;
- graph coverage/profile evidence that the model is not mostly falling back to
  CPU.

Until those checks exist and pass, a successful provider request is not enough
to claim GPU inference.
