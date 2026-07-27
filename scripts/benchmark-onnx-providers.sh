#!/usr/bin/env bash
set -euo pipefail

config="${1:-config/presets/ai-background-blur-migraphx.yaml}"

./scripts/build.sh
echo "== CPU =="
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider cpu
cp artifacts/onnx-gpu-verification.json artifacts/onnx-cpu-verification.json

echo "== MIGraphX =="
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider migraphx
cp artifacts/onnx-gpu-verification.json artifacts/onnx-migraphx-verification.json
