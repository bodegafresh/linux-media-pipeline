#!/usr/bin/env bash
set -euo pipefail

config="${1:-config/presets/ai-background-blur-migraphx.yaml}"

./scripts/build.sh
mkdir -p artifacts
echo "== CPU =="
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider cpu
cp artifacts/onnx-gpu-verification.json artifacts/onnx-cpu-verification.json

echo "== MIGraphX =="
{
  echo "config=${config}"
  echo "timestamp=$(date -Is)"
  echo
  echo "== rocm-smi before migraphx =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
} > artifacts/onnx-migraphx-gpu-activity.txt
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider migraphx
cp artifacts/onnx-gpu-verification.json artifacts/onnx-migraphx-verification.json
{
  echo
  echo "== rocm-smi after migraphx =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
} >> artifacts/onnx-migraphx-gpu-activity.txt
echo "Wrote artifacts/onnx-cpu-verification.json"
echo "Wrote artifacts/onnx-migraphx-verification.json"
echo "Wrote artifacts/onnx-migraphx-gpu-activity.txt"

echo "== ROCm =="
{
  echo "config=${config}"
  echo "timestamp=$(date -Is)"
  echo
  echo "== rocm-smi before rocm =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
} > artifacts/onnx-rocm-gpu-activity.txt
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider rocm
cp artifacts/onnx-gpu-verification.json artifacts/onnx-rocm-verification.json
{
  echo
  echo "== rocm-smi after rocm =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
} >> artifacts/onnx-rocm-gpu-activity.txt
echo "Wrote artifacts/onnx-rocm-verification.json"
echo "Wrote artifacts/onnx-rocm-gpu-activity.txt"
