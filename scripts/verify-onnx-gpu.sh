#!/usr/bin/env bash
set -euo pipefail

provider="${1:-migraphx}"
config="${2:-config/presets/ai-background-blur-migraphx.yaml}"

./scripts/build.sh
mkdir -p artifacts
{
  echo "provider=${provider}"
  echo "config=${config}"
  echo "timestamp=$(date -Is)"
  echo
  echo "== rocm-smi before =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
  echo
  echo "== onnx verification =="
} > artifacts/onnx-gpu-activity.txt
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider "${provider}"
{
  echo
  echo "== rocm-smi after =="
  rocm-smi --showproductname --showuse --showmemuse --showtemp 2>&1 || true
} >> artifacts/onnx-gpu-activity.txt
echo "Wrote artifacts/onnx-gpu-activity.txt"
