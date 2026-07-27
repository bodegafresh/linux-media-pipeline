#!/usr/bin/env bash
set -euo pipefail

provider="${1:-migraphx}"
config="${2:-config/presets/ai-background-blur-migraphx.yaml}"

./scripts/build.sh
./build/dev/lmp --config "${config}" --verify-onnx-gpu --onnx-provider "${provider}"
