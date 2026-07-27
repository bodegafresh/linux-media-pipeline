#!/usr/bin/env bash
set -euo pipefail

./scripts/build.sh
./build/dev/lmp --list-onnx-providers
