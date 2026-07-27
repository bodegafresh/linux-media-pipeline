#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts
sudo dnf install \
  onnxruntime-rocm \
  onnxruntime-rocm-devel \
  migraphx \
  migraphx-devel \
  --assumeno | tee artifacts/dnf-migraphx-dry-run.txt
