#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this script is for macOS only." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: Homebrew is required.

Install Homebrew from https://brew.sh and rerun:
  ./scripts/install-macos-deps.sh
EOF
  exit 1
fi

brew update
brew install \
  cmake \
  ninja \
  pkg-config \
  ffmpeg \
  onnxruntime \
  cppcheck \
  llvm

cat <<'EOF'

macOS dependencies installed.

Validate the portable parts of the project with:
  ./scripts/build.sh
  ./scripts/test.sh
  ./build/dev/lmp --list-onnx-providers

Notes:
  - V4L2 loopback output is Linux-only, so live OBS/Meet virtual camera output
    is still validated on Fedora.
  - ROCm is not available on macOS. ONNX validation on Mac uses the providers
    shipped by Homebrew's onnxruntime package.
EOF
