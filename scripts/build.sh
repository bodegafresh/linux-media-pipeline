#!/usr/bin/env bash
set -euo pipefail

cmake_args=()
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]] && command -v rpm >/dev/null 2>&1; then
  onnxruntime_rocm_library="$(
    rpm -ql onnxruntime-rocm 2>/dev/null |
      grep -E '/libonnxruntime\.so(\.[0-9.]+)?$' |
      head -n 1 || true
  )"
  if [[ -n "${onnxruntime_rocm_library}" ]]; then
    onnxruntime_rocm_lib_dir="$(dirname "${onnxruntime_rocm_library}")"
    cmake_args+=("-DONNXRUNTIME_ROOT=${onnxruntime_rocm_lib_dir}/..")
  fi
fi

cmake --preset dev "${cmake_args[@]}"
cmake --build --preset dev
