#!/usr/bin/env bash
set -euo pipefail

out_dir="artifacts/workstation-baseline"
mkdir -p "${out_dir}"

rpm -qa | sort > "${out_dir}/packages.txt"

clinfo > "${out_dir}/clinfo.txt" 2>&1 || true
rocminfo > "${out_dir}/rocminfo.txt" 2>&1 || true
vulkaninfo --summary > "${out_dir}/vulkan.txt" 2>&1 || true
ffmpeg -version > "${out_dir}/ffmpeg.txt" 2>&1 || true
cmake --version > "${out_dir}/cmake.txt" 2>&1 || true
gcc --version > "${out_dir}/gcc.txt" 2>&1 || true
conan --version > "${out_dir}/conan.txt" 2>&1 || true

ldconfig -p 2>/dev/null | grep -Ei 'onnx|migraphx|hip|hsa|rocblas|miopen' \
  > "${out_dir}/gpu-libraries.txt" || true

echo "Wrote workstation snapshot to ${out_dir}"
