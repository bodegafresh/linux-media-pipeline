#!/usr/bin/env bash
set -euo pipefail

with_unreal=false
if [[ "${1:-}" == "--with-unreal" ]]; then
  with_unreal=true
fi

out_dir="artifacts/workstation-regression"
mkdir -p "${out_dir}"

vulkaninfo --summary > "${out_dir}/vulkan.txt" 2>&1 || true
clinfo > "${out_dir}/clinfo.txt" 2>&1 || true
ffmpeg -version > "${out_dir}/ffmpeg.txt" 2>&1 || true
obs --version > "${out_dir}/obs.txt" 2>&1 || true
gcc --version > "${out_dir}/gcc.txt" 2>&1 || true
conan profile show > "${out_dir}/conan-profile.txt" 2>&1 || true
v4l2-ctl --list-devices > "${out_dir}/v4l2-devices.txt" 2>&1 || true

if ${with_unreal}; then
  if [[ -z "${UE_ROOT:-}" || -z "${GE_UPROJECT:-}" ]]; then
    echo "UE_ROOT and GE_UPROJECT are required for --with-unreal" \
      > "${out_dir}/unreal.txt"
  else
    "${UE_ROOT}/Engine/Build/BatchFiles/Linux/Build.sh" \
      GalacticExplorerEditor \
      Linux \
      Development \
      -Project="${GE_UPROJECT}" \
      -Progress > "${out_dir}/unreal.txt" 2>&1 || true
  fi
fi

echo "Wrote workstation regression report to ${out_dir}"
