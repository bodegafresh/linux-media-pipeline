#!/usr/bin/env bash
set -euo pipefail

out_dir="artifacts/amd-inference-stack"
mkdir -p "${out_dir}"

{
  echo "== GPU PCI device and kernel driver =="
  lspci -nnk | grep -A4 -Ei 'VGA|Display|3D' || true
  echo
  echo "== ROCm visibility =="
  rocminfo || true
  echo
  echo "== ROCm SMI =="
  rocm-smi --showproductname --showuniqueid --showmeminfo vram --showuse \
    --showtemp || true
  echo
  echo "== Device nodes =="
  ls -l /dev/kfd /dev/dri 2>&1 || true
  echo
  echo "== User and groups =="
  id
  getent group render || true
  getent group video || true
} | tee "${out_dir}/report.txt"

echo "Wrote AMD inference stack report to ${out_dir}/report.txt"
