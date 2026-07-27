#!/usr/bin/env bash
set -euo pipefail

binary="${1:-build/dev/lmp}"
out_dir="artifacts/runtime-linkage"
mkdir -p "${out_dir}"

if [[ ! -x "${binary}" ]]; then
  echo "ERROR: binary not found or not executable: ${binary}" >&2
  echo "Run ./scripts/build.sh first, or pass the binary path." >&2
  exit 1
fi

ldd "${binary}" > "${out_dir}/ldd.txt" 2>&1 || true
readelf -d "${binary}" > "${out_dir}/readelf-dynamic.txt" 2>&1 || true

find /usr /usr/local /opt "${HOME}/.local" \
  \( -name 'libonnxruntime.so*' -o -name 'libmigraphx.so*' \) \
  2>/dev/null | sort > "${out_dir}/duplicate-runtimes.txt" || true

echo "Wrote runtime linkage report to ${out_dir}"
