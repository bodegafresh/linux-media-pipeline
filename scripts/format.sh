#!/usr/bin/env bash
set -euo pipefail

files=$(find include src tests benchmarks -type f \( -name '*.hpp' -o -name '*.cpp' \))
if [[ -n "${files}" ]]; then
  clang-format -i ${files}
fi
