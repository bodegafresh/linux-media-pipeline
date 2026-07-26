#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build/benchmarks -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLMP_BUILD_TESTS=OFF \
  -DLMP_BUILD_BENCHMARKS=ON
cmake --build build/benchmarks
cmake --build build/benchmarks --target lmp_benchmarks_placeholder
