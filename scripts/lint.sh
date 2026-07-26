#!/usr/bin/env bash
set -euo pipefail

cmake --preset dev
cmake --build --preset dev
cppcheck --enable=warning,style,performance,portability --error-exitcode=1 \
  --std=c++20 \
  --suppress=missingIncludeSystem \
  include src tests
