#!/usr/bin/env bash
set -euo pipefail

sudo dnf install -y \
  cmake \
  ninja-build \
  gcc-c++ \
  conan \
  git \
  clang-tools-extra \
  cppcheck \
  v4l2loopback \
  v4l-utils \
  ffmpeg \
  ffmpeg-devel \
  ocl-icd-devel \
  opencl-headers
