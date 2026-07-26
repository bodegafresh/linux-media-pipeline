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
  OpenCL-ICD-Loader-devel \
  opencl-headers \
  vulkan-headers \
  vulkan-loader-devel
