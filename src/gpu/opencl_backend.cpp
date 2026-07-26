#include "lmp/gpu/opencl_backend.hpp"

#include <memory>

#if LMP_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace lmp::gpu {

std::string_view OpenClBackend::name() const noexcept { return "opencl"; }

bool OpenClBackend::available() const noexcept {
#if LMP_HAS_OPENCL
  cl_uint platform_count = 0;
  if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS ||
      platform_count == 0U) {
    return false;
  }
  return true;
#else
  return false;
#endif
}

bool OpenClBackend::supports_zero_copy_host_memory() const noexcept {
  return true;
}

std::unique_ptr<GpuBuffer>
OpenClBackend::create_buffer(std::size_t size) const {
  return std::make_unique<GpuBuffer>(GpuBuffer::allocate_host(size));
}

std::unique_ptr<GpuBuffer>
OpenClBackend::import_host_buffer(std::span<std::uint8_t> memory) const {
  return std::make_unique<GpuBuffer>(GpuBuffer::wrap_host(memory));
}

} // namespace lmp::gpu
