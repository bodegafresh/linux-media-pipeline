#include "lmp/gpu/opencl_backend.hpp"

#include <memory>

namespace lmp::gpu {

std::string_view OpenClBackend::name() const noexcept { return "opencl"; }

bool OpenClBackend::available() const noexcept { return false; }

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
