#include "lmp/gpu/vulkan_backend.hpp"

#include <memory>

namespace lmp::gpu {

std::string_view VulkanBackend::name() const noexcept { return "vulkan"; }

bool VulkanBackend::available() const noexcept { return false; }

bool VulkanBackend::supports_zero_copy_host_memory() const noexcept {
  return true;
}

std::unique_ptr<GpuBuffer>
VulkanBackend::create_buffer(std::size_t size) const {
  return std::make_unique<GpuBuffer>(GpuBuffer::allocate_host(size));
}

std::unique_ptr<GpuBuffer>
VulkanBackend::import_host_buffer(std::span<std::uint8_t> memory) const {
  return std::make_unique<GpuBuffer>(GpuBuffer::wrap_host(memory));
}

} // namespace lmp::gpu
