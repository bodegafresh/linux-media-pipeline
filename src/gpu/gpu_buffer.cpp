#include "lmp/gpu/gpu_buffer.hpp"

#include <stdexcept>
#include <utility>

namespace lmp::gpu {

GpuBuffer::GpuBuffer(std::shared_ptr<std::vector<std::uint8_t>> owned,
                     std::span<std::uint8_t> borrowed, MemoryLocation location)
    : owned_(std::move(owned)), borrowed_(borrowed), location_(location) {
  if (host_span().empty()) {
    throw std::invalid_argument("gpu buffer size must be non-zero");
  }
}

GpuBuffer GpuBuffer::allocate_host(std::size_t size) {
  if (size == 0U) {
    throw std::invalid_argument("gpu buffer size must be non-zero");
  }
  auto memory = std::make_shared<std::vector<std::uint8_t>>(size);
  return GpuBuffer{memory, std::span<std::uint8_t>{*memory},
                   MemoryLocation::Host};
}

GpuBuffer GpuBuffer::wrap_host(std::span<std::uint8_t> memory) {
  if (memory.empty()) {
    throw std::invalid_argument("gpu buffer size must be non-zero");
  }
  return GpuBuffer{nullptr, memory, MemoryLocation::Shared};
}

std::size_t GpuBuffer::size() const noexcept { return host_span().size(); }

MemoryLocation GpuBuffer::location() const noexcept { return location_; }

bool GpuBuffer::owns_memory() const noexcept { return owned_ != nullptr; }

bool GpuBuffer::zero_copy_capable() const noexcept {
  return location_ == MemoryLocation::Shared && !owns_memory();
}

std::span<std::uint8_t> GpuBuffer::host_span() noexcept {
  if (owned_ != nullptr) {
    return std::span<std::uint8_t>{*owned_};
  }
  return borrowed_;
}

std::span<const std::uint8_t> GpuBuffer::host_span() const noexcept {
  if (owned_ != nullptr) {
    return std::span<const std::uint8_t>{*owned_};
  }
  return borrowed_;
}

} // namespace lmp::gpu
