#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace lmp::gpu {

enum class MemoryLocation {
  Host,
  Device,
  Shared,
};

class GpuBuffer {
public:
  static GpuBuffer allocate_host(std::size_t size);
  static GpuBuffer wrap_host(std::span<std::uint8_t> memory);

  GpuBuffer(const GpuBuffer &) = delete;
  GpuBuffer &operator=(const GpuBuffer &) = delete;
  GpuBuffer(GpuBuffer &&) noexcept = default;
  GpuBuffer &operator=(GpuBuffer &&) noexcept = default;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] MemoryLocation location() const noexcept;
  [[nodiscard]] bool owns_memory() const noexcept;
  [[nodiscard]] bool zero_copy_capable() const noexcept;
  [[nodiscard]] std::span<std::uint8_t> host_span() noexcept;
  [[nodiscard]] std::span<const std::uint8_t> host_span() const noexcept;

private:
  GpuBuffer(std::shared_ptr<std::vector<std::uint8_t>> owned,
            std::span<std::uint8_t> borrowed, MemoryLocation location);

  std::shared_ptr<std::vector<std::uint8_t>> owned_;
  std::span<std::uint8_t> borrowed_;
  MemoryLocation location_;
};

} // namespace lmp::gpu
