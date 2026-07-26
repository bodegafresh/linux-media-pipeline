#pragma once

#include "lmp/gpu/gpu_buffer.hpp"

#include <memory>
#include <string_view>

namespace lmp::gpu {

class IGpuBackend {
public:
  virtual ~IGpuBackend() = default;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual bool available() const noexcept = 0;
  [[nodiscard]] virtual bool
  supports_zero_copy_host_memory() const noexcept = 0;
  [[nodiscard]] virtual std::unique_ptr<GpuBuffer>
  create_buffer(std::size_t size) const = 0;
  [[nodiscard]] virtual std::unique_ptr<GpuBuffer>
  import_host_buffer(std::span<std::uint8_t> memory) const = 0;
};

} // namespace lmp::gpu
