#pragma once

#include "lmp/gpu/gpu_backend.hpp"

namespace lmp::gpu {

class VulkanBackend final : public IGpuBackend {
public:
  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] bool available() const noexcept override;
  [[nodiscard]] bool supports_zero_copy_host_memory() const noexcept override;
  [[nodiscard]] std::unique_ptr<GpuBuffer>
  create_buffer(std::size_t size) const override;
  [[nodiscard]] std::unique_ptr<GpuBuffer>
  import_host_buffer(std::span<std::uint8_t> memory) const override;
};

} // namespace lmp::gpu
