#pragma once

#include "lmp/gpu/gpu_buffer.hpp"

#include <cstddef>
#include <vector>

namespace lmp::gpu {

class BufferPool {
public:
  BufferPool(std::size_t buffer_size, std::size_t capacity);

  [[nodiscard]] GpuBuffer acquire();
  void release(GpuBuffer buffer);

  [[nodiscard]] std::size_t buffer_size() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t available() const noexcept;

private:
  std::size_t buffer_size_;
  std::size_t capacity_;
  std::vector<GpuBuffer> available_;
};

} // namespace lmp::gpu
