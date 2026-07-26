#include "lmp/gpu/buffer_pool.hpp"

#include <stdexcept>
#include <utility>

namespace lmp::gpu {

BufferPool::BufferPool(std::size_t buffer_size, std::size_t capacity)
    : buffer_size_(buffer_size), capacity_(capacity) {
  if (buffer_size_ == 0U) {
    throw std::invalid_argument("buffer pool size must be non-zero");
  }
  if (capacity_ == 0U) {
    throw std::invalid_argument("buffer pool capacity must be non-zero");
  }
  available_.reserve(capacity_);
  for (std::size_t i = 0; i < capacity_; ++i) {
    available_.push_back(GpuBuffer::allocate_host(buffer_size_));
  }
}

GpuBuffer BufferPool::acquire() {
  if (available_.empty()) {
    return GpuBuffer::allocate_host(buffer_size_);
  }
  auto buffer = std::move(available_.back());
  available_.pop_back();
  return buffer;
}

void BufferPool::release(GpuBuffer buffer) {
  if (buffer.size() != buffer_size_) {
    throw std::invalid_argument("released buffer has unexpected size");
  }
  if (available_.size() < capacity_) {
    available_.push_back(std::move(buffer));
  }
}

std::size_t BufferPool::buffer_size() const noexcept { return buffer_size_; }

std::size_t BufferPool::capacity() const noexcept { return capacity_; }

std::size_t BufferPool::available() const noexcept { return available_.size(); }

} // namespace lmp::gpu
