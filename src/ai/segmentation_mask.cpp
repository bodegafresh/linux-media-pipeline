#include "lmp/ai/segmentation_mask.hpp"

#include <stdexcept>

namespace lmp::ai {

SegmentationMask::SegmentationMask(std::uint32_t width, std::uint32_t height,
                                   std::vector<std::uint8_t> values)
    : width_(width), height_(height), values_(std::move(values)) {
  if (width_ == 0U || height_ == 0U) {
    throw std::invalid_argument(
        "segmentation mask dimensions must be non-zero");
  }
  const auto expected =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  if (values_.size() != expected) {
    throw std::invalid_argument(
        "segmentation mask size does not match dimensions");
  }
}

std::uint32_t SegmentationMask::width() const noexcept { return width_; }

std::uint32_t SegmentationMask::height() const noexcept { return height_; }

std::uint8_t SegmentationMask::at(std::uint32_t x, std::uint32_t y) const {
  if (x >= width_ || y >= height_) {
    throw std::out_of_range("segmentation mask coordinate out of range");
  }
  return values_[(static_cast<std::size_t>(y) * width_) + x];
}

std::span<const std::uint8_t> SegmentationMask::values() const noexcept {
  return values_;
}

} // namespace lmp::ai
