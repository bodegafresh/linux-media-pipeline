#include "lmp/ai/segmentation_mask.hpp"

#include <algorithm>
#include <cmath>
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

SegmentationMask SegmentationMask::blend_with(const SegmentationMask &previous,
                                              double previous_weight) const {
  if (width_ != previous.width_ || height_ != previous.height_) {
    throw std::invalid_argument("cannot blend masks with different dimensions");
  }
  if (previous_weight < 0.0 || previous_weight > 1.0) {
    throw std::invalid_argument("mask blend weight must be in [0, 1]");
  }

  const auto current_weight = 1.0 - previous_weight;
  std::vector<std::uint8_t> blended;
  blended.reserve(values_.size());
  for (std::size_t index = 0; index < values_.size(); ++index) {
    const auto value =
        (static_cast<double>(previous.values_[index]) * previous_weight) +
        (static_cast<double>(values_[index]) * current_weight);
    blended.push_back(static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(std::round(value)), 0, 255)));
  }
  return SegmentationMask{width_, height_, std::move(blended)};
}

SegmentationMask threshold_mask(const SegmentationMask &mask,
                                std::uint8_t threshold) {
  std::vector<std::uint8_t> values;
  values.reserve(mask.values().size());
  for (const auto value : mask.values()) {
    values.push_back(value >= threshold ? 255U : 0U);
  }
  return SegmentationMask{mask.width(), mask.height(), std::move(values)};
}

} // namespace lmp::ai
