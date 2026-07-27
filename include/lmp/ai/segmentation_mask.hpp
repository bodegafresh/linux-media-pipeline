#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lmp::ai {

class SegmentationMask {
public:
  SegmentationMask(std::uint32_t width, std::uint32_t height,
                   std::vector<std::uint8_t> values);

  [[nodiscard]] std::uint32_t width() const noexcept;
  [[nodiscard]] std::uint32_t height() const noexcept;
  [[nodiscard]] std::uint8_t at(std::uint32_t x, std::uint32_t y) const;
  [[nodiscard]] std::span<const std::uint8_t> values() const noexcept;
  [[nodiscard]] SegmentationMask blend_with(const SegmentationMask &previous,
                                            double previous_weight) const;

private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::vector<std::uint8_t> values_;
};

[[nodiscard]] SegmentationMask threshold_mask(const SegmentationMask &mask,
                                              std::uint8_t threshold);
[[nodiscard]] SegmentationMask refine_mask(const SegmentationMask &mask,
                                           std::uint8_t threshold,
                                           std::uint32_t expand_radius,
                                           std::uint32_t feather_radius);
[[nodiscard]] SegmentationMask invert_mask(const SegmentationMask &mask);
[[nodiscard]] SegmentationMask
largest_component_mask(const SegmentationMask &mask, std::uint8_t threshold);
[[nodiscard]] double mask_coverage(const SegmentationMask &mask,
                                   std::uint8_t threshold) noexcept;

} // namespace lmp::ai
