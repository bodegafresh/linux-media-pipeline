#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>

namespace lmp::filters {

class BackgroundBlurFilter final : public IVideoFilter {
public:
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  std::uint32_t radius_;
  std::uint8_t foreground_threshold_;
};

} // namespace lmp::filters
