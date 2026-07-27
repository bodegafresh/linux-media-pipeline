#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>

namespace lmp::filters {

class AutoFrameFilter final : public IVideoFilter {
public:
  AutoFrameFilter(double target_fill, double max_zoom,
                  std::uint8_t foreground_threshold);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  double target_fill_;
  double max_zoom_;
  std::uint8_t foreground_threshold_;
};

} // namespace lmp::filters
