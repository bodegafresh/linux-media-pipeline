#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>

namespace lmp::filters {

class HistogramFilter final : public IVideoFilter {
public:
  HistogramFilter(bool draw_overlay, std::uint32_t x, std::uint32_t y,
                  std::uint32_t width, std::uint32_t height);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  bool draw_overlay_;
  std::uint32_t x_;
  std::uint32_t y_;
  std::uint32_t width_;
  std::uint32_t height_;
};

} // namespace lmp::filters
