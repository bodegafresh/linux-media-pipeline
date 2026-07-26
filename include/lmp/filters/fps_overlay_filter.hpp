#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>

namespace lmp::filters {

class FpsOverlayFilter final : public IVideoFilter {
public:
  FpsOverlayFilter(double fps, std::uint32_t x, std::uint32_t y);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  double fps_;
  std::uint32_t x_;
  std::uint32_t y_;
};

} // namespace lmp::filters
