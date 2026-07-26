#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>
#include <string>

namespace lmp::filters {

class TextOverlayFilter final : public IVideoFilter {
public:
  TextOverlayFilter(std::string text, std::uint32_t x, std::uint32_t y);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  std::string text_;
  std::uint32_t x_;
  std::uint32_t y_;
};

} // namespace lmp::filters
