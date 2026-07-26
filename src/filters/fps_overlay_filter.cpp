#include "lmp/filters/fps_overlay_filter.hpp"

#include "overlay_renderer.hpp"

#include <sstream>

namespace lmp::filters {

FpsOverlayFilter::FpsOverlayFilter(double fps, std::uint32_t x, std::uint32_t y)
    : fps_(fps), x_(x), y_(y) {}

void FpsOverlayFilter::process(frame::Frame &frame) const {
  std::ostringstream label;
  label.precision(1);
  label << std::fixed << "FPS:" << fps_;
  detail::draw_text(frame, x_, y_, label.str(), detail::RgbColor{0U, 255U, 0U});
}

std::string_view FpsOverlayFilter::type() const noexcept {
  return "fps_overlay";
}

} // namespace lmp::filters
