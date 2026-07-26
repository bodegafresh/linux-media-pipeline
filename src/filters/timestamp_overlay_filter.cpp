#include "lmp/filters/timestamp_overlay_filter.hpp"

#include "overlay_renderer.hpp"

#include <chrono>
#include <sstream>

namespace lmp::filters {

TimestampOverlayFilter::TimestampOverlayFilter(std::uint32_t x, std::uint32_t y)
    : x_(x), y_(y) {}

void TimestampOverlayFilter::process(frame::Frame &frame) const {
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          frame.timestamp().time_since_epoch())
          .count();
  std::ostringstream label;
  label << "TS:" << milliseconds;
  detail::draw_text(frame, x_, y_, label.str(),
                    detail::RgbColor{255U, 255U, 0U});
}

std::string_view TimestampOverlayFilter::type() const noexcept {
  return "timestamp_overlay";
}

} // namespace lmp::filters
