#include "lmp/filters/text_overlay_filter.hpp"

#include "overlay_renderer.hpp"

#include <utility>

namespace lmp::filters {

TextOverlayFilter::TextOverlayFilter(std::string text, std::uint32_t x,
                                     std::uint32_t y)
    : text_(std::move(text)), x_(x), y_(y) {}

void TextOverlayFilter::process(frame::Frame &frame) const {
  detail::draw_text(frame, x_, y_, text_, detail::RgbColor{255U, 255U, 255U});
}

std::string_view TextOverlayFilter::type() const noexcept {
  return "text_overlay";
}

} // namespace lmp::filters
