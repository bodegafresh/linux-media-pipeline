#include "lmp/filters/histogram_filter.hpp"

#include "overlay_renderer.hpp"
#include "spatial_filter.hpp"

#include <algorithm>
#include <array>
#include <sstream>

namespace lmp::filters {

namespace {

std::uint8_t luminance(detail::RgbPixel pixel) noexcept {
  return detail::clamp_to_byte((0.299 * pixel.red) + (0.587 * pixel.green) +
                               (0.114 * pixel.blue));
}

} // namespace

HistogramFilter::HistogramFilter(bool draw_overlay, std::uint32_t x,
                                 std::uint32_t y, std::uint32_t width,
                                 std::uint32_t height)
    : draw_overlay_(draw_overlay), x_(x), y_(y), width_(width),
      height_(height) {}

void HistogramFilter::process(frame::Frame &frame) const {
  constexpr auto bins_count = std::size_t{16};
  std::array<std::uint32_t, bins_count> bins{};
  const auto pixels = detail::read_packed_rgb(frame);
  for (const auto pixel : pixels) {
    const auto bin = static_cast<std::size_t>(luminance(pixel)) / bins_count;
    ++bins[std::min(bin, bins_count - 1U)];
  }

  std::ostringstream serialized;
  for (std::size_t i = 0; i < bins.size(); ++i) {
    if (i != 0U) {
      serialized << ',';
    }
    serialized << bins[i];
  }
  frame.metadata()["histogram.luma16"] = serialized.str();

  if (!draw_overlay_ || width_ == 0U || height_ == 0U) {
    return;
  }

  const auto max_bin = *std::max_element(bins.begin(), bins.end());
  if (max_bin == 0U) {
    return;
  }

  for (std::uint32_t i = 0; i < bins.size(); ++i) {
    const auto bar_x =
        x_ + ((i * width_) / static_cast<std::uint32_t>(bins.size()));
    const auto next_x =
        x_ + (((i + 1U) * width_) / static_cast<std::uint32_t>(bins.size()));
    const auto bar_width = std::max<std::uint32_t>(1U, next_x - bar_x);
    const auto bar_height =
        static_cast<std::uint32_t>((bins[i] * height_) / max_bin);
    for (std::uint32_t dx = 0; dx < bar_width; ++dx) {
      for (std::uint32_t dy = 0; dy < bar_height; ++dy) {
        detail::set_pixel(frame, bar_x + dx, y_ + (height_ - 1U - dy),
                          detail::RgbColor{0U, 180U, 255U});
      }
    }
  }
}

std::string_view HistogramFilter::type() const noexcept { return "histogram"; }

} // namespace lmp::filters
