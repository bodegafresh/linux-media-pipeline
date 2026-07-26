#include "lmp/filters/contrast_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <stdexcept>

namespace lmp::filters {

ContrastFilter::ContrastFilter(double factor) : factor_(factor) {
  if (factor_ < 0.0) {
    throw std::invalid_argument("contrast factor must be >= 0");
  }
}

void ContrastFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(((pixel.red - 128.0) * factor_) + 128.0);
    pixel.green =
        detail::clamp_to_byte(((pixel.green - 128.0) * factor_) + 128.0);
    pixel.blue =
        detail::clamp_to_byte(((pixel.blue - 128.0) * factor_) + 128.0);
  });
}

std::string_view ContrastFilter::type() const noexcept { return "contrast"; }

double ContrastFilter::factor() const noexcept { return factor_; }

} // namespace lmp::filters
