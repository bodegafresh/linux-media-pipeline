#include "lmp/filters/saturation_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <stdexcept>

namespace lmp::filters {

SaturationFilter::SaturationFilter(double factor) : factor_(factor) {
  if (factor_ < 0.0) {
    throw std::invalid_argument("saturation factor must be >= 0");
  }
}

void SaturationFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    const auto gray =
        (0.299 * pixel.red) + (0.587 * pixel.green) + (0.114 * pixel.blue);
    pixel.red = detail::clamp_to_byte(gray + ((pixel.red - gray) * factor_));
    pixel.green =
        detail::clamp_to_byte(gray + ((pixel.green - gray) * factor_));
    pixel.blue = detail::clamp_to_byte(gray + ((pixel.blue - gray) * factor_));
  });
}

std::string_view SaturationFilter::type() const noexcept {
  return "saturation";
}

double SaturationFilter::factor() const noexcept { return factor_; }

} // namespace lmp::filters
