#include "lmp/filters/exposure_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <cmath>

namespace lmp::filters {

ExposureFilter::ExposureFilter(double stops) : stops_(stops) {}

void ExposureFilter::process(frame::Frame &frame) const {
  const auto factor = std::pow(2.0, stops_);
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(pixel.red * factor);
    pixel.green = detail::clamp_to_byte(pixel.green * factor);
    pixel.blue = detail::clamp_to_byte(pixel.blue * factor);
  });
}

std::string_view ExposureFilter::type() const noexcept { return "exposure"; }

double ExposureFilter::stops() const noexcept { return stops_; }

} // namespace lmp::filters
