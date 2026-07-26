#include "lmp/filters/temperature_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

TemperatureFilter::TemperatureFilter(double amount) : amount_(amount) {}

void TemperatureFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(pixel.red + amount_);
    pixel.blue = detail::clamp_to_byte(pixel.blue - amount_);
  });
}

std::string_view TemperatureFilter::type() const noexcept {
  return "temperature";
}

double TemperatureFilter::amount() const noexcept { return amount_; }

} // namespace lmp::filters
