#include "lmp/filters/brightness_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

BrightnessFilter::BrightnessFilter(double offset) : offset_(offset) {}

void BrightnessFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(pixel.red + offset_);
    pixel.green = detail::clamp_to_byte(pixel.green + offset_);
    pixel.blue = detail::clamp_to_byte(pixel.blue + offset_);
  });
}

std::string_view BrightnessFilter::type() const noexcept {
  return "brightness";
}

double BrightnessFilter::offset() const noexcept { return offset_; }

} // namespace lmp::filters
