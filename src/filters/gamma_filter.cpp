#include "lmp/filters/gamma_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <cmath>
#include <stdexcept>

namespace lmp::filters {

GammaFilter::GammaFilter(double gamma) : gamma_(gamma) {
  if (gamma_ <= 0.0) {
    throw std::invalid_argument("gamma must be > 0");
  }
}

void GammaFilter::process(frame::Frame &frame) const {
  const auto inverse_gamma = 1.0 / gamma_;
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(
        std::pow(pixel.red / 255.0, inverse_gamma) * 255.0);
    pixel.green = detail::clamp_to_byte(
        std::pow(pixel.green / 255.0, inverse_gamma) * 255.0);
    pixel.blue = detail::clamp_to_byte(
        std::pow(pixel.blue / 255.0, inverse_gamma) * 255.0);
  });
}

std::string_view GammaFilter::type() const noexcept { return "gamma"; }

double GammaFilter::gamma() const noexcept { return gamma_; }

} // namespace lmp::filters
