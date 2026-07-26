#include "lmp/filters/white_balance_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <stdexcept>

namespace lmp::filters {

WhiteBalanceFilter::WhiteBalanceFilter(double red_gain, double green_gain,
                                       double blue_gain)
    : red_gain_(red_gain), green_gain_(green_gain), blue_gain_(blue_gain) {
  if (red_gain_ < 0.0 || green_gain_ < 0.0 || blue_gain_ < 0.0) {
    throw std::invalid_argument("white balance gains must be >= 0");
  }
}

void WhiteBalanceFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.red = detail::clamp_to_byte(pixel.red * red_gain_);
    pixel.green = detail::clamp_to_byte(pixel.green * green_gain_);
    pixel.blue = detail::clamp_to_byte(pixel.blue * blue_gain_);
  });
}

std::string_view WhiteBalanceFilter::type() const noexcept {
  return "white_balance";
}

double WhiteBalanceFilter::red_gain() const noexcept { return red_gain_; }

double WhiteBalanceFilter::green_gain() const noexcept { return green_gain_; }

double WhiteBalanceFilter::blue_gain() const noexcept { return blue_gain_; }

} // namespace lmp::filters
