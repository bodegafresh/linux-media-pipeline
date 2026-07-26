#include "lmp/filters/tint_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

TintFilter::TintFilter(double amount) : amount_(amount) {}

void TintFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    pixel.green = detail::clamp_to_byte(pixel.green + amount_);
  });
}

std::string_view TintFilter::type() const noexcept { return "tint"; }

double TintFilter::amount() const noexcept { return amount_; }

} // namespace lmp::filters
