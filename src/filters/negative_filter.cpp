#include "lmp/filters/negative_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

void NegativeFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [](detail::PixelChannels pixel) {
    pixel.red = static_cast<std::uint8_t>(255U - pixel.red);
    pixel.green = static_cast<std::uint8_t>(255U - pixel.green);
    pixel.blue = static_cast<std::uint8_t>(255U - pixel.blue);
  });
}

std::string_view NegativeFilter::type() const noexcept { return "negative"; }

} // namespace lmp::filters
