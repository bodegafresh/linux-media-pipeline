#include "lmp/filters/grayscale_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

void GrayscaleFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [](detail::PixelChannels pixel) {
    const auto gray = detail::clamp_to_byte(
        (0.299 * pixel.red) + (0.587 * pixel.green) + (0.114 * pixel.blue));
    pixel.red = gray;
    pixel.green = gray;
    pixel.blue = gray;
  });
}

std::string_view GrayscaleFilter::type() const noexcept { return "grayscale"; }

} // namespace lmp::filters
