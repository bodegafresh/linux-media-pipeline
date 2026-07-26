#include "lmp/filters/sepia_filter.hpp"

#include "packed_pixel_filter.hpp"

namespace lmp::filters {

void SepiaFilter::process(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [](detail::PixelChannels pixel) {
    const auto red = pixel.red;
    const auto green = pixel.green;
    const auto blue = pixel.blue;

    pixel.red =
        detail::clamp_to_byte((0.393 * red) + (0.769 * green) + (0.189 * blue));
    pixel.green =
        detail::clamp_to_byte((0.349 * red) + (0.686 * green) + (0.168 * blue));
    pixel.blue =
        detail::clamp_to_byte((0.272 * red) + (0.534 * green) + (0.131 * blue));
  });
}

std::string_view SepiaFilter::type() const noexcept { return "sepia"; }

} // namespace lmp::filters
