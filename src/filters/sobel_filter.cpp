#include "lmp/filters/sobel_filter.hpp"

#include "spatial_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lmp::filters {

namespace {

std::uint8_t luminance(detail::RgbPixel pixel) noexcept {
  return detail::clamp_to_byte((0.299 * pixel.red) + (0.587 * pixel.green) +
                               (0.114 * pixel.blue));
}

} // namespace

void SobelFilter::process(frame::Frame &frame) const {
  const auto source = detail::read_packed_rgb(frame);
  std::vector<detail::RgbPixel> output(source.size());

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto sample = [&](int offset_x, int offset_y) {
        const auto sample_x = detail::clamp_coordinate(
            static_cast<int>(x) + offset_x, frame.width());
        const auto sample_y = detail::clamp_coordinate(
            static_cast<int>(y) + offset_y, frame.height());
        return static_cast<int>(luminance(
            source[detail::pixel_index(sample_x, sample_y, frame.width())]));
      };

      const auto gx = -sample(-1, -1) + sample(1, -1) - (2 * sample(-1, 0)) +
                      (2 * sample(1, 0)) - sample(-1, 1) + sample(1, 1);
      const auto gy = -sample(-1, -1) - (2 * sample(0, -1)) - sample(1, -1) +
                      sample(-1, 1) + (2 * sample(0, 1)) + sample(1, 1);
      const auto magnitude = detail::clamp_to_byte(
          std::sqrt(static_cast<double>((gx * gx) + (gy * gy))));

      output[detail::pixel_index(x, y, frame.width())] =
          detail::RgbPixel{magnitude, magnitude, magnitude};
    }
  }

  detail::write_packed_rgb(frame, output);
}

std::string_view SobelFilter::type() const noexcept { return "sobel"; }

} // namespace lmp::filters
