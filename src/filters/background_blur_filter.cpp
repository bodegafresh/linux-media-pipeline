#include "lmp/filters/background_blur_filter.hpp"

#include "lmp/ai/onnx_runtime_engine.hpp"

#include "spatial_filter.hpp"

#include <stdexcept>
#include <vector>

namespace lmp::filters {

BackgroundBlurFilter::BackgroundBlurFilter(std::uint32_t radius,
                                           std::uint8_t foreground_threshold)
    : radius_(radius), foreground_threshold_(foreground_threshold) {
  if (radius_ == 0U) {
    throw std::invalid_argument("background blur radius must be >= 1");
  }
}

void BackgroundBlurFilter::process(frame::Frame &frame) const {
  const auto original = detail::read_packed_rgb(frame);
  ai::OnnxRuntimeEngine fallback_engine{""};
  const auto mask = fallback_engine.segment_person(frame);

  detail::apply_box_blur(frame, radius_);
  const auto blurred = detail::read_packed_rgb(frame);
  std::vector<detail::RgbPixel> output;
  output.reserve(original.size());

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto index = detail::pixel_index(x, y, frame.width());
      output.push_back(mask.at(x, y) >= foreground_threshold_ ? original[index]
                                                              : blurred[index]);
    }
  }

  detail::write_packed_rgb(frame, output);
}

std::string_view BackgroundBlurFilter::type() const noexcept {
  return "background_blur";
}

} // namespace lmp::filters
