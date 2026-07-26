#include "lmp/filters/box_blur_filter.hpp"

#include "spatial_filter.hpp"

#include <stdexcept>

namespace lmp::filters {

BoxBlurFilter::BoxBlurFilter(std::uint32_t radius) : radius_(radius) {
  if (radius_ == 0) {
    throw std::invalid_argument("box blur radius must be >= 1");
  }
}

void BoxBlurFilter::process(frame::Frame &frame) const {
  detail::apply_box_blur(frame, radius_);
}

std::string_view BoxBlurFilter::type() const noexcept { return "box_blur"; }

std::uint32_t BoxBlurFilter::radius() const noexcept { return radius_; }

} // namespace lmp::filters
