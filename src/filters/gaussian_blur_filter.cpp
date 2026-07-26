#include "lmp/filters/gaussian_blur_filter.hpp"

#include "spatial_filter.hpp"

#include <array>
#include <stdexcept>

namespace lmp::filters {

GaussianBlurFilter::GaussianBlurFilter(std::uint32_t radius) : radius_(radius) {
  if (radius_ == 0) {
    throw std::invalid_argument("gaussian blur radius must be >= 1");
  }
}

void GaussianBlurFilter::process(frame::Frame &frame) const {
  constexpr auto kernel =
      std::array<double, 9>{1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0};
  for (std::uint32_t pass = 0; pass < radius_; ++pass) {
    detail::apply_kernel_3x3(frame, kernel, 16.0, 0.0);
  }
}

std::string_view GaussianBlurFilter::type() const noexcept {
  return "gaussian_blur";
}

std::uint32_t GaussianBlurFilter::radius() const noexcept { return radius_; }

} // namespace lmp::filters
