#include "lmp/filters/sharpen_filter.hpp"

#include "spatial_filter.hpp"

#include <array>
#include <stdexcept>

namespace lmp::filters {

SharpenFilter::SharpenFilter(double amount) : amount_(amount) {
  if (amount_ < 0.0) {
    throw std::invalid_argument("sharpen amount must be >= 0");
  }
}

void SharpenFilter::process(frame::Frame &frame) const {
  const auto kernel = std::array<double, 9>{
      0.0,      -amount_, 0.0,      -amount_, 1.0 + (4.0 * amount_),
      -amount_, 0.0,      -amount_, 0.0};
  detail::apply_kernel_3x3(frame, kernel, 1.0, 0.0);
}

std::string_view SharpenFilter::type() const noexcept { return "sharpen"; }

double SharpenFilter::amount() const noexcept { return amount_; }

} // namespace lmp::filters
