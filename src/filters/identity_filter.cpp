#include "lmp/filters/identity_filter.hpp"

namespace lmp::filters {

void IdentityFilter::process(frame::Frame &frame) const {
  static_cast<void>(frame);
}

std::string_view IdentityFilter::type() const noexcept { return "identity"; }

} // namespace lmp::filters
