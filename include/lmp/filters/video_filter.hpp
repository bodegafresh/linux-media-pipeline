#pragma once

#include "lmp/frame/frame.hpp"

#include <string_view>

namespace lmp::filters {

class IVideoFilter {
public:
  virtual ~IVideoFilter() = default;

  virtual void process(frame::Frame &frame) const = 0;
  [[nodiscard]] virtual std::string_view type() const noexcept = 0;
};

} // namespace lmp::filters
