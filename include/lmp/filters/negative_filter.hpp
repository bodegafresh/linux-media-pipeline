#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class NegativeFilter final : public IVideoFilter {
public:
  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
};

} // namespace lmp::filters
