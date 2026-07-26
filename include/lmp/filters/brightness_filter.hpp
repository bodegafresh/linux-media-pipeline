#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class BrightnessFilter final : public IVideoFilter {
public:
  explicit BrightnessFilter(double offset);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double offset() const noexcept;

private:
  double offset_;
};

} // namespace lmp::filters
