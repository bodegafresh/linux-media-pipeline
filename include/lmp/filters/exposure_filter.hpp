#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class ExposureFilter final : public IVideoFilter {
public:
  explicit ExposureFilter(double stops);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double stops() const noexcept;

private:
  double stops_;
};

} // namespace lmp::filters
