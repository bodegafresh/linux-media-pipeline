#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class GammaFilter final : public IVideoFilter {
public:
  explicit GammaFilter(double gamma);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double gamma() const noexcept;

private:
  double gamma_;
};

} // namespace lmp::filters
