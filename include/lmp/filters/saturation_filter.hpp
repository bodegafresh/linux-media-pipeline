#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class SaturationFilter final : public IVideoFilter {
public:
  explicit SaturationFilter(double factor);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double factor() const noexcept;

private:
  double factor_;
};

} // namespace lmp::filters
