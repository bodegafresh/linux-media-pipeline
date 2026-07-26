#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class TemperatureFilter final : public IVideoFilter {
public:
  explicit TemperatureFilter(double amount);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double amount() const noexcept;

private:
  double amount_;
};

} // namespace lmp::filters
