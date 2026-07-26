#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class WhiteBalanceFilter final : public IVideoFilter {
public:
  WhiteBalanceFilter(double red_gain, double green_gain, double blue_gain);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double red_gain() const noexcept;
  [[nodiscard]] double green_gain() const noexcept;
  [[nodiscard]] double blue_gain() const noexcept;

private:
  double red_gain_;
  double green_gain_;
  double blue_gain_;
};

} // namespace lmp::filters
