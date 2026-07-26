#pragma once

#include "lmp/filters/video_filter.hpp"

#include <string>

namespace lmp::filters {

class ColorAdjustFilter final : public IVideoFilter {
public:
  ColorAdjustFilter(double brightness, double contrast, double saturation);
  ColorAdjustFilter(double brightness, double contrast, double saturation,
                    std::string backend);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  void process_cpu(frame::Frame &frame) const;
  [[nodiscard]] bool process_opencl(frame::Frame &frame) const;

  double brightness_;
  double contrast_;
  double saturation_;
  std::string backend_;
};

} // namespace lmp::filters
