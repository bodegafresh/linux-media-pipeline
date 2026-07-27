#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>
#include <string>

namespace lmp::filters {

class BackgroundBlurFilter final : public IVideoFilter {
public:
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold);
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend);
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend, double brightness, double contrast,
                       double saturation);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  void process_cpu(frame::Frame &frame) const;
  [[nodiscard]] bool process_opencl(frame::Frame &frame) const;

  std::uint32_t radius_;
  std::uint8_t foreground_threshold_;
  std::string backend_;
  double brightness_;
  double contrast_;
  double saturation_;
};

} // namespace lmp::filters
