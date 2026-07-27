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
  BackgroundBlurFilter(std::uint32_t radius, std::uint8_t foreground_threshold,
                       std::string backend, double brightness, double contrast,
                       double saturation, bool auto_frame, double target_fill,
                       double max_zoom, std::string mask_mode);

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
  bool auto_frame_;
  double target_fill_;
  double max_zoom_;
  std::string mask_mode_;
};

} // namespace lmp::filters
