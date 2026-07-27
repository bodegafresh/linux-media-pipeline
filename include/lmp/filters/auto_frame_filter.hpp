#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>
#include <optional>

namespace lmp::filters {

class AutoFrameFilter final : public IVideoFilter {
public:
  AutoFrameFilter(double target_fill, double max_zoom,
                  std::uint8_t foreground_threshold);
  AutoFrameFilter(double target_fill, double max_zoom,
                  std::uint8_t foreground_threshold, double smoothing,
                  double dead_zone);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;

private:
  struct Crop {
    double x;
    double y;
    double width;
    double height;
  };

  [[nodiscard]] Crop smooth_crop(Crop desired) const;

  double target_fill_;
  double max_zoom_;
  std::uint8_t foreground_threshold_;
  double smoothing_;
  double dead_zone_;
  mutable std::optional<Crop> previous_crop_;
};

} // namespace lmp::filters
