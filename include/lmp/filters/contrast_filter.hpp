#pragma once

#include "lmp/filters/video_filter.hpp"

namespace lmp::filters {

class ContrastFilter final : public IVideoFilter {
public:
  explicit ContrastFilter(double factor);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] double factor() const noexcept;

private:
  double factor_;
};

} // namespace lmp::filters
