#pragma once

#include "lmp/filters/video_filter.hpp"

#include <cstdint>

namespace lmp::filters {

class GaussianBlurFilter final : public IVideoFilter {
public:
  explicit GaussianBlurFilter(std::uint32_t radius);

  void process(frame::Frame &frame) const override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] std::uint32_t radius() const noexcept;

private:
  std::uint32_t radius_;
};

} // namespace lmp::filters
