#pragma once

#include "lmp/ai/segmentation_mask.hpp"
#include "lmp/frame/frame.hpp"

#include <string_view>

namespace lmp::ai {

class IInferenceEngine {
public:
  virtual ~IInferenceEngine() = default;

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual bool available() const noexcept = 0;
  [[nodiscard]] virtual SegmentationMask
  segment_person(const frame::Frame &frame) const = 0;
};

} // namespace lmp::ai
