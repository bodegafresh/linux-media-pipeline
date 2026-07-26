#pragma once

#include "lmp/ai/inference_engine.hpp"

#include <string>

namespace lmp::ai {

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
  explicit OnnxRuntimeEngine(std::string model_path);

  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] bool available() const noexcept override;
  [[nodiscard]] SegmentationMask
  segment_person(const frame::Frame &frame) const override;
  [[nodiscard]] std::string_view model_path() const noexcept;

private:
  std::string model_path_;
};

} // namespace lmp::ai
