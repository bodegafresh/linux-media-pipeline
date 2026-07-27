#pragma once

#include "lmp/ai/inference_engine.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace lmp::ai {

class OnnxRuntimeEngine final : public IInferenceEngine {
public:
  explicit OnnxRuntimeEngine(std::string model_path);
  OnnxRuntimeEngine(std::string model_path, std::uint32_t inference_interval,
                    double mask_smoothing);
  ~OnnxRuntimeEngine() override;

  OnnxRuntimeEngine(const OnnxRuntimeEngine &) = delete;
  OnnxRuntimeEngine &operator=(const OnnxRuntimeEngine &) = delete;
  OnnxRuntimeEngine(OnnxRuntimeEngine &&) noexcept;
  OnnxRuntimeEngine &operator=(OnnxRuntimeEngine &&) noexcept;

  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] bool available() const noexcept override;
  [[nodiscard]] SegmentationMask
  segment_person(const frame::Frame &frame) override;
  [[nodiscard]] std::string_view model_path() const noexcept;

private:
  class Impl;

  std::string model_path_;
  std::unique_ptr<Impl> impl_;
};

} // namespace lmp::ai
