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
  OnnxRuntimeEngine(std::string model_path, std::uint32_t inference_interval,
                    double mask_smoothing, std::string input_shape,
                    std::string output_shape);
  OnnxRuntimeEngine(std::string model_path, std::uint32_t inference_interval,
                    double mask_smoothing, std::string input_shape,
                    std::string output_shape, std::string requested_provider,
                    bool allow_provider_fallback);
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
  [[nodiscard]] std::string_view last_error() const noexcept;
  [[nodiscard]] std::string_view requested_provider() const noexcept;
  [[nodiscard]] std::string_view active_provider() const noexcept;
  [[nodiscard]] std::string_view available_providers() const noexcept;
  [[nodiscard]] bool provider_fallback() const noexcept;
  [[nodiscard]] std::string_view provider_fallback_reason() const noexcept;
  [[nodiscard]] std::string_view model_summary() const noexcept;

private:
  class Impl;

  std::string model_path_;
  std::unique_ptr<Impl> impl_;
};

} // namespace lmp::ai
