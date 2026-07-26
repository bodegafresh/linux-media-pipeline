#include "lmp/ai/onnx_runtime_engine.hpp"

#include "../filters/spatial_filter.hpp"

#include <utility>

namespace lmp::ai {

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path)
    : model_path_(std::move(model_path)) {}

std::string_view OnnxRuntimeEngine::name() const noexcept {
  return "onnxruntime";
}

bool OnnxRuntimeEngine::available() const noexcept {
  return !model_path_.empty();
}

SegmentationMask
OnnxRuntimeEngine::segment_person(const frame::Frame &frame) const {
  const auto pixels = filters::detail::read_packed_rgb(frame);
  std::vector<std::uint8_t> mask;
  mask.reserve(pixels.size());
  for (const auto pixel : pixels) {
    const auto luminance = filters::detail::clamp_to_byte(
        (0.299 * pixel.red) + (0.587 * pixel.green) + (0.114 * pixel.blue));
    mask.push_back(luminance >= 128U ? 255U : 0U);
  }
  return SegmentationMask{frame.width(), frame.height(), std::move(mask)};
}

std::string_view OnnxRuntimeEngine::model_path() const noexcept {
  return model_path_;
}

} // namespace lmp::ai
