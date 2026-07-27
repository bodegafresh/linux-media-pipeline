#include "lmp/ai/onnx_runtime_engine.hpp"

#include "../filters/spatial_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#if LMP_HAS_ONNXRUNTIME
#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#error "ONNX Runtime C++ API header was not found"
#endif
#endif

namespace lmp::ai {
namespace {

SegmentationMask fallback_segment_person(const frame::Frame &frame) {
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

#if LMP_HAS_ONNXRUNTIME
std::int64_t dimension_or(std::int64_t value, std::int64_t fallback) noexcept {
  return value > 0 ? value : fallback;
}
#endif

} // namespace

class OnnxRuntimeEngine::Impl {
public:
#if LMP_HAS_ONNXRUNTIME
  explicit Impl(const std::string &model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "linux-media-pipeline"),
        session_options_{}, allocator_{}, input_shape_{1, 3, 256, 256} {
    if (!std::filesystem::exists(model_path)) {
      return;
    }

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_.emplace(env_, model_path.c_str(), session_options_);

    input_name_ = session_->GetInputNameAllocated(0, allocator_);
    output_name_ = session_->GetOutputNameAllocated(0, allocator_);

    const auto input_info =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto shape = input_info.GetShape();
    if (shape.size() == 4U) {
      input_shape_[0] = dimension_or(shape[0], 1);
      input_shape_[1] = dimension_or(shape[1], 3);
      input_shape_[2] = dimension_or(shape[2], 256);
      input_shape_[3] = dimension_or(shape[3], 256);
    }

    ready_ = input_shape_[0] == 1 && input_shape_[1] == 3 &&
             input_shape_[2] > 0 && input_shape_[3] > 0;
  }

  [[nodiscard]] bool ready() const noexcept { return ready_; }

  [[nodiscard]] SegmentationMask
  segment_person(const frame::Frame &frame) const {
    if (!ready_ || !session_.has_value()) {
      return fallback_segment_person(frame);
    }

    const auto input_height = static_cast<std::uint32_t>(input_shape_[2]);
    const auto input_width = static_cast<std::uint32_t>(input_shape_[3]);
    const auto source = filters::detail::read_packed_rgb(frame);
    std::vector<float> input(static_cast<std::size_t>(3U) * input_width *
                             input_height);

    const auto plane_size = static_cast<std::size_t>(input_width) *
                            static_cast<std::size_t>(input_height);
    for (std::uint32_t y = 0; y < input_height; ++y) {
      const auto source_y = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(y) * frame.height()) / input_height);
      for (std::uint32_t x = 0; x < input_width; ++x) {
        const auto source_x = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(x) * frame.width()) / input_width);
        const auto pixel = source[filters::detail::pixel_index(
            source_x, source_y, frame.width())];
        const auto index = filters::detail::pixel_index(x, y, input_width);
        input[index] = static_cast<float>(pixel.red) / 255.0F;
        input[plane_size + index] = static_cast<float>(pixel.green) / 255.0F;
        input[(2U * plane_size) + index] =
            static_cast<float>(pixel.blue) / 255.0F;
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_shape_.data(),
        input_shape_.size());
    const char *input_names[] = {input_name_.get()};
    const char *output_names[] = {output_name_.get()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor,
                                 1, output_names, 1);
    if (outputs.empty() || !outputs.front().IsTensor()) {
      return fallback_segment_person(frame);
    }

    const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    const auto *output = outputs.front().GetTensorData<float>();
    const auto output_count = output_info.GetElementCount();
    if (output == nullptr || output_count == 0U) {
      return fallback_segment_person(frame);
    }

    auto mask_width = input_width;
    auto mask_height = input_height;
    if (output_shape.size() >= 2U) {
      mask_height = static_cast<std::uint32_t>(
          dimension_or(output_shape[output_shape.size() - 2U], input_height));
      mask_width = static_cast<std::uint32_t>(
          dimension_or(output_shape[output_shape.size() - 1U], input_width));
    }
    if (static_cast<std::size_t>(mask_width) * mask_height > output_count) {
      return fallback_segment_person(frame);
    }

    std::vector<std::uint8_t> mask;
    mask.reserve(static_cast<std::size_t>(frame.width()) * frame.height());
    for (std::uint32_t y = 0; y < frame.height(); ++y) {
      const auto mask_y = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(y) * mask_height) / frame.height());
      for (std::uint32_t x = 0; x < frame.width(); ++x) {
        const auto mask_x = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(x) * mask_width) / frame.width());
        const auto value =
            output[filters::detail::pixel_index(mask_x, mask_y, mask_width)];
        mask.push_back(value >= 0.5F ? 255U : 0U);
      }
    }
    return SegmentationMask{frame.width(), frame.height(), std::move(mask)};
  }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::optional<Ort::Session> session_;
  Ort::AllocatedStringPtr input_name_{nullptr};
  Ort::AllocatedStringPtr output_name_{nullptr};
  std::array<std::int64_t, 4> input_shape_;
  bool ready_ = false;
#else
  explicit Impl(const std::string &) {}
  [[nodiscard]] bool ready() const noexcept { return false; }
  [[nodiscard]] SegmentationMask
  segment_person(const frame::Frame &frame) const {
    return fallback_segment_person(frame);
  }
#endif
};

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_)) {}

OnnxRuntimeEngine::~OnnxRuntimeEngine() = default;

OnnxRuntimeEngine::OnnxRuntimeEngine(OnnxRuntimeEngine &&) noexcept = default;

OnnxRuntimeEngine &
OnnxRuntimeEngine::operator=(OnnxRuntimeEngine &&) noexcept = default;

std::string_view OnnxRuntimeEngine::name() const noexcept {
  return "onnxruntime";
}

bool OnnxRuntimeEngine::available() const noexcept {
  return impl_ != nullptr && impl_->ready();
}

SegmentationMask
OnnxRuntimeEngine::segment_person(const frame::Frame &frame) const {
  return impl_->segment_person(frame);
}

std::string_view OnnxRuntimeEngine::model_path() const noexcept {
  return model_path_;
}

} // namespace lmp::ai
