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
constexpr auto kMaxModelDimension = std::int64_t{4096};
constexpr auto kMaxTensorPixels = std::uint64_t{4096ULL * 4096ULL};

std::int64_t dimension_or(std::int64_t value, std::int64_t fallback) noexcept {
  return value > 0 ? value : fallback;
}

bool dimensions_are_sane(std::uint32_t width, std::uint32_t height) noexcept {
  if (width == 0U || height == 0U) {
    return false;
  }
  if (width > static_cast<std::uint32_t>(kMaxModelDimension) ||
      height > static_cast<std::uint32_t>(kMaxModelDimension)) {
    return false;
  }
  return static_cast<std::uint64_t>(width) * height <= kMaxTensorPixels;
}

float probability_from_model_value(float value) noexcept {
  if (value >= 0.0F && value <= 1.0F) {
    return value;
  }
  return 1.0F / (1.0F + std::exp(-value));
}
#endif

} // namespace

class OnnxRuntimeEngine::Impl {
public:
#if LMP_HAS_ONNXRUNTIME
  Impl(const std::string &model_path, std::uint32_t inference_interval,
       double mask_smoothing)
      : env_(ORT_LOGGING_LEVEL_WARNING, "linux-media-pipeline"),
        session_options_{}, allocator_{}, input_shape_{1, 3, 256, 256} {
    inference_interval_ = std::max<std::uint32_t>(1U, inference_interval);
    mask_smoothing_ = std::clamp(mask_smoothing, 0.0, 0.95);
    if (!std::filesystem::exists(model_path)) {
      return;
    }

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_.emplace(env_, model_path.c_str(), session_options_);

    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    auto output_name = session_->GetOutputNameAllocated(0, allocator_);
    input_name_ = input_name.get();
    output_name_ = output_name.get();

    const auto input_info =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto shape = input_info.GetShape();
    if (shape.size() == 4U) {
      const auto looks_channels_last =
          dimension_or(shape[3], 3) == 3 && dimension_or(shape[1], 256) != 3;
      input_shape_[0] = dimension_or(shape[0], 1);
      input_shape_[1] =
          std::min(dimension_or(shape[1], looks_channels_last ? 256 : 3),
                   kMaxModelDimension);
      input_shape_[2] =
          std::min(dimension_or(shape[2], 256), kMaxModelDimension);
      input_shape_[3] =
          std::min(dimension_or(shape[3], looks_channels_last ? 3 : 256),
                   kMaxModelDimension);
    }

    input_channels_last_ = input_shape_[3] == 3 && input_shape_[1] != 3;
    const auto input_height = static_cast<std::uint32_t>(
        input_channels_last_ ? input_shape_[1] : input_shape_[2]);
    const auto input_width = static_cast<std::uint32_t>(
        input_channels_last_ ? input_shape_[2] : input_shape_[3]);
    ready_ = input_shape_[0] == 1 &&
             ((input_channels_last_ && input_shape_[3] == 3) ||
              (!input_channels_last_ && input_shape_[1] == 3)) &&
             dimensions_are_sane(input_width, input_height);
  }

  [[nodiscard]] bool ready() const noexcept { return ready_; }

  [[nodiscard]] SegmentationMask segment_person(const frame::Frame &frame) {
    if (!ready_ || !session_.has_value()) {
      return fallback_segment_person(frame);
    }
    ++frame_index_;
    if (cached_mask_.has_value() && inference_interval_ > 1U &&
        ((frame_index_ - 1U) % inference_interval_) != 0U &&
        cached_mask_->width() == frame.width() &&
        cached_mask_->height() == frame.height()) {
      return *cached_mask_;
    }

    const auto input_height = static_cast<std::uint32_t>(
        input_channels_last_ ? input_shape_[1] : input_shape_[2]);
    const auto input_width = static_cast<std::uint32_t>(
        input_channels_last_ ? input_shape_[2] : input_shape_[3]);
    if (!dimensions_are_sane(input_width, input_height)) {
      return fallback_segment_person(frame);
    }
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
        if (input_channels_last_) {
          const auto base = index * 3U;
          input[base] = static_cast<float>(pixel.red) / 255.0F;
          input[base + 1U] = static_cast<float>(pixel.green) / 255.0F;
          input[base + 2U] = static_cast<float>(pixel.blue) / 255.0F;
        } else {
          input[index] = static_cast<float>(pixel.red) / 255.0F;
          input[plane_size + index] = static_cast<float>(pixel.green) / 255.0F;
          input[(2U * plane_size) + index] =
              static_cast<float>(pixel.blue) / 255.0F;
        }
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(), input_shape_.data(),
        input_shape_.size());
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};
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
    auto channel_count = std::uint32_t{1U};
    auto person_channel = std::uint32_t{0U};
    auto channels_last = false;
    if (output_shape.size() >= 2U) {
      mask_height = static_cast<std::uint32_t>(
          dimension_or(output_shape[output_shape.size() - 2U], input_height));
      mask_width = static_cast<std::uint32_t>(
          dimension_or(output_shape[output_shape.size() - 1U], input_width));
    }
    if (output_shape.size() == 4U) {
      const auto first_channel_dim =
          static_cast<std::uint32_t>(dimension_or(output_shape[1], 1));
      const auto last_channel_dim =
          static_cast<std::uint32_t>(dimension_or(output_shape[3], 1));
      if (first_channel_dim > 1U && first_channel_dim <= 4U) {
        channel_count = first_channel_dim;
        person_channel = std::min<std::uint32_t>(1U, channel_count - 1U);
        mask_height = static_cast<std::uint32_t>(dimension_or(
            output_shape[2], static_cast<std::int64_t>(input_height)));
        mask_width = static_cast<std::uint32_t>(dimension_or(
            output_shape[3], static_cast<std::int64_t>(input_width)));
      } else if (last_channel_dim > 1U && last_channel_dim <= 4U) {
        channel_count = last_channel_dim;
        person_channel = std::min<std::uint32_t>(1U, channel_count - 1U);
        channels_last = true;
        mask_height = static_cast<std::uint32_t>(dimension_or(
            output_shape[1], static_cast<std::int64_t>(input_height)));
        mask_width = static_cast<std::uint32_t>(dimension_or(
            output_shape[2], static_cast<std::int64_t>(input_width)));
      } else if (last_channel_dim == 1U) {
        channels_last = true;
        mask_height = static_cast<std::uint32_t>(dimension_or(
            output_shape[1], static_cast<std::int64_t>(input_height)));
        mask_width = static_cast<std::uint32_t>(dimension_or(
            output_shape[2], static_cast<std::int64_t>(input_width)));
      }
    }
    if (!dimensions_are_sane(mask_width, mask_height)) {
      return fallback_segment_person(frame);
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
        const auto mask_index =
            filters::detail::pixel_index(mask_x, mask_y, mask_width);
        const auto output_index =
            channels_last ? (mask_index * channel_count) + person_channel
                          : (static_cast<std::size_t>(person_channel) *
                             mask_width * mask_height) +
                                mask_index;
        if (output_index >= output_count) {
          return fallback_segment_person(frame);
        }
        const auto value = probability_from_model_value(output[output_index]);
        mask.push_back(value >= 0.5F ? 255U : 0U);
      }
    }
    auto current =
        SegmentationMask{frame.width(), frame.height(), std::move(mask)};
    if (cached_mask_.has_value() && cached_mask_->width() == current.width() &&
        cached_mask_->height() == current.height()) {
      current = current.blend_with(*cached_mask_, mask_smoothing_);
    }
    cached_mask_ = current;
    return current;
  }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::optional<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::array<std::int64_t, 4> input_shape_;
  std::optional<SegmentationMask> cached_mask_;
  std::uint64_t frame_index_ = 0;
  std::uint32_t inference_interval_ = 3;
  double mask_smoothing_ = 0.70;
  bool input_channels_last_ = false;
  bool ready_ = false;
#else
  Impl(const std::string &, std::uint32_t, double) {}
  [[nodiscard]] bool ready() const noexcept { return false; }
  [[nodiscard]] SegmentationMask segment_person(const frame::Frame &frame) {
    return fallback_segment_person(frame);
  }
#endif
};

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_, 3U, 0.70)) {}

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path,
                                     std::uint32_t inference_interval,
                                     double mask_smoothing)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_, inference_interval,
                                   mask_smoothing)) {}

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

SegmentationMask OnnxRuntimeEngine::segment_person(const frame::Frame &frame) {
  return impl_->segment_person(frame);
}

std::string_view OnnxRuntimeEngine::model_path() const noexcept {
  return model_path_;
}

} // namespace lmp::ai
