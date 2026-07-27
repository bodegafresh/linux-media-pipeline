#include "lmp/ai/onnx_runtime_engine.hpp"

#include "../filters/spatial_filter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
  return value > 0 && value <= kMaxModelDimension ? value : fallback;
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

struct TensorShape {
  std::array<std::int64_t, 8> dimensions{};
  std::size_t rank = 0;
};

template <typename TensorInfo>
std::optional<TensorShape> read_shape(const TensorInfo &info) {
  std::size_t rank = 0;
  const auto rank_status = Ort::GetApi().GetDimensionsCount(info, &rank);
  if (rank_status != nullptr) {
    Ort::GetApi().ReleaseStatus(rank_status);
    return std::nullopt;
  }
  if (rank == 0U || rank > shape.dimensions.size()) {
    shape.rank = rank;
    return std::nullopt;
  }
  shape.rank = rank;
  const auto dimensions_status =
      Ort::GetApi().GetDimensions(info, shape.dimensions.data(), rank);
  if (dimensions_status != nullptr) {
    Ort::GetApi().ReleaseStatus(dimensions_status);
    return std::nullopt;
  }
  for (auto &dimension : shape.dimensions) {
    dimension = dimension_or(dimension, 0);
  }
  return shape;
}

std::string describe_shape(std::string_view label,
                           const std::optional<TensorShape> &shape) {
  if (!shape.has_value()) {
    return std::string{label} + "=unreadable";
  }
  auto result = std::string{label} + "_rank=" + std::to_string(shape->rank) +
                " " + std::string{label} + "_dims=[";
  for (std::size_t index = 0; index < shape->rank; ++index) {
    if (index > 0U) {
      result += "x";
    }
    result += std::to_string(shape->dimensions[index]);
  }
  result += "]";
  return result;
}

std::optional<std::size_t> channel_axis(const TensorShape &shape) {
  for (std::size_t index = 0; index < shape.rank; ++index) {
    if (dimension_or(shape.dimensions[index], 0) == 3) {
      return index;
    }
  }
  return std::nullopt;
}

std::vector<std::size_t> spatial_axes(const TensorShape &shape,
                                      std::size_t channel_index) {
  std::vector<std::size_t> axes;
  for (std::size_t index = 0; index < shape.rank; ++index) {
    if (index == channel_index) {
      continue;
    }
    const auto dimension = dimension_or(shape.dimensions[index], 1);
    if (dimension > 1) {
      axes.push_back(index);
    }
  }
  return axes;
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
    try {
      if (!std::filesystem::exists(model_path)) {
        last_error_ = "model file does not exist";
        return;
      }

      session_options_.SetIntraOpNumThreads(1);
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
      session_.emplace(env_, model_path.c_str(), session_options_);

      auto input_name = session_->GetInputNameAllocated(0, allocator_);
      auto output_name = session_->GetOutputNameAllocated(0, allocator_);
      input_name_ = input_name.get();
      output_name_ = output_name.get();

      const auto input_info =
          session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
      const auto maybe_shape = read_shape(input_info);
      if (!maybe_shape.has_value()) {
        last_error_ = "model input shape is unreadable";
        return;
      }
      const auto shape = *maybe_shape;
      const auto channel = channel_axis(shape);
      if (!channel.has_value()) {
        last_error_ = describe_shape("input", maybe_shape) +
                      " unsupported: cannot find RGB channel dimension";
        return;
      }
      const auto spatial = spatial_axes(shape, *channel);
      if (spatial.size() < 2U) {
        last_error_ = describe_shape("input", maybe_shape) +
                      " unsupported: cannot find spatial dimensions";
        return;
      }

      input_shape_rank_ = shape.rank;
      for (std::size_t index = 0; index < shape.rank; ++index) {
        input_shape_[index] = dimension_or(shape.dimensions[index], 1);
      }
      input_height_axis_ = spatial[spatial.size() - 2U];
      input_width_axis_ = spatial[spatial.size() - 1U];
      input_channel_axis_ = *channel;

      const auto input_height =
          static_cast<std::uint32_t>(input_shape_[input_height_axis_]);
      const auto input_width =
          static_cast<std::uint32_t>(input_shape_[input_width_axis_]);
      ready_ = dimension_or(input_shape_[input_channel_axis_], 0) == 3 &&
               dimensions_are_sane(input_width, input_height);
      if (!ready_) {
        last_error_ = describe_shape("input", maybe_shape) +
                      " unsupported: invalid image dimensions";
      }
    } catch (const std::exception &error) {
      ready_ = false;
      session_.reset();
      last_error_ = error.what();
    }
  }

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] std::string_view last_error() const noexcept {
    return last_error_;
  }

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

    const auto input_height =
        static_cast<std::uint32_t>(input_shape_[input_height_axis_]);
    const auto input_width =
        static_cast<std::uint32_t>(input_shape_[input_width_axis_]);
    if (!dimensions_are_sane(input_width, input_height)) {
      return fallback_segment_person(frame);
    }
    const auto source = filters::detail::read_packed_rgb(frame);
    auto input_count = std::size_t{1U};
    for (std::size_t index = 0; index < input_shape_rank_; ++index) {
      input_count *= static_cast<std::size_t>(input_shape_[index]);
    }
    std::vector<float> input(input_count);

    std::array<std::size_t, 8> strides{};
    auto stride = std::size_t{1U};
    for (std::size_t index = input_shape_rank_; index > 0U; --index) {
      strides[index - 1U] = stride;
      stride *= static_cast<std::size_t>(input_shape_[index - 1U]);
    }

    for (std::uint32_t y = 0; y < input_height; ++y) {
      const auto source_y = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(y) * frame.height()) / input_height);
      for (std::uint32_t x = 0; x < input_width; ++x) {
        const auto source_x = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(x) * frame.width()) / input_width);
        const auto pixel = source[filters::detail::pixel_index(
            source_x, source_y, frame.width())];
        auto base = std::size_t{0U};
        base += static_cast<std::size_t>(y) * strides[input_height_axis_];
        base += static_cast<std::size_t>(x) * strides[input_width_axis_];
        input[base] = static_cast<float>(pixel.red) / 255.0F;
        input[base + strides[input_channel_axis_]] =
            static_cast<float>(pixel.green) / 255.0F;
        input[base + (2U * strides[input_channel_axis_])] =
            static_cast<float>(pixel.blue) / 255.0F;
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor =
        Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                        input_shape_.data(), input_shape_rank_);
    const char *input_names[] = {input_name_.c_str()};
    const char *output_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor,
                                 1, output_names, 1);
    if (outputs.empty() || !outputs.front().IsTensor()) {
      return fallback_segment_person(frame);
    }

    const auto output_info = outputs.front().GetTensorTypeAndShapeInfo();
    const auto output_shape = read_shape(output_info);
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
    if (output_shape.has_value()) {
      const auto shape = *output_shape;
      if (shape.rank == 2U) {
        mask_height = static_cast<std::uint32_t>(dimension_or(
            shape.dimensions[0], static_cast<std::int64_t>(input_height)));
        mask_width = static_cast<std::uint32_t>(dimension_or(
            shape.dimensions[1], static_cast<std::int64_t>(input_width)));
      } else if (shape.rank == 3U) {
        const auto first_channel_dim =
            static_cast<std::uint32_t>(dimension_or(shape.dimensions[0], 1));
        const auto last_channel_dim =
            static_cast<std::uint32_t>(dimension_or(shape.dimensions[2], 1));
        if (last_channel_dim <= 4U) {
          channel_count = last_channel_dim;
          person_channel = channel_count > 1U ? 1U : 0U;
          channels_last = true;
          mask_height = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[0], static_cast<std::int64_t>(input_height)));
          mask_width = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[1], static_cast<std::int64_t>(input_width)));
        } else if (first_channel_dim <= 4U) {
          channel_count = first_channel_dim;
          person_channel = channel_count > 1U ? 1U : 0U;
          mask_height = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[1], static_cast<std::int64_t>(input_height)));
          mask_width = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[2], static_cast<std::int64_t>(input_width)));
        }
      } else if (shape.rank == 4U) {
        mask_height = static_cast<std::uint32_t>(
            dimension_or(shape.dimensions[2], input_height));
        mask_width = static_cast<std::uint32_t>(
            dimension_or(shape.dimensions[3], input_width));
        const auto first_channel_dim =
            static_cast<std::uint32_t>(dimension_or(shape.dimensions[1], 1));
        const auto last_channel_dim =
            static_cast<std::uint32_t>(dimension_or(shape.dimensions[3], 1));
        if (last_channel_dim <= 4U) {
          channel_count = last_channel_dim;
          person_channel = channel_count > 1U ? 1U : 0U;
          channels_last = true;
          mask_height = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[1], static_cast<std::int64_t>(input_height)));
          mask_width = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[2], static_cast<std::int64_t>(input_width)));
        } else if (first_channel_dim <= 4U) {
          channel_count = first_channel_dim;
          person_channel = channel_count > 1U ? 1U : 0U;
          mask_height = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[2], static_cast<std::int64_t>(input_height)));
          mask_width = static_cast<std::uint32_t>(dimension_or(
              shape.dimensions[3], static_cast<std::int64_t>(input_width)));
        }
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
  std::array<std::int64_t, 8> input_shape_;
  std::size_t input_shape_rank_ = 4U;
  std::size_t input_height_axis_ = 2U;
  std::size_t input_width_axis_ = 3U;
  std::size_t input_channel_axis_ = 1U;
  std::optional<SegmentationMask> cached_mask_;
  std::uint64_t frame_index_ = 0;
  std::uint32_t inference_interval_ = 3;
  double mask_smoothing_ = 0.70;
  bool ready_ = false;
  std::string last_error_;
#else
  Impl(const std::string &, std::uint32_t, double) {}
  [[nodiscard]] bool ready() const noexcept { return false; }
  [[nodiscard]] std::string_view last_error() const noexcept {
    return "ONNX Runtime support is not compiled in";
  }
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

std::string_view OnnxRuntimeEngine::last_error() const noexcept {
  return impl_ == nullptr ? "ONNX Runtime engine is not initialized"
                          : impl_->last_error();
}

} // namespace lmp::ai
