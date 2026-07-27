#include "lmp/ai/onnx_runtime_engine.hpp"

#include "../filters/spatial_filter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if LMP_HAS_ONNXRUNTIME
#include <dlfcn.h>
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
  auto shape = TensorShape{};
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

std::optional<TensorShape> parse_shape_override(const std::string &value) {
  if (value.empty()) {
    return std::nullopt;
  }
  auto shape = TensorShape{};
  std::istringstream input{value};
  std::string token;
  while (std::getline(input, token, 'x')) {
    if (shape.rank >= shape.dimensions.size()) {
      return std::nullopt;
    }
    const auto dimension = std::stoll(token);
    shape.dimensions[shape.rank] = dimension_or(dimension, 0);
    if (shape.dimensions[shape.rank] <= 0) {
      return std::nullopt;
    }
    ++shape.rank;
  }
  return shape.rank > 0U ? std::optional<TensorShape>{shape} : std::nullopt;
}

std::string canonical_provider(std::string_view provider) {
  if (provider == "cpu" || provider == "CPUExecutionProvider") {
    return "CPUExecutionProvider";
  }
  if (provider == "openvino" || provider == "OpenVINOExecutionProvider") {
    return "OpenVINOExecutionProvider";
  }
  if (provider == "migraphx" || provider == "MIGraphXExecutionProvider") {
    return "MIGraphXExecutionProvider";
  }
  return std::string{provider};
}

std::string join_available_providers() {
  try {
    const auto providers = Ort::GetAvailableProviders();
    std::string result = "[";
    for (std::size_t index = 0; index < providers.size(); ++index) {
      if (index > 0U) {
        result += ",";
      }
      result += providers[index];
    }
    result += "]";
    return result;
  } catch (const std::exception &error) {
    return std::string{"unavailable:"} + error.what();
  }
}

bool provider_list_contains(const std::string &providers,
                            const std::string &provider) {
  return providers.find(provider) != std::string::npos;
}

std::vector<std::string> available_provider_names() {
  try {
    return Ort::GetAvailableProviders();
  } catch (const std::exception &) {
    return {};
  }
}

bool provider_name_contains(const std::vector<std::string> &providers,
                            std::string_view provider) {
  return std::find(providers.begin(), providers.end(), provider) !=
         providers.end();
}

bool future_is_ready(std::future<SegmentationMask> &future) {
  return future.valid() &&
         future.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

void append_openvino_provider(Ort::SessionOptions &options) {
  using AppendProvider = OrtStatus *(*)(OrtSessionOptions *, const char *);
  auto *raw_symbol =
      dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_OpenVINO");
  if (raw_symbol == nullptr) {
    throw std::runtime_error(
        "OrtSessionOptionsAppendExecutionProvider_OpenVINO symbol was not "
        "loaded");
  }
  auto *append_provider = reinterpret_cast<AppendProvider>(raw_symbol);
  Ort::ThrowOnError(append_provider(options, "CPU_FP32"));
}

void append_migraphx_provider(Ort::SessionOptions &options) {
  using AppendProvider = OrtStatus *(*)(OrtSessionOptions *, int);
  auto *raw_symbol =
      dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_MIGraphX");
  if (raw_symbol == nullptr) {
    throw std::runtime_error(
        "OrtSessionOptionsAppendExecutionProvider_MIGraphX symbol was not "
        "loaded");
  }
  auto *append_provider = reinterpret_cast<AppendProvider>(raw_symbol);
  Ort::ThrowOnError(append_provider(options, 0));
}

#endif

} // namespace

class OnnxRuntimeEngine::Impl {
public:
#if LMP_HAS_ONNXRUNTIME
  Impl(const std::string &model_path, std::uint32_t inference_interval,
       double mask_smoothing, std::string input_shape, std::string output_shape,
       std::string requested_provider, bool allow_provider_fallback,
       std::string openvino_device)
      : env_(ORT_LOGGING_LEVEL_WARNING, "linux-media-pipeline"),
        session_options_{}, allocator_{}, input_shape_{1, 3, 256, 256},
        output_shape_(std::move(output_shape)),
        allow_provider_fallback_(allow_provider_fallback),
        requested_provider_(std::move(requested_provider)),
        openvino_device_requested_(std::move(openvino_device)) {
    inference_interval_ = std::max<std::uint32_t>(1U, inference_interval);
    mask_smoothing_ = std::clamp(mask_smoothing, 0.0, 0.95);
    try {
      available_providers_ = join_available_providers();
      select_provider();
      if (!last_error_.empty() && !allow_provider_fallback_) {
        return;
      }
      if (!std::filesystem::exists(model_path)) {
        last_error_ = "model file does not exist";
        return;
      }

      const auto hardware_threads = std::thread::hardware_concurrency();
      const auto inference_threads =
          static_cast<int>(std::clamp(hardware_threads / 2U, 1U, 4U));
      session_options_.SetIntraOpNumThreads(inference_threads);
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_.emplace(env_, model_path.c_str(), session_options_);

      auto input_name = session_->GetInputNameAllocated(0, allocator_);
      auto output_name = session_->GetOutputNameAllocated(0, allocator_);
      input_name_ = input_name.get();
      output_name_ = output_name.get();

      const auto input_info =
          session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
      auto maybe_shape = read_shape(input_info);
      if (!maybe_shape.has_value()) {
        maybe_shape = parse_shape_override(input_shape);
        if (!maybe_shape.has_value()) {
          last_error_ = "model input shape is unreadable";
          return;
        }
        input_shape_overridden_ = true;
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

      const auto output_info =
          session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
      auto maybe_output_shape = read_shape(output_info);
      if (!maybe_output_shape.has_value()) {
        maybe_output_shape = parse_shape_override(output_shape_);
      }
      model_summary_ =
          "input_name=" + input_name_ + " " +
          describe_shape("input", maybe_shape) + " input_layout=" +
          (input_channel_axis_ < input_height_axis_ ? "NCHW" : "NHWC") +
          " output_name=" + output_name_ + " " +
          describe_shape("output", maybe_output_shape);

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
  [[nodiscard]] std::string_view requested_provider() const noexcept {
    return requested_provider_;
  }
  [[nodiscard]] std::string_view active_provider() const noexcept {
    return active_provider_;
  }
  [[nodiscard]] std::string_view available_providers() const noexcept {
    return available_providers_;
  }
  [[nodiscard]] bool provider_fallback() const noexcept {
    return provider_fallback_;
  }
  [[nodiscard]] std::string_view provider_fallback_reason() const noexcept {
    return provider_fallback_reason_;
  }
  [[nodiscard]] std::string_view model_summary() const noexcept {
    return model_summary_;
  }
  [[nodiscard]] std::string_view openvino_available_devices() const noexcept {
    return openvino_available_devices_;
  }
  [[nodiscard]] std::string_view openvino_device_requested() const noexcept {
    return openvino_device_requested_;
  }
  [[nodiscard]] std::string_view openvino_device_active() const noexcept {
    return openvino_device_active_;
  }

  [[nodiscard]] SegmentationMask segment_person(const frame::Frame &frame) {
    if (!ready_ || !session_.has_value()) {
      return fallback_segment_person(frame);
    }
    ++frame_index_;

    if (pending_mask_.valid() && future_is_ready(pending_mask_)) {
      try {
        cached_mask_ = pending_mask_.get();
      } catch (const std::exception &error) {
        last_error_ = error.what();
      }
    }

    const auto should_infer =
        !cached_mask_.has_value() ||
        (inference_interval_ <= 1U ||
         ((frame_index_ - 1U) % inference_interval_) == 0U);
    if (should_infer && !pending_mask_.valid()) {
      if (!cached_mask_.has_value()) {
        cached_mask_ = infer_mask(frame, std::nullopt);
      } else {
        auto snapshot = frame;
        auto previous = cached_mask_;
        pending_mask_ = std::async(std::launch::async,
                                   [this, snapshot = std::move(snapshot),
                                    previous = std::move(previous)] {
                                     return infer_mask(snapshot, previous);
                                   });
      }
    }

    if (cached_mask_.has_value()) {
      return *cached_mask_;
    }
    return fallback_segment_person(frame);
  }

private:
  [[nodiscard]] SegmentationMask
  infer_mask(const frame::Frame &frame,
             const std::optional<SegmentationMask> &previous_mask) {
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
    auto output_shape = read_shape(output_info);
    if (!output_shape.has_value()) {
      output_shape = parse_shape_override(output_shape_);
    }
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
    mask.reserve(static_cast<std::size_t>(mask_width) * mask_height);
    for (std::uint32_t y = 0; y < mask_height; ++y) {
      for (std::uint32_t x = 0; x < mask_width; ++x) {
        const auto mask_index = filters::detail::pixel_index(x, y, mask_width);
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
    auto current = SegmentationMask{mask_width, mask_height, std::move(mask)};
    if (previous_mask.has_value() &&
        previous_mask->width() == current.width() &&
        previous_mask->height() == current.height()) {
      current = current.blend_with(*previous_mask, mask_smoothing_);
    }
    return current;
  }

  void select_provider() {
    if (requested_provider_.empty()) {
      requested_provider_ = "auto";
    }
    active_provider_ = "CPUExecutionProvider";
    provider_fallback_ = false;
    provider_fallback_reason_.clear();

    if (requested_provider_ == "cpu") {
      return;
    }
    auto requested = canonical_provider(requested_provider_);
    if (requested_provider_ == "auto") {
      if (provider_list_contains(available_providers_,
                                 "MIGraphXExecutionProvider")) {
        requested = "MIGraphXExecutionProvider";
      } else {
        return;
      }
    }
    if (!provider_list_contains(available_providers_, requested)) {
      provider_fallback_ = true;
      provider_fallback_reason_ = requested + " unavailable";
      if (!allow_provider_fallback_) {
        last_error_ = provider_fallback_reason_;
      }
      return;
    }

    try {
      if (requested == "OpenVINOExecutionProvider") {
        if (openvino_device_requested_.empty()) {
          openvino_device_requested_ = "CPU";
        }
        if (openvino_device_requested_ != "CPU") {
          throw std::runtime_error(
              "OpenVINO device " + openvino_device_requested_ +
              " is not accepted unless OpenVINO enumerates it; available "
              "devices " +
              openvino_available_devices_);
        }
        append_openvino_provider(session_options_);
        active_provider_ = requested;
        openvino_device_active_ = openvino_device_requested_;
        return;
      }
      if (requested == "MIGraphXExecutionProvider") {
        append_migraphx_provider(session_options_);
        active_provider_ = requested;
        return;
      }
      provider_fallback_ = true;
      provider_fallback_reason_ = requested + " is not supported by lmp";
    } catch (const std::exception &error) {
      provider_fallback_ = true;
      provider_fallback_reason_ =
          requested + " activation failed: " + error.what();
    }
    if (provider_fallback_ && !allow_provider_fallback_) {
      last_error_ = provider_fallback_reason_;
    }
  }

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::optional<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::array<std::int64_t, 8> input_shape_;
  std::string output_shape_;
  std::size_t input_shape_rank_ = 4U;
  std::size_t input_height_axis_ = 2U;
  std::size_t input_width_axis_ = 3U;
  std::size_t input_channel_axis_ = 1U;
  std::optional<SegmentationMask> cached_mask_;
  std::future<SegmentationMask> pending_mask_;
  std::uint64_t frame_index_ = 0;
  std::uint32_t inference_interval_ = 3;
  double mask_smoothing_ = 0.70;
  bool ready_ = false;
  bool input_shape_overridden_ = false;
  bool allow_provider_fallback_ = true;
  bool provider_fallback_ = false;
  std::string requested_provider_ = "auto";
  std::string active_provider_ = "CPUExecutionProvider";
  std::string available_providers_ = "[]";
  std::string provider_fallback_reason_;
  std::string model_summary_;
  std::string openvino_available_devices_ = "[CPU]";
  std::string openvino_device_requested_ = "CPU";
  std::string openvino_device_active_ = "CPU";
  std::string last_error_;
#else
  Impl(const std::string &, std::uint32_t, double, std::string, std::string,
       std::string, bool, std::string) {}
  [[nodiscard]] bool ready() const noexcept { return false; }
  [[nodiscard]] std::string_view last_error() const noexcept {
    return "ONNX Runtime support is not compiled in";
  }
  [[nodiscard]] std::string_view requested_provider() const noexcept {
    return "unavailable";
  }
  [[nodiscard]] std::string_view active_provider() const noexcept {
    return "unavailable";
  }
  [[nodiscard]] std::string_view available_providers() const noexcept {
    return "[]";
  }
  [[nodiscard]] bool provider_fallback() const noexcept { return false; }
  [[nodiscard]] std::string_view provider_fallback_reason() const noexcept {
    return "ONNX Runtime support is not compiled in";
  }
  [[nodiscard]] std::string_view model_summary() const noexcept {
    return "unavailable";
  }
  [[nodiscard]] std::string_view openvino_available_devices() const noexcept {
    return "[]";
  }
  [[nodiscard]] std::string_view openvino_device_requested() const noexcept {
    return "unavailable";
  }
  [[nodiscard]] std::string_view openvino_device_active() const noexcept {
    return "unavailable";
  }
  [[nodiscard]] SegmentationMask segment_person(const frame::Frame &frame) {
    return fallback_segment_person(frame);
  }
#endif
};

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_, 3U, 0.70, "", "", "auto", true,
                                   "CPU")) {}

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path,
                                     std::uint32_t inference_interval,
                                     double mask_smoothing)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_, inference_interval,
                                   mask_smoothing, "", "", "auto", true,
                                   "CPU")) {}

OnnxRuntimeEngine::OnnxRuntimeEngine(std::string model_path,
                                     std::uint32_t inference_interval,
                                     double mask_smoothing,
                                     std::string input_shape,
                                     std::string output_shape)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(model_path_, inference_interval,
                                   mask_smoothing, std::move(input_shape),
                                   std::move(output_shape), "auto", true,
                                   "CPU")) {}

OnnxRuntimeEngine::OnnxRuntimeEngine(
    std::string model_path, std::uint32_t inference_interval,
    double mask_smoothing, std::string input_shape, std::string output_shape,
    std::string requested_provider, bool allow_provider_fallback,
    std::string openvino_device)
    : model_path_(std::move(model_path)),
      impl_(std::make_unique<Impl>(
          model_path_, inference_interval, mask_smoothing,
          std::move(input_shape), std::move(output_shape),
          std::move(requested_provider), allow_provider_fallback,
          std::move(openvino_device))) {}

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

std::string_view OnnxRuntimeEngine::requested_provider() const noexcept {
  return impl_ == nullptr ? "unavailable" : impl_->requested_provider();
}

std::string_view OnnxRuntimeEngine::active_provider() const noexcept {
  return impl_ == nullptr ? "unavailable" : impl_->active_provider();
}

std::string_view OnnxRuntimeEngine::available_providers() const noexcept {
  return impl_ == nullptr ? "[]" : impl_->available_providers();
}

bool OnnxRuntimeEngine::provider_fallback() const noexcept {
  return impl_ != nullptr && impl_->provider_fallback();
}

std::string_view OnnxRuntimeEngine::provider_fallback_reason() const noexcept {
  return impl_ == nullptr ? "ONNX Runtime engine is not initialized"
                          : impl_->provider_fallback_reason();
}

std::string_view OnnxRuntimeEngine::model_summary() const noexcept {
  return impl_ == nullptr ? "unavailable" : impl_->model_summary();
}

std::string_view
OnnxRuntimeEngine::openvino_available_devices() const noexcept {
  return impl_ == nullptr ? "[]" : impl_->openvino_available_devices();
}

std::string_view OnnxRuntimeEngine::openvino_device_requested() const noexcept {
  return impl_ == nullptr ? "unavailable" : impl_->openvino_device_requested();
}

std::string_view OnnxRuntimeEngine::openvino_device_active() const noexcept {
  return impl_ == nullptr ? "unavailable" : impl_->openvino_device_active();
}

std::string OnnxRuntimeEngine::runtime_version() {
#if LMP_HAS_ONNXRUNTIME
  return OrtGetApiBase()->GetVersionString();
#else
  return "unavailable";
#endif
}

std::vector<OnnxProviderInfo> OnnxRuntimeEngine::provider_infos() {
  auto result = std::vector<OnnxProviderInfo>{};
#if LMP_HAS_ONNXRUNTIME
  const auto providers = available_provider_names();
  for (const auto &provider : providers) {
    result.push_back(OnnxProviderInfo{provider, true, true, ""});
  }
  const auto add_missing = [&](std::string name) {
    if (!provider_name_contains(providers, name)) {
      result.push_back(OnnxProviderInfo{
          std::move(name), false, false,
          "provider not included in active ONNX Runtime build"});
    }
  };
  add_missing("MIGraphXExecutionProvider");
  add_missing("OpenVINOExecutionProvider");
  add_missing("CPUExecutionProvider");
#else
  result.push_back(OnnxProviderInfo{"CPUExecutionProvider", false, false,
                                    "ONNX Runtime support is not compiled in"});
  result.push_back(OnnxProviderInfo{"MIGraphXExecutionProvider", false, false,
                                    "ONNX Runtime support is not compiled in"});
#endif
  return result;
}

} // namespace lmp::ai
