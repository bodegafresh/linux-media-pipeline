#include "lmp/filters/background_blur_filter.hpp"

#include "lmp/ai/onnx_runtime_engine.hpp"

#include "spatial_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if LMP_HAS_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace lmp::filters {
namespace {

#if LMP_HAS_OPENCL
const char *background_blur_kernel_source() {
  return R"CLC(
__kernel void background_blur(__global const uchar *input,
                              __global uchar *output,
                              __global const uchar *mask,
                              const uint width,
                              const uint height,
                              const uint stride,
                              const uint pixel_size,
                              const uint red_offset,
                              const uint green_offset,
                              const uint blue_offset,
                              const uint radius,
                              const uchar foreground_threshold,
                              const float brightness,
                              const float contrast,
                              const float saturation,
                              const uint crop_x,
                              const uint crop_y,
                              const uint crop_width,
                              const uint crop_height,
                              const uint mask_mode,
                              const uint mask_width_px,
                              const uint mask_height_px,
                              const float mask_width,
                              const float mask_height) {
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);
  if (x >= width || y >= height) {
    return;
  }

  const uint source_x = crop_x + ((x * crop_width) / width);
  const uint source_y = crop_y + ((y * crop_height) / height);
  const uint base = (y * stride) + (x * pixel_size);
  const uint source_base = (source_y * stride) + (source_x * pixel_size);
  const float red = (float)input[source_base + red_offset];
  const float green = (float)input[source_base + green_offset];
  const float blue = (float)input[source_base + blue_offset];
  const uchar luma = convert_uchar_sat_rte((0.299f * red) + (0.587f * green) + (0.114f * blue));
  const float nx = (((float)x + 0.5f) / (float)width - 0.5f) / mask_width;
  const float ny = (((float)y + 0.5f) / (float)height - 0.48f) / mask_height;
  const bool center_foreground = ((nx * nx) + (ny * ny)) <= 1.0f;
  const uint mask_x = min((source_x * mask_width_px) / width, mask_width_px - 1U);
  const uint mask_y = min((source_y * mask_height_px) / height, mask_height_px - 1U);
  const uchar mask_value = mask[(mask_y * mask_width_px) + mask_x];
  const float foreground_alpha =
      mask_mode == 2U ? clamp(((float)mask_value - (float)foreground_threshold) /
                                  (255.0f - (float)foreground_threshold),
                              0.0f, 1.0f)
      : (mask_mode == 1U ? (center_foreground ? 1.0f : 0.0f)
                         : (luma >= foreground_threshold ? 1.0f : 0.0f));

  float out_red = red;
  float out_green = green;
  float out_blue = blue;

  if (foreground_alpha < 1.0f) {
    uint count = 0;
    uint red_sum = 0;
    uint green_sum = 0;
    uint blue_sum = 0;
    const int signed_radius = (int)radius;
    for (int ky = -signed_radius; ky <= signed_radius; ++ky) {
      const int sy = clamp((int)source_y + ky, 0, (int)height - 1);
      for (int kx = -signed_radius; kx <= signed_radius; ++kx) {
        const int sx = clamp((int)source_x + kx, 0, (int)width - 1);
        const uint sample = (((uint)sy) * stride) + (((uint)sx) * pixel_size);
        red_sum += input[sample + red_offset];
        green_sum += input[sample + green_offset];
        blue_sum += input[sample + blue_offset];
        ++count;
      }
    }
    const float blurred_red = (float)((red_sum + (count / 2U)) / count);
    const float blurred_green = (float)((green_sum + (count / 2U)) / count);
    const float blurred_blue = (float)((blue_sum + (count / 2U)) / count);
    out_red = (red * foreground_alpha) + (blurred_red * (1.0f - foreground_alpha));
    out_green = (green * foreground_alpha) + (blurred_green * (1.0f - foreground_alpha));
    out_blue = (blue * foreground_alpha) + (blurred_blue * (1.0f - foreground_alpha));
  }

  out_red = ((out_red - 128.0f) * contrast) + 128.0f + brightness;
  out_green = ((out_green - 128.0f) * contrast) + 128.0f + brightness;
  out_blue = ((out_blue - 128.0f) * contrast) + 128.0f + brightness;
  const float gray = (0.299f * out_red) + (0.587f * out_green) + (0.114f * out_blue);
  out_red = gray + ((out_red - gray) * saturation);
  out_green = gray + ((out_green - gray) * saturation);
  out_blue = gray + ((out_blue - gray) * saturation);

  output[base + red_offset] = convert_uchar_sat_rte(out_red);
  output[base + green_offset] = convert_uchar_sat_rte(out_green);
  output[base + blue_offset] = convert_uchar_sat_rte(out_blue);
  if (pixel_size == 4U) {
    output[base + 3U] = input[source_base + 3U];
  }
}
)CLC";
}

struct OpenClPixelLayout {
  cl_uint pixel_size;
  cl_uint red_offset;
  cl_uint green_offset;
  cl_uint blue_offset;
};

OpenClPixelLayout opencl_layout(frame::PixelFormat format) {
  switch (format) {
  case frame::PixelFormat::Rgba:
  case frame::PixelFormat::Rgb:
    return OpenClPixelLayout{format == frame::PixelFormat::Rgba ? 4U : 3U, 0U,
                             1U, 2U};
  case frame::PixelFormat::Bgr:
    return OpenClPixelLayout{3U, 2U, 1U, 0U};
  case frame::PixelFormat::Yuv420p:
  case frame::PixelFormat::Nv12:
  case frame::PixelFormat::P010:
    throw std::invalid_argument(
        "filter supports only RGB, BGR, and RGBA frames");
  }
  throw std::invalid_argument("unsupported pixel format");
}

class OpenClBackgroundBlurResources {
public:
  OpenClBackgroundBlurResources() {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS ||
        platform_count == 0U) {
      return;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) !=
        CL_SUCCESS) {
      return;
    }

    for (const auto platform : platforms) {
      cl_uint device_count = 0;
      const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1,
                                         &device_, &device_count);
      if (status == CL_SUCCESS && device_count > 0U) {
        platform_ = platform;
        break;
      }
      device_ = nullptr;
    }
    if (device_ == nullptr) {
      return;
    }

    cl_int error = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    if (error != CL_SUCCESS || context_ == nullptr) {
      reset();
      return;
    }

#if defined(CL_VERSION_2_0)
    const cl_queue_properties queue_properties[] = {0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_,
                                                queue_properties, &error);
#else
    queue_ = clCreateCommandQueue(context_, device_, 0, &error);
#endif
    if (error != CL_SUCCESS || queue_ == nullptr) {
      reset();
      return;
    }

    const char *source = background_blur_kernel_source();
    const std::size_t source_size = std::char_traits<char>::length(source);
    program_ =
        clCreateProgramWithSource(context_, 1, &source, &source_size, &error);
    if (error != CL_SUCCESS || program_ == nullptr) {
      reset();
      return;
    }

    error = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
      reset();
      return;
    }

    kernel_ = clCreateKernel(program_, "background_blur", &error);
    if (error != CL_SUCCESS || kernel_ == nullptr) {
      reset();
      return;
    }
    ready_ = true;
    platform_name_ = read_platform_name();
    device_name_ = read_device_name();
  }

  OpenClBackgroundBlurResources(const OpenClBackgroundBlurResources &) = delete;
  OpenClBackgroundBlurResources &
  operator=(const OpenClBackgroundBlurResources &) = delete;

  ~OpenClBackgroundBlurResources() { reset(); }

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] cl_context context() const noexcept { return context_; }
  [[nodiscard]] cl_command_queue queue() const noexcept { return queue_; }
  [[nodiscard]] cl_kernel kernel() const noexcept { return kernel_; }
  [[nodiscard]] std::string_view platform_name() const noexcept {
    return platform_name_;
  }
  [[nodiscard]] std::string_view device_name() const noexcept {
    return device_name_;
  }

private:
  [[nodiscard]] std::string read_platform_name() const {
    if (platform_ == nullptr) {
      return "unknown";
    }
    std::size_t size = 0;
    if (clGetPlatformInfo(platform_, CL_PLATFORM_NAME, 0, nullptr, &size) !=
            CL_SUCCESS ||
        size == 0U) {
      return "unknown";
    }
    std::string value(size, '\0');
    if (clGetPlatformInfo(platform_, CL_PLATFORM_NAME, value.size(),
                          value.data(), nullptr) != CL_SUCCESS) {
      return "unknown";
    }
    if (!value.empty() && value.back() == '\0') {
      value.pop_back();
    }
    return value;
  }

  [[nodiscard]] std::string read_device_name() const {
    if (device_ == nullptr) {
      return "unknown";
    }
    std::size_t size = 0;
    if (clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &size) !=
            CL_SUCCESS ||
        size == 0U) {
      return "unknown";
    }
    std::string value(size, '\0');
    if (clGetDeviceInfo(device_, CL_DEVICE_NAME, value.size(), value.data(),
                        nullptr) != CL_SUCCESS) {
      return "unknown";
    }
    if (!value.empty() && value.back() == '\0') {
      value.pop_back();
    }
    return value;
  }

  void reset() noexcept {
    if (kernel_ != nullptr) {
      clReleaseKernel(kernel_);
      kernel_ = nullptr;
    }
    if (program_ != nullptr) {
      clReleaseProgram(program_);
      program_ = nullptr;
    }
    if (queue_ != nullptr) {
      clReleaseCommandQueue(queue_);
      queue_ = nullptr;
    }
    if (context_ != nullptr) {
      clReleaseContext(context_);
      context_ = nullptr;
    }
    ready_ = false;
  }

  cl_device_id device_ = nullptr;
  cl_platform_id platform_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  std::string platform_name_ = "unknown";
  std::string device_name_ = "unknown";
  bool ready_ = false;
};

OpenClBackgroundBlurResources &opencl_background_resources() {
  static OpenClBackgroundBlurResources resources;
  return resources;
}
#endif

#if LMP_HAS_OPENCL
struct Bounds {
  std::uint32_t min_x;
  std::uint32_t min_y;
  std::uint32_t max_x;
  std::uint32_t max_y;
};

struct Crop {
  double x;
  double y;
  double width;
  double height;
};

std::uint8_t clamp_to_byte(double value) noexcept {
  const auto rounded = static_cast<int>(value + 0.5);
  return static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
}

std::optional<Bounds> foreground_bounds(std::span<const std::uint8_t> bytes,
                                        frame::PixelFormat format,
                                        std::span<const std::size_t> strides,
                                        std::uint32_t width,
                                        std::uint32_t height,
                                        std::uint8_t threshold) {
  const auto layout_pixel_size = detail::packed_pixel_size(format);
  const auto row_bytes = static_cast<std::size_t>(width) * layout_pixel_size;
  if (strides.empty() || strides.front() < row_bytes) {
    throw std::invalid_argument("frame stride is smaller than row size");
  }

  auto bounds = Bounds{width, height, 0U, 0U};
  bool found = false;
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto row_offset = static_cast<std::size_t>(y) * strides.front();
    if (row_offset + row_bytes > bytes.size()) {
      throw std::invalid_argument("frame data is smaller than stride layout");
    }
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto *pixel = bytes.data() + row_offset +
                          (static_cast<std::size_t>(x) * layout_pixel_size);
      const auto red = format == frame::PixelFormat::Bgr ? pixel[2] : pixel[0];
      const auto green = pixel[1];
      const auto blue = format == frame::PixelFormat::Bgr ? pixel[0] : pixel[2];
      const auto luma =
          clamp_to_byte((0.299 * red) + (0.587 * green) + (0.114 * blue));
      if (luma < threshold) {
        continue;
      }
      bounds.min_x = std::min(bounds.min_x, x);
      bounds.min_y = std::min(bounds.min_y, y);
      bounds.max_x = std::max(bounds.max_x, x);
      bounds.max_y = std::max(bounds.max_y, y);
      found = true;
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return bounds;
}

std::optional<Bounds> mask_bounds(const ai::SegmentationMask &mask,
                                  std::uint8_t threshold) {
  auto bounds = Bounds{mask.width(), mask.height(), 0U, 0U};
  bool found = false;
  for (std::uint32_t y = 0; y < mask.height(); ++y) {
    for (std::uint32_t x = 0; x < mask.width(); ++x) {
      if (mask.at(x, y) < threshold) {
        continue;
      }
      bounds.min_x = std::min(bounds.min_x, x);
      bounds.min_y = std::min(bounds.min_y, y);
      bounds.max_x = std::max(bounds.max_x, x);
      bounds.max_y = std::max(bounds.max_y, y);
      found = true;
    }
  }
  if (!found) {
    return std::nullopt;
  }
  return bounds;
}

Bounds scale_bounds(Bounds bounds, std::uint32_t source_width,
                    std::uint32_t source_height, std::uint32_t target_width,
                    std::uint32_t target_height) {
  const auto min_x = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(bounds.min_x) * target_width) / source_width);
  const auto min_y = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(bounds.min_y) * target_height) /
      source_height);
  const auto max_x = static_cast<std::uint32_t>(
      ((static_cast<std::uint64_t>(bounds.max_x) + 1U) * target_width) /
      source_width);
  const auto max_y = static_cast<std::uint32_t>(
      ((static_cast<std::uint64_t>(bounds.max_y) + 1U) * target_height) /
      source_height);
  return Bounds{
      std::min(min_x, target_width - 1U), std::min(min_y, target_height - 1U),
      std::min(max_x, target_width - 1U), std::min(max_y, target_height - 1U)};
}

Crop crop_from_bounds(Bounds bounds, std::uint32_t frame_width,
                      std::uint32_t frame_height, double target_fill,
                      double max_zoom) {
  const auto aspect =
      static_cast<double>(frame_width) / static_cast<double>(frame_height);
  const auto person_width =
      static_cast<double>(bounds.max_x - bounds.min_x + 1U);
  const auto person_height =
      static_cast<double>(bounds.max_y - bounds.min_y + 1U);
  auto crop_width = std::max(person_width / target_fill,
                             (person_height / target_fill) * aspect);
  crop_width =
      std::max(crop_width, static_cast<double>(frame_width) / max_zoom);
  crop_width = std::min(crop_width, static_cast<double>(frame_width));
  auto crop_height = crop_width / aspect;
  if (crop_height > static_cast<double>(frame_height)) {
    crop_height = static_cast<double>(frame_height);
    crop_width = crop_height * aspect;
  }

  const auto crop_w =
      std::clamp(std::ceil(crop_width), 1.0, static_cast<double>(frame_width));
  const auto crop_h = std::clamp(std::ceil(crop_height), 1.0,
                                 static_cast<double>(frame_height));
  const auto center_x =
      (static_cast<double>(bounds.min_x) + static_cast<double>(bounds.max_x)) /
      2.0;
  const auto center_y =
      (static_cast<double>(bounds.min_y) + static_cast<double>(bounds.max_y)) /
      2.0;
  const auto crop_x = std::clamp(center_x - (crop_w / 2.0), 0.0,
                                 static_cast<double>(frame_width) - crop_w);
  const auto crop_y = std::clamp(center_y - (crop_h / 2.0), 0.0,
                                 static_cast<double>(frame_height) - crop_h);
  return Crop{crop_x, crop_y, crop_w, crop_h};
}

Crop smooth_crop(Crop desired,
                 std::optional<std::array<double, 4>> &previous_crop,
                 std::uint32_t frame_width, std::uint32_t frame_height) {
  if (!previous_crop.has_value()) {
    previous_crop = {desired.x, desired.y, desired.width, desired.height};
    return desired;
  }

  const auto previous = Crop{(*previous_crop)[0], (*previous_crop)[1],
                             (*previous_crop)[2], (*previous_crop)[3]};
  const auto previous_center_x = previous.x + (previous.width / 2.0);
  const auto previous_center_y = previous.y + (previous.height / 2.0);
  const auto desired_center_x = desired.x + (desired.width / 2.0);
  const auto desired_center_y = desired.y + (desired.height / 2.0);
  const auto movement = std::hypot(desired_center_x - previous_center_x,
                                   desired_center_y - previous_center_y);
  const auto size_delta = std::abs(desired.width - previous.width) +
                          std::abs(desired.height - previous.height);
  const auto dead_zone_pixels =
      0.025 * static_cast<double>(std::min(frame_width, frame_height));
  if (movement <= dead_zone_pixels && size_delta <= (dead_zone_pixels * 1.5)) {
    return previous;
  }

  constexpr auto kSmoothing = 0.68;
  constexpr auto kMaxCenterStepFraction = 0.10;
  constexpr auto kMaxSizeStepFraction = 0.06;
  const auto center_blend = 1.0 - kSmoothing;
  const auto max_center_step =
      kMaxCenterStepFraction *
      static_cast<double>(std::min(frame_width, frame_height));
  const auto max_width_step = kMaxSizeStepFraction * frame_width;
  const auto max_height_step = kMaxSizeStepFraction * frame_height;

  auto center_x =
      previous_center_x +
      std::clamp((desired_center_x - previous_center_x) * center_blend,
                 -max_center_step, max_center_step);
  auto center_y =
      previous_center_y +
      std::clamp((desired_center_y - previous_center_y) * center_blend,
                 -max_center_step, max_center_step);
  auto width = previous.width +
               std::clamp((desired.width - previous.width) * center_blend,
                          -max_width_step, max_width_step);
  auto height = previous.height +
                std::clamp((desired.height - previous.height) * center_blend,
                           -max_height_step, max_height_step);

  width = std::clamp(width, 1.0, static_cast<double>(frame_width));
  height = std::clamp(height, 1.0, static_cast<double>(frame_height));
  center_x = std::clamp(center_x, width / 2.0,
                        static_cast<double>(frame_width) - (width / 2.0));
  center_y = std::clamp(center_y, height / 2.0,
                        static_cast<double>(frame_height) - (height / 2.0));

  const auto smoothed =
      Crop{center_x - (width / 2.0), center_y - (height / 2.0), width, height};
  previous_crop = {smoothed.x, smoothed.y, smoothed.width, smoothed.height};
  return smoothed;
}

cl_uint opencl_mask_mode(std::string_view mode) {
  if (mode == "onnx") {
    return 2U;
  }
  if (mode == "tracked_center") {
    return 1U;
  }
  return mode == "center" ? 1U : 0U;
}
#endif

} // namespace

BackgroundBlurFilter::BackgroundBlurFilter(std::uint32_t radius,
                                           std::uint8_t foreground_threshold)
    : BackgroundBlurFilter(radius, foreground_threshold, "cpu") {}

BackgroundBlurFilter::BackgroundBlurFilter(std::uint32_t radius,
                                           std::uint8_t foreground_threshold,
                                           std::string backend)
    : BackgroundBlurFilter(radius, foreground_threshold, std::move(backend),
                           0.0, 1.0, 1.0) {}

BackgroundBlurFilter::BackgroundBlurFilter(std::uint32_t radius,
                                           std::uint8_t foreground_threshold,
                                           std::string backend,
                                           double brightness, double contrast,
                                           double saturation)
    : BackgroundBlurFilter(
          radius, foreground_threshold, std::move(backend), brightness,
          contrast, saturation, false, 1.0, 1.0, "luminance", 0.28, 0.42,
          "assets/models/person-segmentation.onnx", 3U, 0.70, "tracked_center",
          "", "", "auto", true, "CPU", 1U, 3U, false, false, 0.02, 0.85, 0.0) {}

BackgroundBlurFilter::BackgroundBlurFilter(
    std::uint32_t radius, std::uint8_t foreground_threshold,
    std::string backend, double brightness, double contrast, double saturation,
    bool auto_frame, double target_fill, double max_zoom, std::string mask_mode,
    double mask_width, double mask_height, std::string model_path,
    std::uint32_t inference_interval, double mask_smoothing,
    std::string fallback_mask_mode, std::string input_shape,
    std::string output_shape, std::string requested_provider,
    bool allow_provider_fallback, std::string openvino_device,
    std::uint32_t mask_expand, std::uint32_t mask_feather, bool invert_mask,
    bool keep_largest_component, double min_mask_coverage,
    double max_mask_coverage, double hint_y_offset)
    : radius_(radius), foreground_threshold_(foreground_threshold),
      backend_(std::move(backend)), brightness_(brightness),
      contrast_(contrast), saturation_(saturation), auto_frame_(auto_frame),
      target_fill_(target_fill), max_zoom_(max_zoom),
      mask_mode_(std::move(mask_mode)), mask_width_(mask_width),
      mask_height_(mask_height), model_path_(std::move(model_path)),
      inference_interval_(inference_interval), mask_smoothing_(mask_smoothing),
      fallback_mask_mode_(std::move(fallback_mask_mode)),
      input_shape_(std::move(input_shape)),
      output_shape_(std::move(output_shape)),
      requested_provider_(std::move(requested_provider)),
      allow_provider_fallback_(allow_provider_fallback),
      openvino_device_(std::move(openvino_device)), mask_expand_(mask_expand),
      mask_feather_(mask_feather), invert_mask_(invert_mask),
      keep_largest_component_(keep_largest_component),
      min_mask_coverage_(min_mask_coverage),
      max_mask_coverage_(max_mask_coverage), onnx_error_reported_(false) {
  static_cast<void>(hint_y_offset);
  if (radius_ == 0U) {
    throw std::invalid_argument("background blur radius must be >= 1");
  }
  if (foreground_threshold_ >= 255U) {
    throw std::invalid_argument(
        "background_blur foreground_threshold must be < 255");
  }
  if (contrast_ < 0.0) {
    throw std::invalid_argument("background_blur contrast must be >= 0");
  }
  if (saturation_ < 0.0) {
    throw std::invalid_argument("background_blur saturation must be >= 0");
  }
  if (target_fill_ <= 0.0 || target_fill_ > 1.0) {
    throw std::invalid_argument(
        "background_blur target_fill must be in (0, 1]");
  }
  if (max_zoom_ < 1.0) {
    throw std::invalid_argument("background_blur max_zoom must be >= 1");
  }
  if (min_mask_coverage_ < 0.0 || min_mask_coverage_ > 1.0 ||
      max_mask_coverage_ < 0.0 || max_mask_coverage_ > 1.0 ||
      min_mask_coverage_ > max_mask_coverage_) {
    throw std::invalid_argument("background_blur mask coverage bounds must be "
                                "ordered values in [0, 1]");
  }
  if (mask_mode_ != "luminance" && mask_mode_ != "center" &&
      mask_mode_ != "onnx") {
    throw std::invalid_argument(
        "background_blur mask_mode must be luminance, center, or onnx");
  }
  if (fallback_mask_mode_ != "luminance" && fallback_mask_mode_ != "center" &&
      fallback_mask_mode_ != "tracked_center") {
    throw std::invalid_argument(
        "background_blur fallback_mask_mode must be luminance, center, or "
        "tracked_center");
  }
  if (requested_provider_ != "auto" && requested_provider_ != "cpu" &&
      requested_provider_ != "migraphx" && requested_provider_ != "rocm" &&
      requested_provider_ != "openvino") {
    throw std::invalid_argument(
        "background_blur provider must be auto, cpu, migraphx, rocm, or "
        "openvino");
  }
  if (openvino_device_ != "CPU") {
    throw std::invalid_argument(
        "background_blur openvino_device must be CPU unless an "
        "OpenVINO-compatible Intel GPU is enumerated");
  }
  if (mask_width_ <= 0.0 || mask_height_ <= 0.0) {
    throw std::invalid_argument(
        "background_blur mask dimensions must be positive");
  }
  if (inference_interval_ == 0U) {
    throw std::invalid_argument(
        "background_blur inference_interval must be >= 1");
  }
  if (mask_smoothing_ < 0.0 || mask_smoothing_ > 0.95) {
    throw std::invalid_argument(
        "background_blur mask_smoothing must be in [0, 0.95]");
  }
}

std::optional<ai::SegmentationMask>
BackgroundBlurFilter::person_mask(frame::Frame &frame) const {
  if (mask_mode_ != "onnx") {
    return std::nullopt;
  }
  if (onnx_engine_ == nullptr) {
    try {
      onnx_engine_ = std::make_unique<ai::OnnxRuntimeEngine>(
          model_path_, inference_interval_, mask_smoothing_, input_shape_,
          output_shape_, requested_provider_, allow_provider_fallback_,
          openvino_device_);
    } catch (const std::exception &error) {
      frame.metadata()["background_blur_mask"] =
          "onnx_init_error_" + fallback_mask_mode_;
      if (!onnx_error_reported_) {
        std::cerr << "background_blur_onnx_error=init_failed message=\""
                  << error.what() << "\"\n";
        onnx_error_reported_ = true;
      }
      return std::nullopt;
    }
  }
  frame.metadata()["onnx_runtime_available_providers"] =
      std::string{onnx_engine_->available_providers()};
  frame.metadata()["onnx_runtime_provider_requested"] =
      std::string{onnx_engine_->requested_provider()};
  frame.metadata()["onnx_runtime_provider_active"] =
      std::string{onnx_engine_->active_provider()};
  frame.metadata()["onnx_runtime_provider_fallback"] =
      onnx_engine_->provider_fallback() ? "true" : "false";
  frame.metadata()["onnx_runtime_provider_fallback_reason"] =
      std::string{onnx_engine_->provider_fallback_reason()};
  frame.metadata()["onnx_runtime_model"] =
      std::string{onnx_engine_->model_summary()};
  frame.metadata()["openvino_available_devices"] =
      std::string{onnx_engine_->openvino_available_devices()};
  frame.metadata()["openvino_device_requested"] =
      std::string{onnx_engine_->openvino_device_requested()};
  frame.metadata()["openvino_device_active"] =
      std::string{onnx_engine_->openvino_device_active()};
  frame.metadata()["segmentation_inference_backend"] =
      std::string{onnx_engine_->active_provider()};
  const auto active_provider = std::string{onnx_engine_->active_provider()};
  if (active_provider == "ROCMExecutionProvider") {
    frame.metadata()["segmentation_inference_device"] = "ROCm";
  } else if (active_provider == "MIGraphXExecutionProvider") {
    frame.metadata()["segmentation_inference_device"] = "MIGraphX";
  } else {
    frame.metadata()["segmentation_inference_device"] =
        std::string{onnx_engine_->openvino_device_active()};
  }
  if (!onnx_engine_->available()) {
    frame.metadata()["background_blur_mask"] =
        "onnx_unavailable_" + fallback_mask_mode_;
    if (!onnx_error_reported_) {
      std::cerr << "background_blur_onnx_error=unavailable message=\""
                << onnx_engine_->last_error() << "\"\n";
      onnx_error_reported_ = true;
    }
    return std::nullopt;
  }
  try {
    auto mask = onnx_engine_->segment_person(frame);
    if (invert_mask_) {
      mask = ai::invert_mask(mask);
      frame.metadata()["segmentation_mask_inverted"] = "true";
    } else {
      frame.metadata()["segmentation_mask_inverted"] = "false";
    }
    frame.metadata()["segmentation_mask_coverage_raw"] =
        std::to_string(ai::mask_coverage(mask, foreground_threshold_));
    frame.metadata()["segmentation_mask_foreground_threshold"] =
        std::to_string(foreground_threshold_);
    if (keep_largest_component_) {
      mask = ai::largest_component_mask(mask, foreground_threshold_);
      frame.metadata()["segmentation_mask_largest_component"] = "true";
      frame.metadata()["segmentation_mask_coverage_component"] =
          std::to_string(ai::mask_coverage(mask, foreground_threshold_));
    } else {
      frame.metadata()["segmentation_mask_largest_component"] = "false";
    }
    const auto usable_coverage = ai::mask_coverage(mask, foreground_threshold_);
    if (usable_coverage < min_mask_coverage_ ||
        usable_coverage > max_mask_coverage_) {
      frame.metadata()["background_blur_mask"] =
          "onnx_rejected_" + fallback_mask_mode_;
      frame.metadata()["segmentation_mask_rejected"] =
          "coverage:" + std::to_string(usable_coverage);
      constexpr auto kMaxPreviousMaskReuses = 6U;
      if (last_good_person_mask_.has_value() &&
          last_good_person_mask_reuse_count_ < kMaxPreviousMaskReuses) {
        ++last_good_person_mask_reuse_count_;
        frame.metadata()["background_blur_mask"] = "onnx_previous";
        frame.metadata()["segmentation_mask_reused_previous"] = "true";
        frame.metadata()["segmentation_mask_previous_reuse_count"] =
            std::to_string(last_good_person_mask_reuse_count_);
        return last_good_person_mask_;
      }
      return std::nullopt;
    }
    const auto timing = onnx_engine_->last_timing();
    frame.metadata()["onnx_preprocess_ms"] =
        std::to_string(timing.preprocess_ms);
    frame.metadata()["onnx_inference_ms"] = std::to_string(timing.inference_ms);
    frame.metadata()["onnx_postprocess_ms"] =
        std::to_string(timing.postprocess_ms);
    mask = ai::refine_mask(mask, foreground_threshold_, mask_expand_,
                           mask_feather_);
    frame.metadata()["segmentation_mask_coverage_refined"] =
        std::to_string(ai::mask_coverage(mask, foreground_threshold_));
    frame.metadata()["background_blur_mask"] = "onnx";
    frame.metadata()["background_blur_mask_refinement"] =
        "expand:" + std::to_string(mask_expand_) +
        ",feather:" + std::to_string(mask_feather_);
    last_good_person_mask_ = mask;
    last_good_person_mask_reuse_count_ = 0;
    return mask;
  } catch (const std::exception &error) {
    frame.metadata()["background_blur_mask"] =
        "onnx_error_" + fallback_mask_mode_;
    if (!onnx_error_reported_) {
      std::cerr << "background_blur_onnx_error=inference_failed message=\""
                << error.what() << "\"\n";
      onnx_error_reported_ = true;
    }
    constexpr auto kMaxPreviousMaskReuses = 6U;
    if (last_good_person_mask_.has_value() &&
        last_good_person_mask_reuse_count_ < kMaxPreviousMaskReuses) {
      ++last_good_person_mask_reuse_count_;
      frame.metadata()["background_blur_mask"] = "onnx_previous";
      frame.metadata()["segmentation_mask_reused_previous"] = "true";
      frame.metadata()["segmentation_mask_previous_reuse_count"] =
          std::to_string(last_good_person_mask_reuse_count_);
      return last_good_person_mask_;
    }
    return std::nullopt;
  }
}

void BackgroundBlurFilter::process(frame::Frame &frame) const {
  frame.metadata()["background_blur_radius"] = std::to_string(radius_);
  if (backend_ == "opencl" && process_opencl(frame)) {
    frame.metadata()["background_blur_backend"] = "opencl";
    frame.metadata()["background_processing_backend"] = "opencl";
    return;
  }
  process_cpu(frame);
  frame.metadata()["background_blur_backend"] = "cpu";
  frame.metadata()["background_processing_backend"] = "cpu";
  frame.metadata()["background_processing_device"] = "CPU";
}

void BackgroundBlurFilter::process_cpu(frame::Frame &frame) const {
  if (auto_frame_) {
    frame.metadata()["background_blur_auto_frame"] = "opencl_required";
  }
  const auto original = detail::read_packed_rgb(frame);
  auto mask = person_mask(frame);
  if (!mask.has_value()) {
    ai::OnnxRuntimeEngine fallback_engine{""};
    mask = fallback_engine.segment_person(frame);
    frame.metadata()["background_blur_mask"] = mask_mode_ == "center"
                                                   ? "center_cpu_fallback"
                                                   : "luminance_cpu_fallback";
  }

  detail::apply_box_blur(frame, radius_);
  const auto blurred = detail::read_packed_rgb(frame);
  std::vector<detail::RgbPixel> output;
  output.reserve(original.size());

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    const auto mask_y = std::min(
        (static_cast<std::uint64_t>(y) * mask->height()) / frame.height(),
        static_cast<std::uint64_t>(mask->height() - 1U));
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto index = detail::pixel_index(x, y, frame.width());
      const auto mask_x = std::min(
          (static_cast<std::uint64_t>(x) * mask->width()) / frame.width(),
          static_cast<std::uint64_t>(mask->width() - 1U));
      const auto alpha = std::clamp(
          (static_cast<double>(mask->at(static_cast<std::uint32_t>(mask_x),
                                        static_cast<std::uint32_t>(mask_y))) -
           static_cast<double>(foreground_threshold_)) /
              (255.0 - static_cast<double>(foreground_threshold_)),
          0.0, 1.0);
      auto pixel = detail::RgbPixel{
          detail::clamp_to_byte((original[index].red * alpha) +
                                (blurred[index].red * (1.0 - alpha))),
          detail::clamp_to_byte((original[index].green * alpha) +
                                (blurred[index].green * (1.0 - alpha))),
          detail::clamp_to_byte((original[index].blue * alpha) +
                                (blurred[index].blue * (1.0 - alpha)))};
      auto red = ((pixel.red - 128.0) * contrast_) + 128.0 + brightness_;
      auto green = ((pixel.green - 128.0) * contrast_) + 128.0 + brightness_;
      auto blue = ((pixel.blue - 128.0) * contrast_) + 128.0 + brightness_;
      const auto gray = (0.299 * red) + (0.587 * green) + (0.114 * blue);
      red = gray + ((red - gray) * saturation_);
      green = gray + ((green - gray) * saturation_);
      blue = gray + ((blue - gray) * saturation_);
      output.push_back(detail::RgbPixel{detail::clamp_to_byte(red),
                                        detail::clamp_to_byte(green),
                                        detail::clamp_to_byte(blue)});
    }
  }

  detail::write_packed_rgb(frame, output);
}

bool BackgroundBlurFilter::process_opencl(frame::Frame &frame) const {
#if LMP_HAS_OPENCL
  const auto layout = opencl_layout(frame.format());
  const auto strides = frame.strides();
  const auto row_bytes =
      static_cast<std::size_t>(frame.width()) * layout.pixel_size;
  if (strides.empty() || strides.front() < row_bytes) {
    throw std::invalid_argument("frame stride is smaller than row size");
  }

  auto bytes = frame.data();
  if (bytes.empty()) {
    return true;
  }

  auto &resources = opencl_background_resources();
  if (!resources.ready()) {
    return false;
  }
  frame.metadata()["opencl_platform_active"] =
      std::string{resources.platform_name()};
  frame.metadata()["opencl_device_active"] =
      std::string{resources.device_name()};
  frame.metadata()["background_processing_device"] =
      std::string{resources.device_name()};

  auto mask = person_mask(frame);
  std::vector<std::uint8_t> fallback_mask{0U};
  auto mask_values = std::span<const std::uint8_t>{fallback_mask};
  auto mask_width_px = std::uint32_t{1U};
  auto mask_height_px = std::uint32_t{1U};
  auto active_mask_mode = mask_mode_;
  if (mask.has_value()) {
    mask_values = mask->values();
    mask_width_px = mask->width();
    mask_height_px = mask->height();
  } else if (mask_mode_ == "onnx") {
    active_mask_mode = fallback_mask_mode_;
  }

  cl_int error = CL_SUCCESS;
  cl_mem input = clCreateBuffer(resources.context(),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bytes.size(), bytes.data(), &error);
  if (error != CL_SUCCESS || input == nullptr) {
    return false;
  }
  cl_mem output = clCreateBuffer(resources.context(), CL_MEM_WRITE_ONLY,
                                 bytes.size(), nullptr, &error);
  if (error != CL_SUCCESS || output == nullptr) {
    clReleaseMemObject(input);
    return false;
  }
  cl_mem mask_buffer = clCreateBuffer(
      resources.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      mask_values.size(), const_cast<std::uint8_t *>(mask_values.data()),
      &error);
  if (error != CL_SUCCESS || mask_buffer == nullptr) {
    clReleaseMemObject(output);
    clReleaseMemObject(input);
    return false;
  }

  const auto width = static_cast<cl_uint>(frame.width());
  const auto height = static_cast<cl_uint>(frame.height());
  const auto stride = static_cast<cl_uint>(strides.front());
  const auto radius = static_cast<cl_uint>(radius_);
  const auto threshold = static_cast<cl_uchar>(foreground_threshold_);
  const auto brightness = static_cast<float>(brightness_);
  const auto contrast = static_cast<float>(contrast_);
  const auto saturation = static_cast<float>(saturation_);
  auto crop = Crop{0.0, 0.0, static_cast<double>(frame.width()),
                   static_cast<double>(frame.height())};
  if (auto_frame_) {
    auto bounds = std::optional<Bounds>{};
    if (mask.has_value()) {
      bounds = mask_bounds(*mask, foreground_threshold_);
    } else if (active_mask_mode != "tracked_center") {
      bounds = foreground_bounds(bytes, frame.format(), strides, frame.width(),
                                 frame.height(), foreground_threshold_);
    }
    if (bounds.has_value()) {
      if (mask.has_value()) {
        bounds = scale_bounds(*bounds, mask->width(), mask->height(),
                              frame.width(), frame.height());
      }
      crop = crop_from_bounds(*bounds, frame.width(), frame.height(),
                              target_fill_, max_zoom_);
    }
  }
  if (auto_frame_) {
    crop = smooth_crop(crop, previous_auto_frame_crop_, frame.width(),
                       frame.height());
  }
  const auto crop_width = static_cast<std::uint32_t>(std::clamp(
      static_cast<int>(std::round(crop.width)), 1, static_cast<int>(width)));
  const auto crop_height = static_cast<std::uint32_t>(std::clamp(
      static_cast<int>(std::round(crop.height)), 1, static_cast<int>(height)));
  const auto crop_x = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(crop.x)), 0,
                 static_cast<int>(frame.width() - crop_width)));
  const auto crop_y = static_cast<std::uint32_t>(
      std::clamp(static_cast<int>(std::round(crop.y)), 0,
                 static_cast<int>(frame.height() - crop_height)));
  frame.metadata()["background_blur_auto_frame_crop"] =
      std::to_string(crop_x) + "," + std::to_string(crop_y) + "," +
      std::to_string(crop_width) + "," + std::to_string(crop_height);
  const auto cl_crop_x = static_cast<cl_uint>(crop_x);
  const auto cl_crop_y = static_cast<cl_uint>(crop_y);
  const auto cl_crop_width = static_cast<cl_uint>(crop_width);
  const auto cl_crop_height = static_cast<cl_uint>(crop_height);
  const auto mask_mode = opencl_mask_mode(active_mask_mode);
  const auto mask_width_arg = static_cast<cl_uint>(mask_width_px);
  const auto mask_height_arg = static_cast<cl_uint>(mask_height_px);
  const auto mask_width = static_cast<float>(mask_width_);
  const auto mask_height = static_cast<float>(mask_height_);
  auto *kernel = resources.kernel();

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mask_buffer);
  error |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &width);
  error |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &height);
  error |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &stride);
  error |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layout.pixel_size);
  error |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &layout.red_offset);
  error |= clSetKernelArg(kernel, 8, sizeof(cl_uint), &layout.green_offset);
  error |= clSetKernelArg(kernel, 9, sizeof(cl_uint), &layout.blue_offset);
  error |= clSetKernelArg(kernel, 10, sizeof(cl_uint), &radius);
  error |= clSetKernelArg(kernel, 11, sizeof(cl_uchar), &threshold);
  error |= clSetKernelArg(kernel, 12, sizeof(float), &brightness);
  error |= clSetKernelArg(kernel, 13, sizeof(float), &contrast);
  error |= clSetKernelArg(kernel, 14, sizeof(float), &saturation);
  error |= clSetKernelArg(kernel, 15, sizeof(cl_uint), &cl_crop_x);
  error |= clSetKernelArg(kernel, 16, sizeof(cl_uint), &cl_crop_y);
  error |= clSetKernelArg(kernel, 17, sizeof(cl_uint), &cl_crop_width);
  error |= clSetKernelArg(kernel, 18, sizeof(cl_uint), &cl_crop_height);
  error |= clSetKernelArg(kernel, 19, sizeof(cl_uint), &mask_mode);
  error |= clSetKernelArg(kernel, 20, sizeof(cl_uint), &mask_width_arg);
  error |= clSetKernelArg(kernel, 21, sizeof(cl_uint), &mask_height_arg);
  error |= clSetKernelArg(kernel, 22, sizeof(float), &mask_width);
  error |= clSetKernelArg(kernel, 23, sizeof(float), &mask_height);

  const std::size_t global_work_size[] = {frame.width(), frame.height()};
  if (error == CL_SUCCESS) {
    error =
        clEnqueueNDRangeKernel(resources.queue(), kernel, 2, nullptr,
                               global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (error == CL_SUCCESS) {
    error =
        clEnqueueReadBuffer(resources.queue(), output, CL_TRUE, 0, bytes.size(),
                            bytes.data(), 0, nullptr, nullptr);
  }

  clReleaseMemObject(output);
  clReleaseMemObject(mask_buffer);
  clReleaseMemObject(input);
  if (error == CL_SUCCESS) {
    if (mask.has_value() &&
        !frame.metadata().contains("background_blur_mask")) {
      frame.metadata()["background_blur_mask"] = "onnx";
    } else if (!frame.metadata().contains("background_blur_mask")) {
      frame.metadata()["background_blur_mask"] = active_mask_mode;
    }
  }
  return error == CL_SUCCESS;
#else
  (void)frame;
  return false;
#endif
}

std::string_view BackgroundBlurFilter::type() const noexcept {
  return "background_blur";
}

} // namespace lmp::filters
