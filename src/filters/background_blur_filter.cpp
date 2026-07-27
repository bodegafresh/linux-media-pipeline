#include "lmp/filters/background_blur_filter.hpp"

#include "lmp/ai/onnx_runtime_engine.hpp"

#include "spatial_filter.hpp"

#include <cstddef>
#include <cstdint>
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
                              const float saturation) {
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);
  if (x >= width || y >= height) {
    return;
  }

  const uint base = (y * stride) + (x * pixel_size);
  const float red = (float)input[base + red_offset];
  const float green = (float)input[base + green_offset];
  const float blue = (float)input[base + blue_offset];
  const uchar luma = convert_uchar_sat_rte((0.299f * red) + (0.587f * green) + (0.114f * blue));

  float out_red = red;
  float out_green = green;
  float out_blue = blue;

  if (luma < foreground_threshold) {
    uint count = 0;
    uint red_sum = 0;
    uint green_sum = 0;
    uint blue_sum = 0;
    const int signed_radius = (int)radius;
    for (int ky = -signed_radius; ky <= signed_radius; ++ky) {
      const int sy = clamp((int)y + ky, 0, (int)height - 1);
      for (int kx = -signed_radius; kx <= signed_radius; ++kx) {
        const int sx = clamp((int)x + kx, 0, (int)width - 1);
        const uint sample = (((uint)sy) * stride) + (((uint)sx) * pixel_size);
        red_sum += input[sample + red_offset];
        green_sum += input[sample + green_offset];
        blue_sum += input[sample + blue_offset];
        ++count;
      }
    }
    out_red = (float)((red_sum + (count / 2U)) / count);
    out_green = (float)((green_sum + (count / 2U)) / count);
    out_blue = (float)((blue_sum + (count / 2U)) / count);
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
    output[base + 3U] = input[base + 3U];
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
  }

  OpenClBackgroundBlurResources(const OpenClBackgroundBlurResources &) = delete;
  OpenClBackgroundBlurResources &
  operator=(const OpenClBackgroundBlurResources &) = delete;

  ~OpenClBackgroundBlurResources() { reset(); }

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] cl_context context() const noexcept { return context_; }
  [[nodiscard]] cl_command_queue queue() const noexcept { return queue_; }
  [[nodiscard]] cl_kernel kernel() const noexcept { return kernel_; }

private:
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
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  bool ready_ = false;
};

OpenClBackgroundBlurResources &opencl_background_resources() {
  static OpenClBackgroundBlurResources resources;
  return resources;
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
    : radius_(radius), foreground_threshold_(foreground_threshold),
      backend_(std::move(backend)), brightness_(brightness),
      contrast_(contrast), saturation_(saturation) {
  if (radius_ == 0U) {
    throw std::invalid_argument("background blur radius must be >= 1");
  }
  if (contrast_ < 0.0) {
    throw std::invalid_argument("background_blur contrast must be >= 0");
  }
  if (saturation_ < 0.0) {
    throw std::invalid_argument("background_blur saturation must be >= 0");
  }
}

void BackgroundBlurFilter::process(frame::Frame &frame) const {
  if (backend_ == "opencl" && process_opencl(frame)) {
    frame.metadata()["background_blur_backend"] = "opencl";
    return;
  }
  process_cpu(frame);
  frame.metadata()["background_blur_backend"] = "cpu";
}

void BackgroundBlurFilter::process_cpu(frame::Frame &frame) const {
  const auto original = detail::read_packed_rgb(frame);
  ai::OnnxRuntimeEngine fallback_engine{""};
  const auto mask = fallback_engine.segment_person(frame);

  detail::apply_box_blur(frame, radius_);
  const auto blurred = detail::read_packed_rgb(frame);
  std::vector<detail::RgbPixel> output;
  output.reserve(original.size());

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto index = detail::pixel_index(x, y, frame.width());
      auto pixel = mask.at(x, y) >= foreground_threshold_ ? original[index]
                                                          : blurred[index];
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

  const auto width = static_cast<cl_uint>(frame.width());
  const auto height = static_cast<cl_uint>(frame.height());
  const auto stride = static_cast<cl_uint>(strides.front());
  const auto radius = static_cast<cl_uint>(radius_);
  const auto threshold = static_cast<cl_uchar>(foreground_threshold_);
  const auto brightness = static_cast<float>(brightness_);
  const auto contrast = static_cast<float>(contrast_);
  const auto saturation = static_cast<float>(saturation_);
  auto *kernel = resources.kernel();

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  error |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
  error |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
  error |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &stride);
  error |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &layout.pixel_size);
  error |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layout.red_offset);
  error |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &layout.green_offset);
  error |= clSetKernelArg(kernel, 8, sizeof(cl_uint), &layout.blue_offset);
  error |= clSetKernelArg(kernel, 9, sizeof(cl_uint), &radius);
  error |= clSetKernelArg(kernel, 10, sizeof(cl_uchar), &threshold);
  error |= clSetKernelArg(kernel, 11, sizeof(float), &brightness);
  error |= clSetKernelArg(kernel, 12, sizeof(float), &contrast);
  error |= clSetKernelArg(kernel, 13, sizeof(float), &saturation);

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
  clReleaseMemObject(input);
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
