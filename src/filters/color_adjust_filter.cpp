#include "lmp/filters/color_adjust_filter.hpp"

#include "packed_pixel_filter.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#if LMP_HAS_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

namespace lmp::filters {
namespace {

#if LMP_HAS_OPENCL
const char *color_adjust_kernel_source() {
  return R"CLC(
__kernel void color_adjust(__global uchar *data,
                           const uint width,
                           const uint height,
                           const uint stride,
                           const uint pixel_size,
                           const uint red_offset,
                           const uint green_offset,
                           const uint blue_offset,
                           const float brightness,
                           const float contrast,
                           const float saturation) {
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);
  if (x >= width || y >= height) {
    return;
  }

  const uint base = (y * stride) + (x * pixel_size);
  float red = (((float)data[base + red_offset] - 128.0f) * contrast) + 128.0f + brightness;
  float green = (((float)data[base + green_offset] - 128.0f) * contrast) + 128.0f + brightness;
  float blue = (((float)data[base + blue_offset] - 128.0f) * contrast) + 128.0f + brightness;

  const float gray = (0.299f * red) + (0.587f * green) + (0.114f * blue);
  red = gray + ((red - gray) * saturation);
  green = gray + ((green - gray) * saturation);
  blue = gray + ((blue - gray) * saturation);

  data[base + red_offset] = convert_uchar_sat_rte(red);
  data[base + green_offset] = convert_uchar_sat_rte(green);
  data[base + blue_offset] = convert_uchar_sat_rte(blue);
}
)CLC";
}

class OpenClColorAdjustResources {
public:
  OpenClColorAdjustResources() {
    cl_platform_id platform = nullptr;
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(1, &platform, &platform_count) != CL_SUCCESS ||
        platform_count == 0U) {
      return;
    }

    cl_uint device_count = 0;
    auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_,
                                 &device_count);
    if (status != CL_SUCCESS || device_count == 0U) {
      status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device_,
                              &device_count);
      if (status != CL_SUCCESS || device_count == 0U) {
        return;
      }
    }

    cl_int error = CL_SUCCESS;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    if (error != CL_SUCCESS || context_ == nullptr) {
      reset();
      return;
    }

    queue_ = clCreateCommandQueue(context_, device_, 0, &error);
    if (error != CL_SUCCESS || queue_ == nullptr) {
      reset();
      return;
    }

    const char *source = color_adjust_kernel_source();
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

    kernel_ = clCreateKernel(program_, "color_adjust", &error);
    if (error != CL_SUCCESS || kernel_ == nullptr) {
      reset();
      return;
    }

    ready_ = true;
  }

  OpenClColorAdjustResources(const OpenClColorAdjustResources &) = delete;
  OpenClColorAdjustResources &
  operator=(const OpenClColorAdjustResources &) = delete;

  ~OpenClColorAdjustResources() { reset(); }

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

OpenClColorAdjustResources &opencl_resources() {
  static OpenClColorAdjustResources resources;
  return resources;
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
#endif

} // namespace

ColorAdjustFilter::ColorAdjustFilter(double brightness, double contrast,
                                     double saturation)
    : ColorAdjustFilter(brightness, contrast, saturation, "cpu") {}

ColorAdjustFilter::ColorAdjustFilter(double brightness, double contrast,
                                     double saturation, std::string backend)
    : brightness_(brightness), contrast_(contrast), saturation_(saturation),
      backend_(std::move(backend)) {
  if (contrast_ < 0.0) {
    throw std::invalid_argument("color_adjust contrast must be >= 0");
  }
  if (saturation_ < 0.0) {
    throw std::invalid_argument("color_adjust saturation must be >= 0");
  }
}

void ColorAdjustFilter::process(frame::Frame &frame) const {
  if (backend_ == "opencl" && process_opencl(frame)) {
    frame.metadata()["filter_backend"] = "opencl";
    return;
  }
  process_cpu(frame);
  frame.metadata()["filter_backend"] = "cpu";
}

void ColorAdjustFilter::process_cpu(frame::Frame &frame) const {
  detail::for_each_packed_pixel(frame, [&](detail::PixelChannels pixel) {
    auto red = ((pixel.red - 128.0) * contrast_) + 128.0 + brightness_;
    auto green = ((pixel.green - 128.0) * contrast_) + 128.0 + brightness_;
    auto blue = ((pixel.blue - 128.0) * contrast_) + 128.0 + brightness_;

    const auto gray = (0.299 * red) + (0.587 * green) + (0.114 * blue);
    red = gray + ((red - gray) * saturation_);
    green = gray + ((green - gray) * saturation_);
    blue = gray + ((blue - gray) * saturation_);

    pixel.red = detail::clamp_to_byte(red);
    pixel.green = detail::clamp_to_byte(green);
    pixel.blue = detail::clamp_to_byte(blue);
  });
}

bool ColorAdjustFilter::process_opencl(frame::Frame &frame) const {
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

  auto &resources = opencl_resources();
  if (!resources.ready()) {
    return false;
  }

  cl_int error = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(resources.context(),
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 bytes.size(), bytes.data(), &error);
  if (error != CL_SUCCESS || buffer == nullptr) {
    if (buffer != nullptr) {
      clReleaseMemObject(buffer);
    }
    return false;
  }

  const auto width = static_cast<cl_uint>(frame.width());
  const auto height = static_cast<cl_uint>(frame.height());
  const auto stride = static_cast<cl_uint>(strides.front());
  const auto brightness = static_cast<float>(brightness_);
  const auto contrast = static_cast<float>(contrast_);
  const auto saturation = static_cast<float>(saturation_);
  auto *kernel = resources.kernel();

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
  error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &width);
  error |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &height);
  error |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &stride);
  error |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &layout.pixel_size);
  error |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &layout.red_offset);
  error |= clSetKernelArg(kernel, 6, sizeof(cl_uint), &layout.green_offset);
  error |= clSetKernelArg(kernel, 7, sizeof(cl_uint), &layout.blue_offset);
  error |= clSetKernelArg(kernel, 8, sizeof(float), &brightness);
  error |= clSetKernelArg(kernel, 9, sizeof(float), &contrast);
  error |= clSetKernelArg(kernel, 10, sizeof(float), &saturation);

  const std::size_t global_work_size[] = {frame.width(), frame.height()};
  if (error == CL_SUCCESS) {
    error =
        clEnqueueNDRangeKernel(resources.queue(), kernel, 2, nullptr,
                               global_work_size, nullptr, 0, nullptr, nullptr);
  }
  if (error == CL_SUCCESS) {
    error =
        clEnqueueReadBuffer(resources.queue(), buffer, CL_TRUE, 0, bytes.size(),
                            bytes.data(), 0, nullptr, nullptr);
  }

  clReleaseMemObject(buffer);
  return error == CL_SUCCESS;
#else
  (void)frame;
  return false;
#endif
}

std::string_view ColorAdjustFilter::type() const noexcept {
  return "color_adjust";
}

} // namespace lmp::filters
