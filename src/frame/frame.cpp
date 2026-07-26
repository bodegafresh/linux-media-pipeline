#include "lmp/frame/frame.hpp"

#include <stdexcept>

namespace lmp::frame {

Frame::Frame(std::uint32_t width, std::uint32_t height, PixelFormat format,
             std::vector<std::uint8_t> data, std::vector<std::size_t> strides,
             Clock::time_point timestamp, Metadata metadata)
    : width_(width), height_(height), format_(format), data_(std::move(data)),
      strides_(std::move(strides)), timestamp_(timestamp),
      metadata_(std::move(metadata)) {
  if (width_ == 0 || height_ == 0) {
    throw std::invalid_argument("frame dimensions must be non-zero");
  }
  if (data_.size() < minimum_contiguous_size(width_, height_, format_)) {
    throw std::invalid_argument(
        "frame data is smaller than the pixel format requires");
  }
  if (strides_.empty()) {
    throw std::invalid_argument("frame requires at least one stride");
  }
}

std::uint32_t Frame::width() const noexcept { return width_; }

std::uint32_t Frame::height() const noexcept { return height_; }

PixelFormat Frame::format() const noexcept { return format_; }

Frame::Clock::time_point Frame::timestamp() const noexcept {
  return timestamp_;
}

const Frame::Metadata &Frame::metadata() const noexcept { return metadata_; }

Frame::Metadata &Frame::metadata() noexcept { return metadata_; }

std::span<std::uint8_t> Frame::data() noexcept { return data_; }

std::span<const std::uint8_t> Frame::data() const noexcept { return data_; }

std::span<const std::size_t> Frame::strides() const noexcept {
  return strides_;
}

std::size_t bytes_per_pixel(PixelFormat format) noexcept {
  switch (format) {
  case PixelFormat::Rgba:
    return 4;
  case PixelFormat::Rgb:
  case PixelFormat::Bgr:
    return 3;
  case PixelFormat::P010:
    return 2;
  case PixelFormat::Yuv420p:
  case PixelFormat::Nv12:
    return 1;
  }
  return 1;
}

std::size_t minimum_contiguous_size(std::uint32_t width, std::uint32_t height,
                                    PixelFormat format) noexcept {
  const auto pixels =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  switch (format) {
  case PixelFormat::Yuv420p:
  case PixelFormat::Nv12:
    return (pixels * 3U) / 2U;
  case PixelFormat::P010:
    return pixels * 3U;
  case PixelFormat::Rgba:
  case PixelFormat::Rgb:
  case PixelFormat::Bgr:
    return pixels * bytes_per_pixel(format);
  }
  return pixels;
}

} // namespace lmp::frame
