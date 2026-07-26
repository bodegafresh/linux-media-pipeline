#pragma once

#include "lmp/frame/frame.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace lmp::filters::detail {

struct PixelChannels {
  std::uint8_t &red;
  std::uint8_t &green;
  std::uint8_t &blue;
};

inline std::uint8_t clamp_to_byte(double value) noexcept {
  const auto rounded = static_cast<int>(value + 0.5);
  return static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
}

inline std::size_t packed_pixel_size(frame::PixelFormat format) {
  switch (format) {
  case frame::PixelFormat::Rgba:
    return 4U;
  case frame::PixelFormat::Rgb:
  case frame::PixelFormat::Bgr:
    return 3U;
  case frame::PixelFormat::Yuv420p:
  case frame::PixelFormat::Nv12:
  case frame::PixelFormat::P010:
    throw std::invalid_argument(
        "filter supports only RGB, BGR, and RGBA frames");
  }
  throw std::invalid_argument("unsupported pixel format");
}

inline PixelChannels channels(std::uint8_t *pixel, frame::PixelFormat format) {
  switch (format) {
  case frame::PixelFormat::Rgba:
  case frame::PixelFormat::Rgb:
    return PixelChannels{pixel[0], pixel[1], pixel[2]};
  case frame::PixelFormat::Bgr:
    return PixelChannels{pixel[2], pixel[1], pixel[0]};
  case frame::PixelFormat::Yuv420p:
  case frame::PixelFormat::Nv12:
  case frame::PixelFormat::P010:
    throw std::invalid_argument(
        "filter supports only RGB, BGR, and RGBA frames");
  }
  throw std::invalid_argument("unsupported pixel format");
}

template <typename Operation>
void for_each_packed_pixel(frame::Frame &frame, Operation operation) {
  const auto pixel_size = packed_pixel_size(frame.format());
  const auto row_bytes = static_cast<std::size_t>(frame.width()) * pixel_size;
  const auto strides = frame.strides();
  if (strides.empty() || strides.front() < row_bytes) {
    throw std::invalid_argument("frame stride is smaller than row size");
  }

  auto bytes = frame.data();
  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    const auto row_offset = static_cast<std::size_t>(y) * strides.front();
    if (row_offset + row_bytes > bytes.size()) {
      throw std::invalid_argument("frame data is smaller than stride layout");
    }
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      auto *pixel = bytes.data() + row_offset +
                    (static_cast<std::size_t>(x) * pixel_size);
      operation(channels(pixel, frame.format()));
    }
  }
}

} // namespace lmp::filters::detail
