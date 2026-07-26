#pragma once

#include "packed_pixel_filter.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lmp::filters::detail {

struct RgbPixel {
  std::uint8_t red;
  std::uint8_t green;
  std::uint8_t blue;
};

inline std::size_t pixel_index(std::uint32_t x, std::uint32_t y,
                               std::uint32_t width) noexcept {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
         static_cast<std::size_t>(x);
}

inline std::uint32_t clamp_coordinate(int value, std::uint32_t upper) noexcept {
  return static_cast<std::uint32_t>(
      std::clamp(value, 0, static_cast<int>(upper) - 1));
}

inline std::vector<RgbPixel> read_packed_rgb(const frame::Frame &frame) {
  const auto pixel_size = packed_pixel_size(frame.format());
  const auto row_bytes = static_cast<std::size_t>(frame.width()) * pixel_size;
  const auto strides = frame.strides();
  if (strides.empty() || strides.front() < row_bytes) {
    throw std::invalid_argument("frame stride is smaller than row size");
  }

  const auto bytes = frame.data();
  std::vector<RgbPixel> pixels;
  pixels.reserve(static_cast<std::size_t>(frame.width()) *
                 static_cast<std::size_t>(frame.height()));

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    const auto row_offset = static_cast<std::size_t>(y) * strides.front();
    if (row_offset + row_bytes > bytes.size()) {
      throw std::invalid_argument("frame data is smaller than stride layout");
    }
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      const auto *pixel = bytes.data() + row_offset +
                          (static_cast<std::size_t>(x) * pixel_size);
      if (frame.format() == frame::PixelFormat::Bgr) {
        pixels.push_back(RgbPixel{pixel[2], pixel[1], pixel[0]});
      } else {
        pixels.push_back(RgbPixel{pixel[0], pixel[1], pixel[2]});
      }
    }
  }
  return pixels;
}

inline void write_packed_rgb(frame::Frame &frame,
                             const std::vector<RgbPixel> &pixels) {
  auto index = std::size_t{0};
  for_each_packed_pixel(frame, [&](PixelChannels pixel) {
    pixel.red = pixels.at(index).red;
    pixel.green = pixels.at(index).green;
    pixel.blue = pixels.at(index).blue;
    ++index;
  });
}

inline void apply_kernel_3x3(frame::Frame &frame,
                             const std::array<double, 9> &kernel,
                             double divisor, double bias) {
  const auto source = read_packed_rgb(frame);
  std::vector<RgbPixel> output(source.size());

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      double red = 0.0;
      double green = 0.0;
      double blue = 0.0;
      auto kernel_index = std::size_t{0};

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          const auto sample_x =
              clamp_coordinate(static_cast<int>(x) + kx, frame.width());
          const auto sample_y =
              clamp_coordinate(static_cast<int>(y) + ky, frame.height());
          const auto sample =
              source[pixel_index(sample_x, sample_y, frame.width())];
          const auto weight = kernel[kernel_index];
          red += static_cast<double>(sample.red) * weight;
          green += static_cast<double>(sample.green) * weight;
          blue += static_cast<double>(sample.blue) * weight;
          ++kernel_index;
        }
      }

      output[pixel_index(x, y, frame.width())] =
          RgbPixel{clamp_to_byte((red / divisor) + bias),
                   clamp_to_byte((green / divisor) + bias),
                   clamp_to_byte((blue / divisor) + bias)};
    }
  }

  write_packed_rgb(frame, output);
}

inline void apply_box_blur(frame::Frame &frame, std::uint32_t radius) {
  const auto source = read_packed_rgb(frame);
  std::vector<RgbPixel> output(source.size());
  const auto signed_radius = static_cast<int>(radius);

  for (std::uint32_t y = 0; y < frame.height(); ++y) {
    for (std::uint32_t x = 0; x < frame.width(); ++x) {
      std::uint32_t count = 0;
      std::uint32_t red = 0;
      std::uint32_t green = 0;
      std::uint32_t blue = 0;

      for (int ky = -signed_radius; ky <= signed_radius; ++ky) {
        for (int kx = -signed_radius; kx <= signed_radius; ++kx) {
          const auto sample_x =
              clamp_coordinate(static_cast<int>(x) + kx, frame.width());
          const auto sample_y =
              clamp_coordinate(static_cast<int>(y) + ky, frame.height());
          const auto sample =
              source[pixel_index(sample_x, sample_y, frame.width())];
          red += sample.red;
          green += sample.green;
          blue += sample.blue;
          ++count;
        }
      }

      output[pixel_index(x, y, frame.width())] =
          RgbPixel{static_cast<std::uint8_t>((red + (count / 2U)) / count),
                   static_cast<std::uint8_t>((green + (count / 2U)) / count),
                   static_cast<std::uint8_t>((blue + (count / 2U)) / count)};
    }
  }

  write_packed_rgb(frame, output);
}

} // namespace lmp::filters::detail
