#pragma once

#include "packed_pixel_filter.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <string_view>

namespace lmp::filters::detail {

struct RgbColor {
  std::uint8_t red;
  std::uint8_t green;
  std::uint8_t blue;
};

inline void set_pixel(frame::Frame &frame, std::uint32_t x, std::uint32_t y,
                      RgbColor color) {
  if (x >= frame.width() || y >= frame.height()) {
    return;
  }
  const auto pixel_size = packed_pixel_size(frame.format());
  const auto row_bytes = static_cast<std::size_t>(frame.width()) * pixel_size;
  const auto strides = frame.strides();
  if (strides.empty() || strides.front() < row_bytes) {
    throw std::invalid_argument("frame stride is smaller than row size");
  }
  auto data = frame.data();
  const auto offset = (static_cast<std::size_t>(y) * strides.front()) +
                      (static_cast<std::size_t>(x) * pixel_size);
  if (offset + 2U >= data.size()) {
    throw std::invalid_argument("frame data is smaller than stride layout");
  }
  auto pixel = channels(data.data() + offset, frame.format());
  pixel.red = color.red;
  pixel.green = color.green;
  pixel.blue = color.blue;
}

inline std::array<std::uint8_t, 5> glyph(char input) noexcept {
  const auto c =
      static_cast<char>(std::toupper(static_cast<unsigned char>(input)));
  switch (c) {
  case '0':
    return {0b111, 0b101, 0b101, 0b101, 0b111};
  case '1':
    return {0b010, 0b110, 0b010, 0b010, 0b111};
  case '2':
    return {0b111, 0b001, 0b111, 0b100, 0b111};
  case '3':
    return {0b111, 0b001, 0b111, 0b001, 0b111};
  case '4':
    return {0b101, 0b101, 0b111, 0b001, 0b001};
  case '5':
    return {0b111, 0b100, 0b111, 0b001, 0b111};
  case '6':
    return {0b111, 0b100, 0b111, 0b101, 0b111};
  case '7':
    return {0b111, 0b001, 0b010, 0b010, 0b010};
  case '8':
    return {0b111, 0b101, 0b111, 0b101, 0b111};
  case '9':
    return {0b111, 0b101, 0b111, 0b001, 0b111};
  case 'A':
    return {0b010, 0b101, 0b111, 0b101, 0b101};
  case 'F':
    return {0b111, 0b100, 0b110, 0b100, 0b100};
  case 'M':
    return {0b101, 0b111, 0b111, 0b101, 0b101};
  case 'P':
    return {0b110, 0b101, 0b110, 0b100, 0b100};
  case 'S':
    return {0b111, 0b100, 0b111, 0b001, 0b111};
  case 'T':
    return {0b111, 0b010, 0b010, 0b010, 0b010};
  case ':':
    return {0b000, 0b010, 0b000, 0b010, 0b000};
  case '.':
    return {0b000, 0b000, 0b000, 0b000, 0b010};
  case '-':
    return {0b000, 0b000, 0b111, 0b000, 0b000};
  case ' ':
    return {0b000, 0b000, 0b000, 0b000, 0b000};
  default:
    return {0b111, 0b101, 0b101, 0b101, 0b111};
  }
}

inline void draw_text(frame::Frame &frame, std::uint32_t x, std::uint32_t y,
                      std::string_view text, RgbColor color) {
  auto pen_x = x;
  for (const auto c : text) {
    const auto bitmap = glyph(c);
    for (std::uint32_t row = 0; row < bitmap.size(); ++row) {
      for (std::uint32_t col = 0; col < 3U; ++col) {
        const auto mask = static_cast<std::uint8_t>(1U << (2U - col));
        if ((bitmap[row] & mask) != 0U) {
          set_pixel(frame, pen_x + col, y + row, color);
        }
      }
    }
    pen_x += 4U;
  }
}

} // namespace lmp::filters::detail
