#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lmp::frame {

enum class PixelFormat {
  Rgba,
  Rgb,
  Bgr,
  Yuv420p,
  Nv12,
  P010,
};

class Frame {
public:
  using Clock = std::chrono::steady_clock;
  using Metadata = std::unordered_map<std::string, std::string>;

  Frame(std::uint32_t width, std::uint32_t height, PixelFormat format,
        std::vector<std::uint8_t> data, std::vector<std::size_t> strides,
        Clock::time_point timestamp, Metadata metadata = {});

  [[nodiscard]] std::uint32_t width() const noexcept;
  [[nodiscard]] std::uint32_t height() const noexcept;
  [[nodiscard]] PixelFormat format() const noexcept;
  [[nodiscard]] Clock::time_point timestamp() const noexcept;
  [[nodiscard]] const Metadata &metadata() const noexcept;
  [[nodiscard]] Metadata &metadata() noexcept;
  [[nodiscard]] std::span<std::uint8_t> data() noexcept;
  [[nodiscard]] std::span<const std::uint8_t> data() const noexcept;
  [[nodiscard]] std::span<const std::size_t> strides() const noexcept;

private:
  std::uint32_t width_;
  std::uint32_t height_;
  PixelFormat format_;
  std::vector<std::uint8_t> data_;
  std::vector<std::size_t> strides_;
  Clock::time_point timestamp_;
  Metadata metadata_;
};

[[nodiscard]] std::size_t bytes_per_pixel(PixelFormat format) noexcept;
[[nodiscard]] std::size_t minimum_contiguous_size(std::uint32_t width,
                                                  std::uint32_t height,
                                                  PixelFormat format) noexcept;

} // namespace lmp::frame
