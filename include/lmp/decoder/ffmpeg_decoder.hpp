#pragma once

#include "lmp/frame/frame.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace lmp::decoder {

class FfmpegDecoder {
public:
  FfmpegDecoder(std::string input_url, std::uint32_t width,
                std::uint32_t height);
  ~FfmpegDecoder();

  FfmpegDecoder(const FfmpegDecoder &) = delete;
  FfmpegDecoder &operator=(const FfmpegDecoder &) = delete;
  FfmpegDecoder(FfmpegDecoder &&) noexcept;
  FfmpegDecoder &operator=(FfmpegDecoder &&) noexcept;

  [[nodiscard]] frame::Frame read_frame();
  [[nodiscard]] bool is_open() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace lmp::decoder
