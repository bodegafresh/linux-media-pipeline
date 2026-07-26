#pragma once

#include "lmp/output/output_sink.hpp"

#include <string>

namespace lmp::output {

class V4l2Output final : public IOutputSink {
public:
  explicit V4l2Output(std::string device);
  ~V4l2Output() override;

  V4l2Output(const V4l2Output &) = delete;
  V4l2Output &operator=(const V4l2Output &) = delete;
  V4l2Output(V4l2Output &&) = delete;
  V4l2Output &operator=(V4l2Output &&) = delete;

  void open() override;
  void close() noexcept override;
  void configure_rgb24(std::uint32_t width, std::uint32_t height,
                       std::uint32_t fps);
  void write(const frame::Frame &frame) override;
  [[nodiscard]] bool is_open() const noexcept override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] std::string_view device() const noexcept;
  [[nodiscard]] int native_handle() const noexcept;

private:
  std::string device_;
  int fd_;
};

} // namespace lmp::output
