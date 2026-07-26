#pragma once

#include "lmp/frame/frame.hpp"

#include <string_view>

namespace lmp::output {

class IOutputSink {
public:
  virtual ~IOutputSink() = default;

  virtual void open() = 0;
  virtual void close() noexcept = 0;
  virtual void write(const frame::Frame &frame) = 0;
  [[nodiscard]] virtual bool is_open() const noexcept = 0;
  [[nodiscard]] virtual std::string_view type() const noexcept = 0;
};

} // namespace lmp::output
