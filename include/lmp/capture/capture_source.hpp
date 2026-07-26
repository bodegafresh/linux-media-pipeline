#pragma once

#include <string_view>

namespace lmp::capture {

class ICaptureSource {
public:
  virtual ~ICaptureSource() = default;

  virtual void open() = 0;
  virtual void close() noexcept = 0;
  [[nodiscard]] virtual bool is_open() const noexcept = 0;
  [[nodiscard]] virtual std::string_view type() const noexcept = 0;
};

} // namespace lmp::capture
