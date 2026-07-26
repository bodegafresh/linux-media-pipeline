#include "lmp/output/v4l2_output.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace lmp::output {
namespace {

constexpr auto invalid_fd = -1;

} // namespace

V4l2Output::V4l2Output(std::string device)
    : device_(std::move(device)), fd_(invalid_fd) {
  if (device_.empty()) {
    throw std::invalid_argument("v4l2 device cannot be empty");
  }
}

V4l2Output::~V4l2Output() { close(); }

void V4l2Output::open() {
  if (is_open()) {
    return;
  }
  const auto fd = ::open(device_.c_str(), O_WRONLY | O_NONBLOCK);
  if (fd == invalid_fd) {
    throw std::runtime_error("cannot open V4L2 output device " + device_ +
                             ": " + std::strerror(errno));
  }
  fd_ = fd;
}

void V4l2Output::close() noexcept {
  if (fd_ != invalid_fd) {
    ::close(fd_);
    fd_ = invalid_fd;
  }
}

void V4l2Output::write(const frame::Frame &frame) {
  if (!is_open()) {
    throw std::runtime_error("V4L2 output must be open before writing frames");
  }
  const auto bytes = frame.data();
  const auto written = ::write(fd_, bytes.data(), bytes.size());
  if (written < 0) {
    throw std::runtime_error("cannot write V4L2 frame to " + device_ + ": " +
                             std::strerror(errno));
  }
  if (static_cast<std::size_t>(written) != bytes.size()) {
    throw std::runtime_error("partial V4L2 frame write");
  }
}

bool V4l2Output::is_open() const noexcept { return fd_ != invalid_fd; }

std::string_view V4l2Output::type() const noexcept { return "v4l2"; }

std::string_view V4l2Output::device() const noexcept { return device_; }

int V4l2Output::native_handle() const noexcept { return fd_; }

} // namespace lmp::output
