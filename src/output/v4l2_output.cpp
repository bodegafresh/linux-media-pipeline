#include "lmp/output/v4l2_output.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#ifdef __linux__
#include <linux/videodev2.h>
#endif
#include <stdexcept>
#include <string>
#ifdef __linux__
#include <sys/ioctl.h>
#endif
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
                             ": " + std::strerror(errno) +
                             ". Run ./scripts/setup-loopback.sh 20 "
                             "linux-media-pipeline and verify v4l2-ctl "
                             "--list-devices.");
  }
  fd_ = fd;
}

void V4l2Output::close() noexcept {
  if (fd_ != invalid_fd) {
    ::close(fd_);
    fd_ = invalid_fd;
  }
}

void V4l2Output::configure_rgb24(std::uint32_t width, std::uint32_t height,
                                 std::uint32_t fps) {
  if (!is_open()) {
    throw std::runtime_error("V4L2 output must be open before configuration");
  }
  if (width == 0U || height == 0U || fps == 0U) {
    throw std::invalid_argument("V4L2 width, height, and fps must be non-zero");
  }

#ifdef __linux__
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  format.fmt.pix.width = width;
  format.fmt.pix.height = height;
  format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  format.fmt.pix.field = V4L2_FIELD_NONE;
  format.fmt.pix.bytesperline = width * 3U;
  format.fmt.pix.sizeimage = format.fmt.pix.bytesperline * height;
  if (::ioctl(fd_, VIDIOC_S_FMT, &format) != 0) {
    throw std::runtime_error("cannot configure V4L2 RGB24 format on " +
                             device_ + ": " + std::strerror(errno));
  }

  v4l2_streamparm parameters{};
  parameters.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  parameters.parm.output.timeperframe.numerator = 1U;
  parameters.parm.output.timeperframe.denominator = fps;
  static_cast<void>(::ioctl(fd_, VIDIOC_S_PARM, &parameters));
#else
  static_cast<void>(width);
  static_cast<void>(height);
  static_cast<void>(fps);
  throw std::runtime_error(
      "V4L2 format configuration is only supported on Linux");
#endif
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
