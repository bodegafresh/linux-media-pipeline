#include "lmp/capture/gopro_udp_source.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace lmp::capture {
namespace {

constexpr auto invalid_socket = -1;

} // namespace

UdpEndpoint parse_udp_endpoint(std::string_view address) {
  constexpr auto prefix = std::string_view{"udp://"};
  if (address.substr(0, prefix.size()) != prefix) {
    throw std::invalid_argument("GoPro UDP address must start with udp://");
  }

  const auto endpoint = address.substr(prefix.size());
  const auto separator = endpoint.rfind(':');
  if (separator == std::string_view::npos || separator == 0U ||
      separator == endpoint.size() - 1U) {
    throw std::invalid_argument("GoPro UDP address must be udp://host:port");
  }

  auto host = std::string{endpoint.substr(0, separator)};
  const auto port_text = endpoint.substr(separator + 1U);
  int parsed_port = 0;
  const auto result = std::from_chars(
      port_text.data(), port_text.data() + port_text.size(), parsed_port);
  if (result.ec != std::errc{} ||
      result.ptr != port_text.data() + port_text.size() || parsed_port <= 0 ||
      parsed_port > 65535) {
    throw std::invalid_argument("GoPro UDP port must be in range 1-65535");
  }

  return UdpEndpoint{std::move(host), static_cast<std::uint16_t>(parsed_port)};
}

GoProUdpSource::GoProUdpSource(std::string address)
    : address_(std::move(address)), endpoint_(parse_udp_endpoint(address_)),
      socket_fd_(invalid_socket) {}

GoProUdpSource::~GoProUdpSource() { close(); }

void GoProUdpSource::open() {
  if (is_open()) {
    return;
  }

  const auto fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd == invalid_socket) {
    throw std::runtime_error(std::string{"cannot create UDP socket: "} +
                             std::strerror(errno));
  }

  sockaddr_in bind_address{};
  bind_address.sin_family = AF_INET;
  bind_address.sin_port = htons(endpoint_.port);
  if (::inet_pton(AF_INET, endpoint_.host.c_str(), &bind_address.sin_addr) !=
      1) {
    ::close(fd);
    throw std::invalid_argument("GoPro UDP host must be an IPv4 address");
  }

  if (::bind(fd, reinterpret_cast<sockaddr *>(&bind_address),
             sizeof(bind_address)) != 0) {
    const auto message =
        std::string{"cannot bind GoPro UDP socket: "} + std::strerror(errno);
    ::close(fd);
    throw std::runtime_error(message);
  }

  socket_fd_ = fd;
}

void GoProUdpSource::close() noexcept {
  if (socket_fd_ != invalid_socket) {
    ::close(socket_fd_);
    socket_fd_ = invalid_socket;
  }
}

bool GoProUdpSource::is_open() const noexcept {
  return socket_fd_ != invalid_socket;
}

std::string_view GoProUdpSource::type() const noexcept { return "gopro_udp"; }

const UdpEndpoint &GoProUdpSource::endpoint() const noexcept {
  return endpoint_;
}

int GoProUdpSource::native_handle() const noexcept { return socket_fd_; }

} // namespace lmp::capture
