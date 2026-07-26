#pragma once

#include "lmp/capture/capture_source.hpp"

#include <cstdint>
#include <string>

namespace lmp::capture {

struct UdpEndpoint {
  std::string host;
  std::uint16_t port;
};

class GoProUdpSource final : public ICaptureSource {
public:
  explicit GoProUdpSource(std::string address);
  ~GoProUdpSource() override;

  GoProUdpSource(const GoProUdpSource &) = delete;
  GoProUdpSource &operator=(const GoProUdpSource &) = delete;
  GoProUdpSource(GoProUdpSource &&) = delete;
  GoProUdpSource &operator=(GoProUdpSource &&) = delete;

  void open() override;
  void close() noexcept override;
  [[nodiscard]] bool is_open() const noexcept override;
  [[nodiscard]] std::string_view type() const noexcept override;
  [[nodiscard]] const UdpEndpoint &endpoint() const noexcept;
  [[nodiscard]] int native_handle() const noexcept;

private:
  std::string address_;
  UdpEndpoint endpoint_;
  int socket_fd_;
};

[[nodiscard]] UdpEndpoint parse_udp_endpoint(std::string_view address);

} // namespace lmp::capture
