#include "lmp/config/config_loader.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/filters/grayscale_filter.hpp"
#include "lmp/filters/negative_filter.hpp"
#include "lmp/filters/sepia_filter.hpp"
#include "lmp/frame/frame.hpp"
#include "lmp/version.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    return false;
  }
  return true;
}

lmp::frame::Frame make_rgba_frame(std::vector<std::uint8_t> data) {
  return lmp::frame::Frame{2U,
                           1U,
                           lmp::frame::PixelFormat::Rgba,
                           std::move(data),
                           std::vector<std::size_t>{8U},
                           lmp::frame::Frame::Clock::now()};
}

std::vector<std::uint8_t> bytes(const lmp::frame::Frame &frame) {
  return {frame.data().begin(), frame.data().end()};
}

} // namespace

int main() {
  const auto current = lmp::version();
  bool ok = true;
  ok = expect(current.major == 0, "major version") && ok;
  ok = expect(current.minor == 1, "minor version") && ok;
  ok = expect(current.patch == 0, "patch version") && ok;
  ok = expect(lmp::version_string() == "0.1.0", "version string") && ok;

  const lmp::config::ConfigLoader loader;
  const auto config = loader.load_file("config/default.yaml");
  ok = expect(config.capture.type == "gopro_udp", "capture type") && ok;
  ok = expect(config.capture.address == "udp://0.0.0.0:8554",
              "capture address") &&
       ok;
  ok = expect(config.gpu.backend == "opencl", "gpu backend") && ok;
  ok = expect(config.pipeline.threads == "auto", "pipeline threads") && ok;
  ok = expect(config.pipeline.queue_size == 4U, "pipeline queue size") && ok;
  ok = expect(config.filters.size() == 4U, "filter count") && ok;
  ok = expect(config.filters.front().type == "identity",
              "identity filter config") &&
       ok;
  ok = expect(config.filters.front().enabled, "identity filter enabled") && ok;
  ok = expect(config.output.type == "v4l2", "output type") && ok;
  ok = expect(config.output.device == "/dev/video20", "output device") && ok;

  auto frame = lmp::frame::Frame{
      2U,
      2U,
      lmp::frame::PixelFormat::Rgba,
      std::vector<std::uint8_t>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U,
                                12U, 13U, 14U, 15U, 16U},
      std::vector<std::size_t>{8U},
      lmp::frame::Frame::Clock::now(),
      {{"source", "test"}}};

  const auto before =
      std::vector<std::uint8_t>{frame.data().begin(), frame.data().end()};
  const auto registry = lmp::filters::create_default_registry();
  const auto pipeline =
      lmp::filters::FilterPipeline::from_config(config.filters, registry);
  pipeline.process(frame);

  const auto after =
      std::vector<std::uint8_t>{frame.data().begin(), frame.data().end()};
  ok = expect(registry.contains("identity"), "identity registered") && ok;
  ok = expect(registry.contains("grayscale"), "grayscale registered") && ok;
  ok = expect(registry.contains("negative"), "negative registered") && ok;
  ok = expect(registry.contains("sepia"), "sepia registered") && ok;
  ok = expect(registry.size() == 4U, "default registry size") && ok;
  ok = expect(pipeline.size() == 1U, "identity pipeline size") && ok;
  ok = expect(before == after, "identity keeps frame bytes unchanged") && ok;
  ok = expect(frame.metadata().at("source") == "test", "frame metadata") && ok;

  auto grayscale = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::GrayscaleFilter{}.process(grayscale);
  ok = expect(bytes(grayscale) == std::vector<std::uint8_t>{18U, 18U, 18U, 40U,
                                                            124U, 124U, 124U,
                                                            255U},
              "grayscale rgba") &&
       ok;

  auto negative = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::NegativeFilter{}.process(negative);
  ok = expect(bytes(negative) == std::vector<std::uint8_t>{245U, 235U, 225U,
                                                           40U, 55U, 155U, 205U,
                                                           255U},
              "negative rgba") &&
       ok;

  auto sepia = make_rgba_frame({10U, 20U, 30U, 40U, 200U, 100U, 50U, 255U});
  lmp::filters::SepiaFilter{}.process(sepia);
  ok = expect(bytes(sepia) == std::vector<std::uint8_t>{25U, 22U, 17U, 40U,
                                                        165U, 147U, 114U, 255U},
              "sepia rgba") &&
       ok;

  auto bgr = lmp::frame::Frame{1U,
                               1U,
                               lmp::frame::PixelFormat::Bgr,
                               std::vector<std::uint8_t>{30U, 20U, 10U},
                               std::vector<std::size_t>{3U},
                               lmp::frame::Frame::Clock::now()};
  lmp::filters::NegativeFilter{}.process(bgr);
  ok = expect(bytes(bgr) == std::vector<std::uint8_t>{225U, 235U, 245U},
              "negative bgr channel order") &&
       ok;
  return ok ? 0 : 1;
}
