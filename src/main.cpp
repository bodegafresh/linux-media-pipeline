#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/frame/frame.hpp"
#include "lmp/output/v4l2_output.hpp"
#include "lmp/version.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

lmp::frame::Frame make_test_pattern(std::uint32_t width, std::uint32_t height,
                                    std::uint32_t frame_index) {
  std::vector<std::uint8_t> bytes;
  bytes.resize(static_cast<std::size_t>(width) * height * 3U);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto offset = ((static_cast<std::size_t>(y) * width) + x) * 3U;
      const auto bar = ((x + frame_index * 8U) / 80U) % 6U;
      bytes[offset] = bar == 0U || bar == 3U ? 255U : 32U;
      bytes[offset + 1U] = bar == 1U || bar == 3U || bar == 4U ? 255U : 32U;
      bytes[offset + 2U] = bar == 2U || bar == 4U || bar == 5U ? 255U : 32U;
      if (((x / 32U) + (y / 32U) + frame_index) % 2U == 0U) {
        bytes[offset] = static_cast<std::uint8_t>(bytes[offset] / 2U);
      }
    }
  }
  return lmp::frame::Frame{
      width,
      height,
      lmp::frame::PixelFormat::Rgb,
      std::move(bytes),
      std::vector<std::size_t>{static_cast<std::size_t>(width) * 3U},
      lmp::frame::Frame::Clock::now()};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const lmp::config::ConfigLoader loader;
    const auto config = loader.load_file("config/default.yaml");
    const auto registry = lmp::filters::create_default_registry();
    const auto pipeline =
        lmp::filters::FilterPipeline::from_config(config.filters, registry);

    auto open_capture = false;
    auto check_output = false;
    auto test_pattern = false;
    for (int index = 1; index < argc; ++index) {
      if (std::string_view{argv[index]} == "--open-capture") {
        open_capture = true;
      } else if (std::string_view{argv[index]} == "--check-output") {
        check_output = true;
      } else if (std::string_view{argv[index]} == "--test-pattern") {
        test_pattern = true;
      }
    }

    lmp::capture::GoProUdpSource capture{config.capture.address};
    if (config.capture.type != capture.type()) {
      throw std::runtime_error("unsupported capture type: " +
                               config.capture.type);
    }
    if (open_capture) {
      capture.open();
    }

    lmp::output::V4l2Output output{config.output.device};
    if (config.output.type != output.type()) {
      throw std::runtime_error("unsupported output type: " +
                               config.output.type);
    }
    if (check_output || test_pattern) {
      output.open();
    }
    if (test_pattern) {
      constexpr auto width = 1280U;
      constexpr auto height = 720U;
      constexpr auto fps = 30U;
      output.configure_rgb24(width, height, fps);
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " streaming test_pattern=true output=" << output.type()
                << " device=" << output.device() << " format=RGB24"
                << " width=" << width << " height=" << height << " fps=" << fps
                << '\n';
      for (std::uint32_t frame_index = 0;; ++frame_index) {
        auto frame = make_test_pattern(width, height, frame_index);
        pipeline.process(frame);
        output.write(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds{1000 / fps});
      }
    }

    std::cout << "linux-media-pipeline " << lmp::version_string()
              << " capture=" << capture.type()
              << " udp=" << capture.endpoint().host << ':'
              << capture.endpoint().port
              << " capture_open=" << (capture.is_open() ? "true" : "false")
              << " output=" << output.type() << " device=" << output.device()
              << " output_open=" << (output.is_open() ? "true" : "false")
              << " filters=" << pipeline.size() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linux-media-pipeline: " << error.what() << '\n';
    return 1;
  }
}
