#include "lmp/capture/gopro_udp_source.hpp"
#include "lmp/config/config_loader.hpp"
#include "lmp/decoder/ffmpeg_decoder.hpp"
#include "lmp/filters/filter_pipeline.hpp"
#include "lmp/filters/filter_registry.hpp"
#include "lmp/frame/frame.hpp"
#include "lmp/output/v4l2_output.hpp"
#include "lmp/version.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void print_help() {
  std::cout
      << "linux-media-pipeline options:\n"
      << "  --help           Show this help.\n"
      << "  --config PATH    Load YAML config. Default: config/default.yaml.\n"
      << "  --open-capture   Bind the configured GoPro UDP listener.\n"
      << "  --check-output   Open the configured V4L2 output device.\n"
      << "  --stream-live    Decode configured capture with FFmpeg and stream "
         "to V4L2.\n"
      << "  --test-pattern   Stream a live RGB test pattern to V4L2 for OBS.\n";
}

std::string
active_filter_list(const std::vector<lmp::config::FilterConfig> &filters) {
  std::string result = "[";
  auto first = true;
  for (const auto &filter : filters) {
    if (!filter.enabled) {
      continue;
    }
    if (!first) {
      result += ",";
    }
    result += filter.type;
    first = false;
  }
  result += "]";
  return result;
}

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
    auto config_path = std::string{"config/default.yaml"};
    auto open_capture = false;
    auto check_output = false;
    auto stream_live = false;
    auto test_pattern = false;
    for (int index = 1; index < argc; ++index) {
      const auto option = std::string_view{argv[index]};
      if (option == "--help") {
        print_help();
        return 0;
      }
      if (option == "--config") {
        ++index;
        if (index >= argc) {
          throw std::runtime_error("--config requires a path");
        }
        config_path = argv[index];
      } else if (option == "--open-capture") {
        open_capture = true;
      } else if (option == "--check-output") {
        check_output = true;
      } else if (option == "--stream-live") {
        stream_live = true;
      } else if (option == "--test-pattern") {
        test_pattern = true;
      } else {
        throw std::runtime_error("unknown option: " + std::string{option});
      }
    }

    const lmp::config::ConfigLoader loader;
    const auto config = loader.load_file(config_path);
    const auto registry = lmp::filters::create_default_registry();
    const auto pipeline =
        lmp::filters::FilterPipeline::from_config(config.filters, registry);
    const auto filters_active = active_filter_list(config.filters);

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
    if (check_output || test_pattern || stream_live) {
      output.open();
    }
    if (stream_live) {
      if (config.output.pixel_format != "RGB24") {
        throw std::runtime_error("live output requires RGB24 pixel_format");
      }
      const auto width = static_cast<std::uint32_t>(config.output.width);
      const auto height = static_cast<std::uint32_t>(config.output.height);
      const auto fps = static_cast<std::uint32_t>(config.output.fps);
      output.configure_rgb24(width, height, fps);
      lmp::decoder::FfmpegDecoder decoder{config.capture.address, width,
                                          height};
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " streaming live=true capture=" << capture.type()
                << " input=" << config.capture.address
                << " output=" << output.type() << " device=" << output.device()
                << " format=" << config.output.pixel_format
                << " width=" << width << " height=" << height << " fps=" << fps
                << " filter_backend=requested:" << config.gpu.backend
                << " filters=" << pipeline.size()
                << " filters_active=" << filters_active << '\n';
      bool reported_filter_backend = false;
      bool reported_background_blur_backend = false;
      while (true) {
        auto frame = decoder.read_frame();
        pipeline.process(frame);
        if (!reported_filter_backend) {
          if (const auto found = frame.metadata().find("filter_backend");
              found != frame.metadata().end()) {
            std::cout << "filter_backend_active=" << found->second << '\n';
            reported_filter_backend = true;
          }
        }
        if (!reported_background_blur_backend) {
          if (const auto found =
                  frame.metadata().find("background_blur_backend");
              found != frame.metadata().end()) {
            std::cout << "background_blur_backend_active=" << found->second
                      << '\n';
            reported_background_blur_backend = true;
          }
        }
        output.write(frame);
      }
    }
    if (test_pattern) {
      if (config.output.pixel_format != "RGB24") {
        throw std::runtime_error(
            "test pattern output requires RGB24 pixel_format");
      }
      const auto width = static_cast<std::uint32_t>(config.output.width);
      const auto height = static_cast<std::uint32_t>(config.output.height);
      const auto fps = static_cast<std::uint32_t>(config.output.fps);
      output.configure_rgb24(width, height, fps);
      std::cout << "linux-media-pipeline " << lmp::version_string()
                << " streaming test_pattern=true output=" << output.type()
                << " device=" << output.device()
                << " format=" << config.output.pixel_format
                << " width=" << width << " height=" << height << " fps=" << fps
                << " filter_backend=requested:" << config.gpu.backend
                << " filters=" << pipeline.size()
                << " filters_active=" << filters_active << '\n';
      bool reported_filter_backend = false;
      bool reported_background_blur_backend = false;
      for (std::uint32_t frame_index = 0;; ++frame_index) {
        auto frame = make_test_pattern(width, height, frame_index);
        pipeline.process(frame);
        if (!reported_filter_backend) {
          if (const auto found = frame.metadata().find("filter_backend");
              found != frame.metadata().end()) {
            std::cout << "filter_backend_active=" << found->second << '\n';
            reported_filter_backend = true;
          }
        }
        if (!reported_background_blur_backend) {
          if (const auto found =
                  frame.metadata().find("background_blur_backend");
              found != frame.metadata().end()) {
            std::cout << "background_blur_backend_active=" << found->second
                      << '\n';
            reported_background_blur_backend = true;
          }
        }
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
              << " filter_backend=requested:" << config.gpu.backend
              << " filters=" << pipeline.size()
              << " filters_active=" << filters_active << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "linux-media-pipeline: " << error.what() << '\n';
    return 1;
  }
}
